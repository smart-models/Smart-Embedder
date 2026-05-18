import asyncio
import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


SERVER_PATH = Path(__file__).with_name("bge-m3_server.py")


class FakeBGEM3FlagModel:
    def __init__(self, *args, **kwargs):
        pass

    def encode(self, *args, **kwargs):
        return {"dense_vecs": []}


class FakeFlagReranker:
    calls = []

    def __init__(self, model_name, use_fp16=False):
        self.model_name = model_name
        self.use_fp16 = use_fp16

    def compute_score(self, pairs, normalize=False):
        self.calls.append((pairs, normalize))
        return [0.25 for _ in pairs]


class FakeTokenizer:
    def __init__(self):
        self.encoded = []
        self.pad_token = None
        self.eos_token = "<eos>"

    def convert_tokens_to_ids(self, token):
        return {"no": 0, "yes": 1}[token]

    def encode(self, text, add_special_tokens=False):
        return [10, 11]

    def __call__(self, pairs, **kwargs):
        self.encoded.extend(pairs)
        return {"input_ids": [[21, 22] for _ in pairs]}

    def pad(self, inputs, **kwargs):
        if self.pad_token is None:
            raise ValueError(
                "Asking to pad but the tokenizer does not have a padding token"
            )
        return {"input_ids": FakeTensor(inputs["input_ids"])}


class FakeAutoTokenizer:
    @classmethod
    def from_pretrained(cls, model_name, padding_side="left"):
        return FakeTokenizer()


class FakeCausalLM:
    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, **inputs):
        return types.SimpleNamespace(logits=FakeLogits())


class FakeAutoModelForCausalLM:
    @classmethod
    def from_pretrained(cls, model_name, **kwargs):
        return FakeCausalLM()


class FakeEmbeddingModel:
    def __init__(self, dense_size=1024):
        self.dense_size = dense_size
        self.last_dense_tensor = None

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, **inputs):
        self.last_dense_tensor = FakeDenseTensor(self.dense_size)
        return types.SimpleNamespace(
            last_hidden_state=FakeEmbeddingTensor(self.last_dense_tensor)
        )


class FakeAutoModel:
    @classmethod
    def from_pretrained(cls, model_name, **kwargs):
        if os.getenv("FAKE_QWEN_DENSE_LOAD_ERROR") == "1":
            raise RuntimeError("qwen dense load failed")
        dense_size = int(os.getenv("FAKE_QWEN_DENSE_DIM", "1024"))
        return FakeEmbeddingModel(dense_size)


class FakeTensor:
    def __init__(self, value):
        self.value = value

    def to(self, device):
        return self


class FakeEmbeddingTensor:
    shape = (1, 1, 1024)

    def __init__(self, dense_tensor):
        self.dense_tensor = dense_tensor

    def __getitem__(self, key):
        return self.dense_tensor


class FakeDenseTensor:
    def __init__(self, dense_size=1024):
        self.dense_size = dense_size
        self.shape = (1, dense_size)
        self.normalized = False

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return [[1.0] * self.dense_size]


class FakeLogits:
    def __getitem__(self, key):
        return self


class FakeTorchModule(types.ModuleType):
    float16 = "float16"
    Tensor = FakeTensor

    def __init__(self):
        super().__init__("torch")
        self.cuda = types.SimpleNamespace(
            is_available=lambda: False,
            device_count=lambda: 0,
        )
        self.nn = types.SimpleNamespace(
            functional=types.SimpleNamespace(
                log_softmax=self._log_softmax,
                normalize=self._normalize,
            )
        )

    class _NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    def no_grad(self):
        return self._NoGrad()

    def stack(self, values, dim=1):
        return FakeProbability()

    def _log_softmax(self, values, dim=1):
        return FakeProbability()

    def _normalize(self, value, p=2, dim=1):
        value.normalized = True
        return value


class FakeProbability:
    def __getitem__(self, key):
        return self

    def exp(self):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return [0.75]


class FakeFastAPI:
    def __init__(self, *args, **kwargs):
        pass

    def get(self, *args, **kwargs):
        return self._decorator

    def post(self, *args, **kwargs):
        return self._decorator

    def middleware(self, *args, **kwargs):
        return self._decorator

    @staticmethod
    def _decorator(func):
        return func


class FakeHTTPException(Exception):
    def __init__(self, status_code=None, detail=None, headers=None):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
        self.headers = headers


class FakeHTTPBearer:
    def __init__(self, *args, **kwargs):
        pass


class FakeBaseModel:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class FakeMetric:
    def __init__(self):
        self.last_info = None

    def labels(self, *args, **kwargs):
        return self

    def inc(self, *args, **kwargs):
        return None

    def dec(self, *args, **kwargs):
        return None

    def set(self, *args, **kwargs):
        return None

    def observe(self, *args, **kwargs):
        return None

    def info(self, *args, **kwargs):
        self.last_info = args[0] if args else kwargs
        return None

    def time(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def identity_decorator(*args, **kwargs):
    def decorate(func):
        return func

    return decorate


def install_support_fakes():
    fake_numpy = types.ModuleType("numpy")
    fake_numpy.ndarray = list
    fake_numpy.array = lambda value: value
    fake_numpy.maximum = lambda value, minimum: value
    fake_numpy.linalg = types.SimpleNamespace(norm=lambda *args, **kwargs: 1)

    fake_fastapi = types.ModuleType("fastapi")
    fake_fastapi.Depends = lambda dependency=None: dependency
    fake_fastapi.FastAPI = FakeFastAPI
    fake_fastapi.HTTPException = FakeHTTPException
    fake_fastapi.Request = object

    fake_responses = types.ModuleType("fastapi.responses")
    fake_responses.JSONResponse = object
    fake_responses.Response = object

    fake_security = types.ModuleType("fastapi.security")
    fake_security.HTTPAuthorizationCredentials = object
    fake_security.HTTPBearer = FakeHTTPBearer

    fake_prometheus = types.ModuleType("prometheus_client")
    fake_prometheus.CONTENT_TYPE_LATEST = "text/plain"
    fake_prometheus.Counter = lambda *args, **kwargs: FakeMetric()
    fake_prometheus.Gauge = lambda *args, **kwargs: FakeMetric()
    fake_prometheus.Histogram = lambda *args, **kwargs: FakeMetric()
    fake_prometheus.Info = lambda *args, **kwargs: FakeMetric()
    fake_prometheus.generate_latest = lambda: b""

    fake_pydantic = types.ModuleType("pydantic")
    fake_pydantic.BaseModel = FakeBaseModel
    fake_pydantic.Field = lambda default, **kwargs: default
    fake_pydantic.field_validator = identity_decorator
    fake_pydantic.model_validator = identity_decorator

    fake_starlette_status = types.ModuleType("starlette.status")
    fake_starlette_status.HTTP_504_GATEWAY_TIMEOUT = 504

    return {
        "numpy": fake_numpy,
        "fastapi": fake_fastapi,
        "fastapi.responses": fake_responses,
        "fastapi.security": fake_security,
        "prometheus_client": fake_prometheus,
        "pydantic": fake_pydantic,
        "starlette.status": fake_starlette_status,
    }


def load_server(env=None):
    module_name = "bge_m3_server_under_test"
    sys.modules.pop(module_name, None)

    fake_flag_embedding = types.ModuleType("FlagEmbedding")
    fake_flag_embedding.BGEM3FlagModel = FakeBGEM3FlagModel
    fake_flag_embedding.FlagReranker = FakeFlagReranker

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModel = FakeAutoModel
    fake_transformers.AutoTokenizer = FakeAutoTokenizer
    fake_transformers.AutoModelForCausalLM = FakeAutoModelForCausalLM

    modules = {
        **install_support_fakes(),
        "FlagEmbedding": fake_flag_embedding,
        "transformers": fake_transformers,
        "torch": FakeTorchModule(),
    }

    with patch.dict(sys.modules, modules), patch.dict(os.environ, env or {}, clear=False):
        spec = importlib.util.spec_from_file_location(module_name, SERVER_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module


class TrackingBgeBackend:
    def __init__(self):
        self.calls = []

    def embed(
        self,
        sentences,
        return_dense=True,
        return_sparse=True,
        return_colbert=True,
        normalize_dense=False,
    ):
        self.calls.append(
            {
                "sentences": sentences,
                "return_dense": return_dense,
                "return_sparse": return_sparse,
                "return_colbert": return_colbert,
                "normalize_dense": normalize_dense,
            }
        )
        result = {}
        if return_dense:
            result["dense_vecs"] = [[3.0, 4.0] for _ in sentences]
        if return_sparse:
            result["lexical_weights"] = [{"42": 0.5} for _ in sentences]
        if return_colbert:
            result["colbert_vecs"] = [[[0.1, 0.2]] for _ in sentences]
        return result


class TrackingQwenDenseBackend:
    def __init__(self):
        self.calls = []

    def embed_dense(self, sentences, normalize_dense=False):
        self.calls.append(
            {"sentences": sentences, "normalize_dense": normalize_dense}
        )
        return {"dense_vecs": [[9.0] * 1024 for _ in sentences]}


class FailingQwenDenseBackend:
    def embed_dense(self, sentences, normalize_dense=False):
        raise RuntimeError("qwen dense failed")


def build_embedding_service(module, dense_model_name):
    service = module.EmbeddingService.__new__(module.EmbeddingService)
    service.bge_backend = TrackingBgeBackend()
    service.qwen_dense_backend = (
        TrackingQwenDenseBackend()
        if dense_model_name == module.QWEN_DENSE_EMBEDDING_MODEL
        else None
    )
    service.bge_model_name = module.BGE_EMBEDDING_MODEL
    service.dense_model_name = dense_model_name
    service.sparse_model_name = module.BGE_EMBEDDING_MODEL
    service.colbert_model_name = module.BGE_EMBEDDING_MODEL
    return service


class RerankerWrapperTests(unittest.TestCase):
    def test_resolve_reranker_model_defaults_to_bge_for_invalid_value(self):
        module = load_server({"RERANKER_MODEL": "invalid"})
        self.assertEqual(module._resolve_reranker_model(), module.BGE_RERANKER_MODEL)

    def test_resolve_dense_embedding_model_defaults_to_bge_for_invalid_value(self):
        module = load_server()
        with patch.dict(os.environ, {"DENSE_EMBEDDING_MODEL": "invalid"}):
            self.assertEqual(
                module._resolve_dense_embedding_model(), module.BGE_EMBEDDING_MODEL
            )

    def test_resolve_dense_embedding_model_accepts_qwen(self):
        module = load_server()
        with patch.dict(
            os.environ,
            {"DENSE_EMBEDDING_MODEL": "Qwen/Qwen3-Embedding-0.6B"},
        ):
            self.assertEqual(
                module._resolve_dense_embedding_model(),
                module.QWEN_DENSE_EMBEDDING_MODEL,
            )

    def test_qwen_dense_only_request_calls_only_qwen_backend(self):
        module = load_server()
        service = build_embedding_service(module, module.QWEN_DENSE_EMBEDDING_MODEL)
        result = service.embed(
            ["alpha"],
            return_dense=True,
            return_sparse=False,
            return_colbert=False,
            normalize_dense=True,
        )
        self.assertEqual(result["dense_vecs"][0], [9.0] * 1024)
        self.assertEqual(service.bge_backend.calls, [])
        self.assertEqual(
            service.qwen_dense_backend.calls,
            [{"sentences": ["alpha"], "normalize_dense": True}],
        )

    def test_qwen_sparse_colbert_request_calls_only_bge_backend(self):
        module = load_server()
        service = build_embedding_service(module, module.QWEN_DENSE_EMBEDDING_MODEL)
        result = service.embed(
            ["alpha"],
            return_dense=False,
            return_sparse=True,
            return_colbert=True,
            normalize_dense=False,
        )
        self.assertNotIn("dense_vecs", result)
        self.assertEqual(result["lexical_weights"], [{"42": 0.5}])
        self.assertEqual(result["colbert_vecs"], [[[0.1, 0.2]]])
        self.assertEqual(service.qwen_dense_backend.calls, [])
        self.assertEqual(len(service.bge_backend.calls), 1)
        self.assertFalse(service.bge_backend.calls[0]["return_dense"])

    def test_qwen_hybrid_request_merges_dense_with_bge_sparse_and_colbert(self):
        module = load_server()
        service = build_embedding_service(module, module.QWEN_DENSE_EMBEDDING_MODEL)
        result = service.embed(
            ["alpha"],
            return_dense=True,
            return_sparse=True,
            return_colbert=True,
            normalize_dense=False,
        )
        self.assertEqual(result["dense_vecs"][0], [9.0] * 1024)
        self.assertEqual(result["lexical_weights"], [{"42": 0.5}])
        self.assertEqual(result["colbert_vecs"], [[[0.1, 0.2]]])
        self.assertEqual(len(service.qwen_dense_backend.calls), 1)
        self.assertEqual(len(service.bge_backend.calls), 1)
        self.assertFalse(service.bge_backend.calls[0]["return_dense"])

    def test_bge_dense_request_uses_bge_for_all_requested_outputs(self):
        module = load_server()
        service = build_embedding_service(module, module.BGE_EMBEDDING_MODEL)
        result = service.embed(
            ["alpha"],
            return_dense=True,
            return_sparse=True,
            return_colbert=True,
            normalize_dense=True,
        )
        self.assertEqual(result["dense_vecs"], [[3.0, 4.0]])
        self.assertEqual(result["lexical_weights"], [{"42": 0.5}])
        self.assertEqual(result["colbert_vecs"], [[[0.1, 0.2]]])
        self.assertIsNone(service.qwen_dense_backend)
        self.assertEqual(len(service.bge_backend.calls), 1)
        self.assertTrue(service.bge_backend.calls[0]["return_dense"])
        self.assertTrue(service.bge_backend.calls[0]["normalize_dense"])

    def test_qwen_hybrid_backend_failure_raises_without_partial_response(self):
        module = load_server()
        service = build_embedding_service(module, module.QWEN_DENSE_EMBEDDING_MODEL)
        service.qwen_dense_backend = FailingQwenDenseBackend()
        with self.assertRaisesRegex(RuntimeError, "qwen dense failed"):
            service.embed(
                ["alpha"],
                return_dense=True,
                return_sparse=True,
                return_colbert=True,
                normalize_dense=False,
            )

    def test_bge_backend_uses_flag_reranker_compute_score(self):
        module = load_server({"RERANKER_MODEL": "BAAI/bge-reranker-v2-m3"})
        wrapper = module.RerankerWrapper(module.BGE_RERANKER_MODEL)
        scores = wrapper.score([["query", "passage"]], normalize=True)
        self.assertEqual(scores, [0.25])
        self.assertEqual(wrapper.model.calls[-1], ([["query", "passage"]], True))

    def test_qwen_backend_uses_yes_no_probability_scorer(self):
        module = load_server({"RERANKER_MODEL": "Qwen/Qwen3-Reranker-0.6B"})
        wrapper = module.RerankerWrapper(module.QWEN_RERANKER_MODEL)
        scores = wrapper.score([["query", "passage"]], normalize=False)
        self.assertEqual(scores, [0.75])
        self.assertEqual(wrapper.tokenizer.pad_token, wrapper.tokenizer.eos_token)
        self.assertIn("<Query>: query", wrapper.tokenizer.encoded[0])
        self.assertIn("<Document>: passage", wrapper.tokenizer.encoded[0])

    def test_qwen_normalize_is_api_compatible_noop(self):
        module = load_server({"RERANKER_MODEL": "Qwen/Qwen3-Reranker-0.6B"})
        wrapper = module.RerankerWrapper(module.QWEN_RERANKER_MODEL)
        self.assertEqual(
            wrapper.score([["query", "passage"]], normalize=True),
            wrapper.score([["query", "passage"]], normalize=False),
        )

    def test_health_exposes_selected_models(self):
        module = load_server(
            {
                "DENSE_EMBEDDING_MODEL": "Qwen/Qwen3-Embedding-0.6B",
                "RERANKER_MODEL": "Qwen/Qwen3-Reranker-0.6B",
            }
        )
        health = asyncio.run(module.health_check())
        self.assertEqual(
            health["dense_embedding_model"], module.QWEN_DENSE_EMBEDDING_MODEL
        )
        self.assertEqual(health["model"], module.BGE_EMBEDDING_MODEL)
        self.assertEqual(health["reranker_model"], module.QWEN_RERANKER_MODEL)

    def test_server_info_exposes_dense_embedding_model(self):
        module = load_server(
            {"DENSE_EMBEDDING_MODEL": "Qwen/Qwen3-Embedding-0.6B"}
        )
        self.assertEqual(
            module.server_info.last_info["dense_embedding_model"],
            module.QWEN_DENSE_EMBEDDING_MODEL,
        )

    def test_embeddings_response_model_declares_backend_metadata(self):
        module = load_server()
        fields = module.EmbeddingsListResponse.__annotations__
        self.assertIs(fields["model_name"], str)
        self.assertIs(fields["dense_model_name"], str)
        self.assertIs(fields["sparse_model_name"], str)
        self.assertIs(fields["colbert_model_name"], str)

    def test_embeddings_endpoint_uses_bge_model_name_by_default(self):
        module = load_server({"DENSE_EMBEDDING_MODEL": "BAAI/bge-m3"})

        async def fake_process_request(request):
            return {"dense_vecs": [[1.0]]}

        module.processor.process_request = fake_process_request
        request = module.EmbedRequest(
            sentences=["alpha"],
            return_dense=True,
            return_sparse=False,
            return_colbert=False,
            normalize_dense=False,
            sparse_as_indices=False,
        )

        response = asyncio.run(module.get_embeddings(request))
        self.assertEqual(response.model_name, module.BGE_EMBEDDING_MODEL)
        self.assertEqual(response.dense_model_name, module.BGE_EMBEDDING_MODEL)


if __name__ == "__main__":
    unittest.main()
