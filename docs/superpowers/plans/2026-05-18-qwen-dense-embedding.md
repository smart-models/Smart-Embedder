# Qwen Dense Embedding Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional `Qwen/Qwen3-Embedding-0.6B` dense embeddings while keeping BGE-M3 sparse and ColBERT outputs available through the existing `/embeddings/` endpoint.

**Architecture:** Introduce an `EmbeddingService` that orchestrates a required BGE-M3 backend and an optional Qwen dense backend. The request processor keeps one `.embed(...)` call, while the service routes dense, sparse, and ColBERT inference to the correct backend and merges the result.

**Tech Stack:** FastAPI, FlagEmbedding `BGEM3FlagModel`, Transformers `AutoTokenizer` and `AutoModel`, PyTorch, Docker Compose, Windows batch, Bash, stdlib unittest.

---

## File Structure

- Modify `bge-m3_server.py`: add dense model constants and resolver, split embedding backends, add `EmbeddingService`, wire response metadata, update health and Prometheus server info.
- Modify `test_reranker_wrapper.py`: extend the existing fake module harness and add dense embedding backend/service tests without downloading models.
- Modify `start_server.bat`: prompt for dense embedding backend and set `DENSE_EMBEDDING_MODEL`.
- Modify `start_server.sh`: prompt for dense embedding backend and export `DENSE_EMBEDDING_MODEL`.
- Modify `docker-compose.yml`: pass `DENSE_EMBEDDING_MODEL` into the service.
- Modify `.env.example`: document the launcher-managed dense embedding variable.
- Modify `README.md`: document hybrid embeddings, response metadata, health metadata, and config.

### Task 1: Dense Model Resolver Tests

**Files:**
- Modify: `test_reranker_wrapper.py`
- Modify: `bge-m3_server.py`

- [ ] **Step 1: Write failing resolver tests**

Add these tests to `RerankerWrapperTests` in `test_reranker_wrapper.py`:

```python
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
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
py -m unittest test_reranker_wrapper.RerankerWrapperTests.test_resolve_dense_embedding_model_defaults_to_bge_for_invalid_value test_reranker_wrapper.RerankerWrapperTests.test_resolve_dense_embedding_model_accepts_qwen
```

Expected: both tests fail because `_resolve_dense_embedding_model`, `BGE_EMBEDDING_MODEL`, and `QWEN_DENSE_EMBEDDING_MODEL` do not exist.

- [ ] **Step 3: Add dense model constants and resolver**

In `bge-m3_server.py`, add `AutoModel` to the Transformers import:

```python
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
```

Add these constants near the existing model constants:

```python
BGE_EMBEDDING_MODEL = "BAAI/bge-m3"
QWEN_DENSE_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
QWEN_DENSE_VECTOR_SIZE = 1024
SUPPORTED_DENSE_EMBEDDING_MODELS = {
    BGE_EMBEDDING_MODEL,
    QWEN_DENSE_EMBEDDING_MODEL,
}
```

Add the resolver after `_resolve_reranker_model()`:

```python
def _resolve_dense_embedding_model() -> str:
    configured = os.getenv("DENSE_EMBEDDING_MODEL", BGE_EMBEDDING_MODEL).strip()
    if configured in SUPPORTED_DENSE_EMBEDDING_MODELS:
        return configured

    logging.warning(
        f"Unsupported DENSE_EMBEDDING_MODEL '{configured}', "
        f"using {BGE_EMBEDDING_MODEL}"
    )
    return BGE_EMBEDDING_MODEL
```

- [ ] **Step 4: Update fake Transformers module for `AutoModel`**

In `test_reranker_wrapper.py`, add this fake class near `FakeAutoModelForCausalLM`:

```python
class FakeAutoModel:
    @classmethod
    def from_pretrained(cls, model_name, **kwargs):
        if os.getenv("FAKE_QWEN_DENSE_LOAD_ERROR") == "1":
            raise RuntimeError("qwen dense load failed")
        dense_size = int(os.getenv("FAKE_QWEN_DENSE_DIM", "1024"))
        return FakeEmbeddingModel(dense_size)
```

Add this fake model near `FakeCausalLM`:

```python
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
```

Add this tensor near `FakeTensor`:

```python
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
```

Update `FakeTorchModule.__init__` so `functional` includes dense normalization support:

```python
        self.nn = types.SimpleNamespace(
            functional=types.SimpleNamespace(
                log_softmax=self._log_softmax,
                normalize=self._normalize,
            )
        )
```

Add this method to `FakeTorchModule`:

```python
    def _normalize(self, value, p=2, dim=1):
        value.normalized = True
        return value
```

Update `load_server()`. Replace:

```python
    fake_transformers.AutoTokenizer = FakeAutoTokenizer
    fake_transformers.AutoModelForCausalLM = FakeAutoModelForCausalLM
```

with:

```python
    fake_transformers.AutoModel = FakeAutoModel
    fake_transformers.AutoModelForCausalLM = FakeAutoModelForCausalLM
    fake_transformers.AutoTokenizer = FakeAutoTokenizer
```

- [ ] **Step 5: Run resolver tests and verify pass**

Run:

```powershell
py -m unittest test_reranker_wrapper.RerankerWrapperTests.test_resolve_dense_embedding_model_defaults_to_bge_for_invalid_value test_reranker_wrapper.RerankerWrapperTests.test_resolve_dense_embedding_model_accepts_qwen
```

Expected: both resolver tests pass.

- [ ] **Step 6: Commit**

Run:

```powershell
git add bge-m3_server.py test_reranker_wrapper.py
git commit -m "test: cover dense embedding model resolver"
```

### Task 2: Embedding Service Orchestration

**Files:**
- Modify: `test_reranker_wrapper.py`
- Modify: `bge-m3_server.py`

- [ ] **Step 1: Write failing service orchestration tests**

Add these helper fakes above `RerankerWrapperTests`:

```python
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
```

Add this factory helper:

```python
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
```

Add these tests:

```python
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
```

- [ ] **Step 2: Run service tests and verify failure**

Run:

```powershell
py -m unittest test_reranker_wrapper.RerankerWrapperTests.test_qwen_dense_only_request_calls_only_qwen_backend test_reranker_wrapper.RerankerWrapperTests.test_qwen_sparse_colbert_request_calls_only_bge_backend test_reranker_wrapper.RerankerWrapperTests.test_qwen_hybrid_request_merges_dense_with_bge_sparse_and_colbert test_reranker_wrapper.RerankerWrapperTests.test_bge_dense_request_uses_bge_for_all_requested_outputs test_reranker_wrapper.RerankerWrapperTests.test_qwen_hybrid_backend_failure_raises_without_partial_response
```

Expected: tests fail because `EmbeddingService` does not exist.

- [ ] **Step 3: Rename `M3Wrapper` to `BgeM3EmbeddingBackend`**

In `bge-m3_server.py`, replace:

```python
class M3Wrapper:
```

with:

```python
class BgeM3EmbeddingBackend:
```

Keep the constructor and `embed(...)` behavior the same except for docstrings that should describe BGE-M3 specifically.

Add this temporary compatibility alias immediately after the renamed class:

```python
M3Wrapper = BgeM3EmbeddingBackend
```

This keeps the module importable until Task 3 replaces the global `model = M3Wrapper(...)` initialization with `embedding_service = EmbeddingService(...)`.

- [ ] **Step 4: Add Qwen dense backend**

Add this class after `BgeM3EmbeddingBackend`:

```python
class QwenDenseEmbeddingBackend:
    """Produce dense embeddings with Qwen3-Embedding."""

    def __init__(self, model_name: str, device: str = device):
        self.model_name = model_name
        self.device = device
        logging.info(f"Initializing Qwen dense embedding model '{model_name}'")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, padding_side="left"
        )
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {"dtype": torch.float16} if has_cuda else {}
        self.model = AutoModel.from_pretrained(self.model_name, **model_kwargs)
        self.model = self.model.to(self.device).eval()
        logging.info("Qwen dense embedding model ready.")

    def embed_dense(
        self, sentences: List[str], normalize_dense: bool = False
    ) -> Dict[str, np.ndarray]:
        encoded = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_LENGTH,
            return_tensors="pt",
        )
        encoded = {
            name: value.to(self.device) if hasattr(value, "to") else value
            for name, value in encoded.items()
        }

        with torch.no_grad():
            outputs = self.model(**encoded)
            dense_vecs = outputs.last_hidden_state[:, -1, :]
            if dense_vecs.shape[-1] != QWEN_DENSE_VECTOR_SIZE:
                raise RuntimeError(
                    f"{self.model_name} returned dense dimension "
                    f"{dense_vecs.shape[-1]}, expected {QWEN_DENSE_VECTOR_SIZE}"
                )
            if normalize_dense:
                dense_vecs = torch.nn.functional.normalize(dense_vecs, p=2, dim=1)

        return {"dense_vecs": dense_vecs.detach().cpu().numpy()}
```

- [ ] **Step 5: Add embedding service**

Add this class after `QwenDenseEmbeddingBackend`:

```python
class EmbeddingService:
    """Route embedding requests between BGE-M3 and the selected dense backend."""

    def __init__(
        self,
        bge_model_name: str,
        dense_model_name: str,
        devices: Optional[List[str]] = None,
    ):
        self.bge_model_name = bge_model_name
        self.dense_model_name = dense_model_name
        self.sparse_model_name = bge_model_name
        self.colbert_model_name = bge_model_name
        self.bge_backend = BgeM3EmbeddingBackend(bge_model_name, devices=devices)
        self.qwen_dense_backend = (
            QwenDenseEmbeddingBackend(dense_model_name)
            if dense_model_name == QWEN_DENSE_EMBEDDING_MODEL
            else None
        )

    def embed(
        self,
        sentences: List[str],
        return_dense: bool = True,
        return_sparse: bool = True,
        return_colbert: bool = True,
        normalize_dense: bool = False,
    ) -> Dict[str, Union[np.ndarray, List]]:
        result: Dict[str, Union[np.ndarray, List]] = {}
        use_qwen_dense = self.qwen_dense_backend is not None
        bge_return_dense = return_dense and not use_qwen_dense
        needs_bge = bge_return_dense or return_sparse or return_colbert

        if needs_bge:
            result.update(
                self.bge_backend.embed(
                    sentences,
                    return_dense=bge_return_dense,
                    return_sparse=return_sparse,
                    return_colbert=return_colbert,
                    normalize_dense=normalize_dense if bge_return_dense else False,
                )
            )

        if return_dense and use_qwen_dense:
            result.update(
                self.qwen_dense_backend.embed_dense(
                    sentences, normalize_dense=normalize_dense
                )
            )

        return result
```

- [ ] **Step 6: Update request processor type hint**

Change the constructor hint from:

```python
        model_wrapper: M3Wrapper,
```

to:

```python
        model_wrapper: Union[BgeM3EmbeddingBackend, EmbeddingService],
```

Leave the attribute name `model_wrapper` for a small diff. The union keeps the intermediate Task 2 state accurate while the global startup path still uses the temporary `M3Wrapper` alias.

- [ ] **Step 7: Run service tests and verify pass**

Run:

```powershell
py -m unittest test_reranker_wrapper.RerankerWrapperTests.test_qwen_dense_only_request_calls_only_qwen_backend test_reranker_wrapper.RerankerWrapperTests.test_qwen_sparse_colbert_request_calls_only_bge_backend test_reranker_wrapper.RerankerWrapperTests.test_qwen_hybrid_request_merges_dense_with_bge_sparse_and_colbert test_reranker_wrapper.RerankerWrapperTests.test_bge_dense_request_uses_bge_for_all_requested_outputs test_reranker_wrapper.RerankerWrapperTests.test_qwen_hybrid_backend_failure_raises_without_partial_response
```

Expected: all service orchestration tests pass.

- [ ] **Step 8: Commit**

Run:

```powershell
git add bge-m3_server.py test_reranker_wrapper.py
git commit -m "feat: route dense embeddings through selectable backend"
```

### Task 3: API Metadata and Health

**Files:**
- Modify: `test_reranker_wrapper.py`
- Modify: `bge-m3_server.py`

- [ ] **Step 1: Write failing health and metrics metadata tests**

Update `FakeMetric` so tests can inspect the `Info.info(...)` payload. Replace:

```python
class FakeMetric:
    def labels(self, *args, **kwargs):
        return self
```

with:

```python
class FakeMetric:
    def __init__(self):
        self.last_info = None

    def labels(self, *args, **kwargs):
        return self
```

Replace the `info()` method:

```python
    def info(self, *args, **kwargs):
        return None
```

with:

```python
    def info(self, *args, **kwargs):
        self.last_info = args[0] if args else kwargs
        return None
```

Replace `test_health_exposes_selected_reranker_model` with:

```python
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
```

- [ ] **Step 2: Write failing response model metadata test**

Add this test:

```python
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
```

- [ ] **Step 3: Run metadata tests and verify failure**

Run:

```powershell
py -m unittest test_reranker_wrapper.RerankerWrapperTests.test_health_exposes_selected_models test_reranker_wrapper.RerankerWrapperTests.test_server_info_exposes_dense_embedding_model test_reranker_wrapper.RerankerWrapperTests.test_embeddings_response_model_declares_backend_metadata test_reranker_wrapper.RerankerWrapperTests.test_embeddings_endpoint_uses_bge_model_name_by_default
```

Expected: health and server info tests fail because `dense_embedding_model` is absent, response metadata test fails because the new response fields are not annotated yet, and the endpoint test fails because the response does not expose `dense_model_name` yet.

- [ ] **Step 4: Add response metadata fields**

In `EmbeddingsListResponse`, replace:

```python
    model_name: str
    processing_time_ms: float
```

with:

```python
    model_name: str
    dense_model_name: str
    sparse_model_name: str
    colbert_model_name: str
    processing_time_ms: float
```

- [ ] **Step 5: Replace global model initialization**

Replace:

```python
model = M3Wrapper("BAAI/bge-m3", devices=MULTI_GPU_DEVICES)
reranker = RerankerWrapper(_resolve_reranker_model())
stats = ServerStats()
processor = RequestProcessor(
    model,
```

with:

```python
embedding_service = EmbeddingService(
    BGE_EMBEDDING_MODEL,
    _resolve_dense_embedding_model(),
    devices=MULTI_GPU_DEVICES,
)
reranker = RerankerWrapper(_resolve_reranker_model())
stats = ServerStats()
processor = RequestProcessor(
    embedding_service,
```

Delete the temporary alias from Task 2 after this replacement:

```python
M3Wrapper = BgeM3EmbeddingBackend
```

- [ ] **Step 6: Update Prometheus server info**

Replace:

```python
        "model": "BAAI/bge-m3",
```

with:

```python
        "model": BGE_EMBEDDING_MODEL,
        "dense_embedding_model": embedding_service.dense_model_name,
```

- [ ] **Step 7: Update health response**

Replace:

```python
        "model": model.model_name,
        "reranker_model": reranker.model_name,
```

with:

```python
        "model": embedding_service.bge_model_name,
        "dense_embedding_model": embedding_service.dense_model_name,
        "reranker_model": reranker.model_name,
```

- [ ] **Step 8: Update embeddings endpoint response**

Replace:

```python
                model_name=model.model_name,
                processing_time_ms=round(processing_time_ms, 2),
```

with:

```python
                model_name=embedding_service.dense_model_name,
                dense_model_name=embedding_service.dense_model_name,
                sparse_model_name=embedding_service.sparse_model_name,
                colbert_model_name=embedding_service.colbert_model_name,
                processing_time_ms=round(processing_time_ms, 2),
```

- [ ] **Step 9: Run metadata tests and verify pass**

Run:

```powershell
py -m unittest test_reranker_wrapper.RerankerWrapperTests.test_health_exposes_selected_models test_reranker_wrapper.RerankerWrapperTests.test_server_info_exposes_dense_embedding_model test_reranker_wrapper.RerankerWrapperTests.test_embeddings_response_model_declares_backend_metadata test_reranker_wrapper.RerankerWrapperTests.test_embeddings_endpoint_uses_bge_model_name_by_default
```

Expected: all four metadata tests pass.

- [ ] **Step 10: Commit**

Run:

```powershell
git add bge-m3_server.py test_reranker_wrapper.py
git commit -m "feat: expose embedding backend metadata"
```

### Task 4: Qwen Dense Backend Behavior

**Files:**
- Modify: `test_reranker_wrapper.py`
- Modify: `bge-m3_server.py`

- [ ] **Step 1: Write failing Qwen backend tests**

Add these tests:

```python
    def test_qwen_dense_backend_returns_1024_dimension_vectors(self):
        module = load_server(
            {"DENSE_EMBEDDING_MODEL": "Qwen/Qwen3-Embedding-0.6B"}
        )
        backend = module.embedding_service.qwen_dense_backend
        result = backend.embed_dense(["alpha"], normalize_dense=False)
        self.assertEqual(len(result["dense_vecs"][0]), 1024)

    def test_qwen_dense_backend_applies_normalization_when_requested(self):
        module = load_server(
            {"DENSE_EMBEDDING_MODEL": "Qwen/Qwen3-Embedding-0.6B"}
        )
        backend = module.embedding_service.qwen_dense_backend
        result = backend.embed_dense(["alpha"], normalize_dense=True)
        self.assertEqual(len(result["dense_vecs"][0]), 1024)
        self.assertTrue(backend.model.last_dense_tensor.normalized)

    def test_qwen_dense_backend_wrong_dimension_raises_runtime_error(self):
        module = load_server(
            {
                "DENSE_EMBEDDING_MODEL": "Qwen/Qwen3-Embedding-0.6B",
                "FAKE_QWEN_DENSE_DIM": "768",
            }
        )
        backend = module.embedding_service.qwen_dense_backend
        with self.assertRaisesRegex(RuntimeError, "expected 1024"):
            backend.embed_dense(["alpha"], normalize_dense=False)

    def test_qwen_dense_load_failure_fails_server_startup(self):
        with self.assertRaisesRegex(RuntimeError, "qwen dense load failed"):
            load_server(
                {
                    "DENSE_EMBEDDING_MODEL": "Qwen/Qwen3-Embedding-0.6B",
                    "FAKE_QWEN_DENSE_LOAD_ERROR": "1",
                }
            )
```

- [ ] **Step 2: Run Qwen backend tests and verify failure**

Run:

```powershell
py -m unittest test_reranker_wrapper.RerankerWrapperTests.test_qwen_dense_backend_returns_1024_dimension_vectors test_reranker_wrapper.RerankerWrapperTests.test_qwen_dense_backend_applies_normalization_when_requested test_reranker_wrapper.RerankerWrapperTests.test_qwen_dense_backend_wrong_dimension_raises_runtime_error test_reranker_wrapper.RerankerWrapperTests.test_qwen_dense_load_failure_fails_server_startup
```

Expected: tests fail until `QwenDenseEmbeddingBackend` validates dimensions, normalizes through `torch.nn.functional.normalize`, and Qwen startup failure is allowed to propagate.

- [ ] **Step 3: Adjust Qwen backend implementation**

Confirm the backend keeps left padding and includes this comment immediately before pooling:

```python
            # With left padding, -1 is the real final token for every row.
            dense_vecs = outputs.last_hidden_state[:, -1, :]
```

Confirm `QwenDenseEmbeddingBackend.embed_dense()` contains this normalization block:

```python
            if normalize_dense:
                dense_vecs = torch.nn.functional.normalize(dense_vecs, p=2, dim=1)
```

Confirm it validates vector size before returning:

```python
            if dense_vecs.shape[-1] != QWEN_DENSE_VECTOR_SIZE:
                raise RuntimeError(
                    f"{self.model_name} returned dense dimension "
                    f"{dense_vecs.shape[-1]}, expected {QWEN_DENSE_VECTOR_SIZE}"
                )
```

- [ ] **Step 4: Run Qwen backend tests and verify pass**

Run:

```powershell
py -m unittest test_reranker_wrapper.RerankerWrapperTests.test_qwen_dense_backend_returns_1024_dimension_vectors test_reranker_wrapper.RerankerWrapperTests.test_qwen_dense_backend_applies_normalization_when_requested test_reranker_wrapper.RerankerWrapperTests.test_qwen_dense_backend_wrong_dimension_raises_runtime_error test_reranker_wrapper.RerankerWrapperTests.test_qwen_dense_load_failure_fails_server_startup
```

Expected: all four tests pass.

- [ ] **Step 5: Commit**

Run:

```powershell
git add bge-m3_server.py test_reranker_wrapper.py
git commit -m "test: cover qwen dense embedding backend"
```

### Task 5: Startup Scripts and Docker Configuration

**Files:**
- Modify: `start_server.bat`
- Modify: `start_server.sh`
- Modify: `docker-compose.yml`
- Modify: `.env.example`

- [ ] **Step 1: Update Windows launcher prompt**

In `start_server.bat`, add this prompt block before the reranker prompt:

```bat
echo ======================================
echo  Select dense embedding backend
echo ======================================
echo.
echo Do you want dense embeddings to use:
echo   [1] BGE  ^(BAAI/bge-m3^)
echo   [2] QWEN ^(Qwen/Qwen3-Embedding-0.6B^)
echo.
set /p dense_choice="Enter choice ^(1 or 2^): "
if "%dense_choice%"=="2" (
    set "DENSE_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B"
) else (
    if not "%dense_choice%"=="" if not "%dense_choice%"=="1" echo [WARNING] Invalid choice, defaulting dense embeddings to BGE
    set "DENSE_EMBEDDING_MODEL=BAAI/bge-m3"
)
echo.
```

Add this line to the startup summary:

```bat
echo  Dense:  %DENSE_EMBEDDING_MODEL%
```

- [ ] **Step 2: Update Bash launcher prompt**

In `start_server.sh`, add this function before `ask_reranker()`:

```bash
ask_dense_embedding_model() {
    echo "======================================" >&2
    echo "  Select dense embedding backend" >&2
    echo "======================================" >&2
    echo "" >&2
    echo "Do you want dense embeddings to use:" >&2
    echo "  [1] BGE  (BAAI/bge-m3)" >&2
    echo "  [2] QWEN (Qwen/Qwen3-Embedding-0.6B)" >&2
    echo "" >&2
    read -r -p "Enter choice (1 or 2): " choice >&2

    case "$choice" in
        2) echo "Qwen/Qwen3-Embedding-0.6B" ;;
        1) echo "BAAI/bge-m3" ;;
        *)
            echo "[WARNING] Invalid choice, defaulting dense embeddings to BGE" >&2
            echo "BAAI/bge-m3"
            ;;
    esac
}
```

Add this before `RERANKER_MODEL="$(ask_reranker)"`:

```bash
DENSE_EMBEDDING_MODEL="$(ask_dense_embedding_model)"
export DENSE_EMBEDDING_MODEL
```

Add this line to the startup summary:

```bash
echo "  Dense:  $DENSE_EMBEDDING_MODEL"
```

- [ ] **Step 3: Update Docker Compose environment**

In `docker-compose.yml`, add this environment entry before `RERANKER_MODEL`:

```yaml
      DENSE_EMBEDDING_MODEL: ${DENSE_EMBEDDING_MODEL:-BAAI/bge-m3}
```

- [ ] **Step 4: Update `.env.example`**

In `.env.example`, add this block before the reranker block:

```text
# Dense embedding model selected by start_server.bat/start_server.sh.
# The launcher prompts for BGE or QWEN and sets this automatically.
# Defensive default: BAAI/bge-m3
# DENSE_EMBEDDING_MODEL=BAAI/bge-m3
```

- [ ] **Step 5: Verify script syntax**

Run:

```powershell
bash -n ./start_server.sh
```

Expected: no output and exit code 0.

- [ ] **Step 6: Commit**

Run:

```powershell
git add start_server.bat start_server.sh docker-compose.yml .env.example
git commit -m "feat: add startup selection for dense embedding backend"
```

### Task 6: README Documentation

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update feature summary**

Change the top feature table so the embedding row explains hybrid support:

```markdown
| **Embeddings** | BGE-M3 dense/sparse/ColBERT by default; optional Qwen3 dense with BGE-M3 sparse and ColBERT |
```

- [ ] **Step 2: Document startup prompts**

In the Quick Start section after the command table, add:

```markdown
Startup asks two independent model questions:

1. Dense embedding backend: choose BGE for current all-BGE embeddings, or QWEN to return Qwen dense vectors while keeping BGE sparse and ColBERT vectors.
2. Reranker backend: choose BGE or QWEN for `/rerank`.
```

- [ ] **Step 3: Update embeddings response example**

Update the `/embeddings/` example response metadata to include:

```json
  "model_name": "Qwen/Qwen3-Embedding-0.6B",
  "dense_model_name": "Qwen/Qwen3-Embedding-0.6B",
  "sparse_model_name": "BAAI/bge-m3",
  "colbert_model_name": "BAAI/bge-m3",
  "processing_time_ms": 123.45
```

- [ ] **Step 4: Update configuration table**

Add this row before `RERANKER_MODEL`:

```markdown
| `DENSE_EMBEDDING_MODEL` | `BAAI/bge-m3` | Dense embedding backend selected by launcher (`BAAI/bge-m3` or `Qwen/Qwen3-Embedding-0.6B`) |
```

- [ ] **Step 5: Add model note**

Add this note near the reranker model note:

```markdown
When `DENSE_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B`, only dense vectors change. Sparse lexical weights and ColBERT vectors still come from `BAAI/bge-m3`, so mixed requests are supported through the same `/embeddings/` endpoint.

The Qwen dense path intentionally does not add query/document instruction prefixes. This keeps the existing `/embeddings/` API transparent, but deployments optimizing retrieval quality should benchmark task-specific Qwen formatting separately before changing request semantics.
```

- [ ] **Step 6: Commit**

Run:

```powershell
git add README.md
git commit -m "docs: document qwen dense embedding selection"
```

### Task 7: Final Verification

**Files:**
- Verify: `bge-m3_server.py`
- Verify: `test_reranker_wrapper.py`
- Verify: `start_server.sh`
- Verify: `README.md`

- [ ] **Step 1: Run unit tests**

Run:

```powershell
py -m unittest test_reranker_wrapper
```

Expected: all tests pass.

- [ ] **Step 2: Run Bash syntax check**

Run:

```powershell
bash -n ./start_server.sh
```

Expected: no output and exit code 0.

- [ ] **Step 3: Run Python syntax compile**

Run:

```powershell
py -m py_compile bge-m3_server.py test_reranker_wrapper.py
```

Expected: no output and exit code 0.

- [ ] **Step 4: Inspect diff**

Run:

```powershell
git diff --stat HEAD
git diff -- bge-m3_server.py test_reranker_wrapper.py start_server.bat start_server.sh docker-compose.yml .env.example README.md
```

Expected: changes are limited to dense embedding selection, metadata, startup wiring, tests, and docs.

- [ ] **Step 5: Final commit if verification edits were needed**

If Step 1, Step 2, or Step 3 required small fixes, commit them:

```powershell
git add bge-m3_server.py test_reranker_wrapper.py start_server.bat start_server.sh docker-compose.yml .env.example README.md
git commit -m "chore: verify qwen dense embedding selection"
```

Expected: no commit is created if there were no verification fixes.
