# Interactive Reranker Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an interactive BGE/Qwen reranker choice to the launch scripts while keeping `/rerank` unchanged.

**Architecture:** The launch scripts always prompt for a reranker and pass the selected Hugging Face model id through `RERANKER_MODEL`. `bge-m3_server.py` keeps one `RerankerWrapper` interface and chooses either `FlagReranker` or a Qwen CausalLM yes/no-logit scorer internally.

**Tech Stack:** FastAPI, FlagEmbedding, transformers AutoTokenizer/AutoModelForCausalLM, Docker Compose, Windows batch, Bash, stdlib unittest.

---

## File Structure

- Modify `bge-m3_server.py`: add reranker model selection helpers, add Qwen CausalLM backend inside `RerankerWrapper`, expose `reranker_model` in `/health`.
- Modify `start_server.bat`: add mandatory reranker prompt and pass `RERANKER_MODEL` to local and Docker runs.
- Modify `start_server.sh`: add mandatory reranker prompt and export `RERANKER_MODEL` to local and Docker runs.
- Modify `docker-compose.yml`: pass `RERANKER_MODEL` into the container and allow a longer cold-start healthcheck.
- Modify `README.md`: document interactive reranker selection and remove BGE-only wording where misleading.
- Modify `.env.example`: document the internal variable as launcher-managed.
- Create `test_reranker_wrapper.py`: focused stdlib `unittest` coverage with fake model modules to avoid downloading models.

### Task 1: Server Reranker Backend Selection

**Files:**
- Modify: `bge-m3_server.py`

- [ ] **Step 1: Add imports and constants**

Change the typing import and add the Transformers imports immediately after the `FlagEmbedding` import:

```python
from typing import Callable, Dict, List, Optional, Union

from transformers import AutoModelForCausalLM, AutoTokenizer
```

Add these constants after `MULTI_GPU_DEVICES = None`:

```python
BGE_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
QWEN_RERANKER_MODEL = "Qwen/Qwen3-Reranker-0.6B"
QWEN_RERANKER_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)
QWEN_RERANKER_PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and "
    'the Instruct provided. Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n<|im_start|>user\n"
)
QWEN_RERANKER_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
SUPPORTED_RERANKER_MODELS = {
    BGE_RERANKER_MODEL,
    QWEN_RERANKER_MODEL,
}
```

- [ ] **Step 2: Add model resolver and Qwen token limit**

Add after the `RERANK_GPU_TIMEOUT` block:

```python
QWEN_RERANK_MAX_LENGTH = _env_int_range(
    "QWEN_RERANK_MAX_LENGTH", 8192, min_value=128, max_value=32768
)


def _resolve_reranker_model() -> str:
    configured = os.getenv("RERANKER_MODEL", BGE_RERANKER_MODEL).strip()
    if configured in SUPPORTED_RERANKER_MODELS:
        return configured

    logging.warning(
        f"Unsupported RERANKER_MODEL '{configured}', using {BGE_RERANKER_MODEL}"
    )
    return BGE_RERANKER_MODEL
```

- [ ] **Step 3: Update `RerankerWrapper`**

Replace the wrapper internals so the constructor chooses backend by `model_name` and `score()` keeps the existing method signature:

```python
class RerankerWrapper:
    """Encapsulate the selected reranker model for cross-encoder scoring."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.executor = ThreadPoolExecutor(max_workers=1)
        self._score_fn: Callable[[List[List[str]], bool], List[float]]

        if self.model_name == QWEN_RERANKER_MODEL:
            self._init_qwen()
        else:
            self._init_bge()

    def _init_bge(self) -> None:
        use_fp16 = has_cuda
        logging.info(
            f"Initializing BGE reranker '{self.model_name}' on '{device}' "
            f"with FP16: {use_fp16}"
        )
        self.model = FlagReranker(self.model_name, use_fp16=use_fp16)
        self._score_fn = self._score_bge

        if has_cuda:
            logging.info("Performing reranker warm-up...")
            _ = self.model.compute_score(
                [["warm-up query", "warm-up passage"]], normalize=False
            )
        logging.info("Reranker ready.")

    def _init_qwen(self) -> None:
        logging.info(
            f"Initializing Qwen reranker '{self.model_name}' on '{device}'"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, padding_side="left"
        )
        model_kwargs = {"torch_dtype": torch.float16} if has_cuda else {}
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, **model_kwargs
        ).to(device).eval()
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.prefix_tokens = self.tokenizer.encode(
            QWEN_RERANKER_PREFIX, add_special_tokens=False
        )
        self.suffix_tokens = self.tokenizer.encode(
            QWEN_RERANKER_SUFFIX, add_special_tokens=False
        )
        self._score_fn = self._score_qwen

        if has_cuda:
            logging.info("Performing reranker warm-up...")
            _ = self._score_qwen([["warm-up query", "warm-up passage"]], normalize=False)
        logging.info("Reranker ready.")

    def score(self, pairs: List[List[str]], normalize: bool) -> List[float]:
        return self._score_fn(pairs, normalize)

    def _score_bge(self, pairs: List[List[str]], normalize: bool) -> List[float]:
        raw = self.model.compute_score(pairs, normalize=normalize)
        return self._coerce_scores(raw)

    def _score_qwen(self, pairs: List[List[str]], normalize: bool) -> List[float]:
        formatted_pairs = [
            self._format_qwen_instruction(query, passage) for query, passage in pairs
        ]
        inputs = self._tokenize_qwen_pairs(formatted_pairs)

        with torch.no_grad():
            batch_scores = self.model(**inputs).logits[:, -1, :]
            true_vector = batch_scores[:, self.token_true_id]
            false_vector = batch_scores[:, self.token_false_id]
            binary_scores = torch.stack([false_vector, true_vector], dim=1)
            probabilities = torch.nn.functional.log_softmax(binary_scores, dim=1)
            return probabilities[:, 1].exp().detach().cpu().tolist()

    def _format_qwen_instruction(self, query: str, passage: str) -> str:
        return (
            f"<Instruct>: {QWEN_RERANKER_INSTRUCTION}\n"
            f"<Query>: {query}\n"
            f"<Document>: {passage}"
        )

    def _tokenize_qwen_pairs(self, formatted_pairs: List[str]) -> Dict[str, torch.Tensor]:
        max_pair_length = max(
            1,
            QWEN_RERANK_MAX_LENGTH
            - len(self.prefix_tokens)
            - len(self.suffix_tokens),
        )
        inputs = self.tokenizer(
            formatted_pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=max_pair_length,
        )
        for index, input_ids in enumerate(inputs["input_ids"]):
            inputs["input_ids"][index] = (
                self.prefix_tokens + input_ids + self.suffix_tokens
            )
        padded = self.tokenizer.pad(
            inputs,
            padding=True,
            return_tensors="pt",
            max_length=QWEN_RERANK_MAX_LENGTH,
        )
        return {key: value.to(device) for key, value in padded.items()}

    @staticmethod
    def _coerce_scores(raw) -> List[float]:
        if isinstance(raw, (int, float)):
            return [float(raw)]
        if hasattr(raw, "tolist"):
            raw = raw.tolist()
        return [float(score) for score in raw]
```

- [ ] **Step 4: Use resolved model at startup**

Replace:

```python
reranker = RerankerWrapper("BAAI/bge-reranker-v2-m3")
```

with:

```python
reranker = RerankerWrapper(_resolve_reranker_model())
```

- [ ] **Step 5: Expose selected reranker in health**

Add to the `/health` response:

```python
"reranker_model": reranker.model_name,
```

### Task 2: Interactive Startup Scripts

**Files:**
- Modify: `start_server.bat`
- Modify: `start_server.sh`
- Modify: `docker-compose.yml`
- Modify: `.env.example`
- Modify: `README.md`

- [ ] **Step 1: Add reranker prompt to `start_server.bat`**

Add after device validation and before CUDA detection:

```bat
echo ======================================
echo  Select reranker
echo ======================================
echo.
echo Do you want to use:
echo   [1] BGE  ^(BAAI/bge-reranker-v2-m3^)
echo   [2] QWEN ^(Qwen/Qwen3-Reranker-0.6B^)
echo.
set /p reranker_choice="Enter choice ^(1 or 2^): "
if "%reranker_choice%"=="2" (
    set "RERANKER_MODEL=Qwen/Qwen3-Reranker-0.6B"
) else (
    if not "%reranker_choice%"=="" if not "%reranker_choice%"=="1" echo [WARNING] Invalid choice, defaulting to BGE
    set "RERANKER_MODEL=BAAI/bge-reranker-v2-m3"
)
echo.
```

Add to the final summary:

```bat
echo  Reranker: %RERANKER_MODEL%
```

- [ ] **Step 2: Add reranker prompt to `start_server.sh`**

Add a function near `ask_gpu_or_cpu`:

```bash
ask_reranker() {
    echo "======================================" >&2
    echo "  Select reranker" >&2
    echo "======================================" >&2
    echo "" >&2
    echo "Do you want to use:" >&2
    echo "  [1] BGE  (BAAI/bge-reranker-v2-m3)" >&2
    echo "  [2] QWEN (Qwen/Qwen3-Reranker-0.6B)" >&2
    echo "" >&2
    read -r -p "Enter choice (1 or 2): " choice >&2

    case "$choice" in
        2) echo "Qwen/Qwen3-Reranker-0.6B" ;;
        1) echo "BAAI/bge-reranker-v2-m3" ;;
        *)
            echo "[WARNING] Invalid choice, defaulting to BGE" >&2
            echo "BAAI/bge-reranker-v2-m3"
            ;;
    esac
}
```

Call it after validation and before CUDA detection:

```bash
RERANKER_MODEL="$(ask_reranker)"
export RERANKER_MODEL
```

Add to the final summary:

```bash
echo "  Reranker: $RERANKER_MODEL"
```

- [ ] **Step 3: Pass model through Docker Compose**

Add under `environment:` in `docker-compose.yml`:

```yaml
      RERANKER_MODEL: ${RERANKER_MODEL:-BAAI/bge-reranker-v2-m3}
```

Change the Compose healthcheck `start_period` from `180s` to `300s`:

```yaml
      start_period: 300s
```

Do not pre-download Qwen in `Dockerfile`. The choice is exclusive, and forcing both BGE and Qwen downloads during every build would slow the default BGE path and increase image build requirements.

- [ ] **Step 4: Document launcher-managed environment variables**

Add to `.env.example` under model/inference:

```text
# Reranker model id selected by start_server.bat/start_server.sh.
# The launcher prompts for BGE or QWEN and sets this automatically.
# Defensive default: BAAI/bge-reranker-v2-m3
# RERANKER_MODEL=BAAI/bge-reranker-v2-m3

# Max token length used by Qwen/Qwen3-Reranker-0.6B CausalLM scoring.
# Applies only when QWEN is selected at startup.
QWEN_RERANK_MAX_LENGTH=8192
```

- [ ] **Step 5: Update README model wording**

Update the top model table to describe reranking as selectable:

```markdown
| **Reranking** | Interactive startup choice: `BAAI/bge-reranker-v2-m3` or `Qwen/Qwen3-Reranker-0.6B` |
```

Update startup documentation to mention the extra prompt:

```markdown
Both startup scripts always prompt for the reranker before launching the server.
Choose BGE to preserve the original behavior, or QWEN to use `Qwen/Qwen3-Reranker-0.6B`.
```

Add a Docker note:

```markdown
The first Docker startup with QWEN selected downloads `Qwen/Qwen3-Reranker-0.6B`
into the Hugging Face cache volume. Startup can take longer than BGE on an empty
cache; later runs reuse the cached model.
```

Update the `/rerank` example response section so `model_name` is described as the selected reranker, not always BGE:

```markdown
`model_name` reports the reranker selected at startup.
```

### Task 3: Focused Unit Tests

**Files:**
- Create: `test_reranker_wrapper.py`

- [ ] **Step 1: Add stdlib tests with fake model dependencies**

Create `test_reranker_wrapper.py`:

```python
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

    def convert_tokens_to_ids(self, token):
        return {"no": 0, "yes": 1}[token]

    def encode(self, text, add_special_tokens=False):
        return [10, 11]

    def __call__(self, pairs, **kwargs):
        self.encoded.extend(pairs)
        return {"input_ids": [[21, 22] for _ in pairs]}

    def pad(self, inputs, **kwargs):
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


class FakeTensor:
    def __init__(self, value):
        self.value = value

    def to(self, device):
        return self


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
            functional=types.SimpleNamespace(log_softmax=self._log_softmax)
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


def load_server(env=None):
    module_name = "bge_m3_server_under_test"
    sys.modules.pop(module_name, None)

    fake_flag_embedding = types.ModuleType("FlagEmbedding")
    fake_flag_embedding.BGEM3FlagModel = FakeBGEM3FlagModel
    fake_flag_embedding.FlagReranker = FakeFlagReranker

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = FakeAutoTokenizer
    fake_transformers.AutoModelForCausalLM = FakeAutoModelForCausalLM

    modules = {
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


class RerankerWrapperTests(unittest.TestCase):
    def test_resolve_reranker_model_defaults_to_bge_for_invalid_value(self):
        module = load_server({"RERANKER_MODEL": "invalid"})
        self.assertEqual(module._resolve_reranker_model(), module.BGE_RERANKER_MODEL)

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
        self.assertIn("<Query>: query", wrapper.tokenizer.encoded[0])
        self.assertIn("<Document>: passage", wrapper.tokenizer.encoded[0])

    def test_qwen_normalize_is_api_compatible_noop(self):
        module = load_server({"RERANKER_MODEL": "Qwen/Qwen3-Reranker-0.6B"})
        wrapper = module.RerankerWrapper(module.QWEN_RERANKER_MODEL)
        self.assertEqual(
            wrapper.score([["query", "passage"]], normalize=True),
            wrapper.score([["query", "passage"]], normalize=False),
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the focused unit tests**

Run:

```powershell
python test_reranker_wrapper.py
```

Expected: `OK`.

### Task 4: Verification

**Files:**
- Verify: `bge-m3_server.py`
- Verify: `start_server.sh`
- Verify: `start_server.bat`
- Verify: `docker-compose.cpu.yml`
- Verify: `README.md`

- [ ] **Step 1: Run Bash syntax check**

Run:

```bash
bash -n ./start_server.sh
```

Expected: exit code `0`.

- [ ] **Step 2: Run Python syntax check**

Run a targeted compile check:

```powershell
python -m py_compile bge-m3_server.py
```

Expected: exit code `0`.

- [ ] **Step 3: Verify Docker CPU overlay keeps base environment**

Run:

```powershell
docker compose -f docker-compose.yml -f docker-compose.cpu.yml config
```

Expected: the rendered `embedder.environment` contains both:

```yaml
CUDA_VISIBLE_DEVICES: "-1"
RERANKER_MODEL: BAAI/bge-reranker-v2-m3
```

- [ ] **Step 4: Review batch script prompt flow**

Inspect `start_server.bat` and confirm the reranker prompt appears before the final summary and before both local and Docker branches.

- [ ] **Step 5: Search for stale BGE-only reranker documentation**

Run:

```powershell
rg -n "BAAI/bge-reranker-v2-m3|BGE-Reranker-v2-m3|bge-reranker" README.md docs
```

Expected: remaining matches either describe BGE as one selectable option or document the default fallback.

- [ ] **Step 6: Smoke-run with existing server only if models are already cached**

Run:

```powershell
.\start_server.bat local cpu
```

Choose `1` for BGE and confirm `/health` returns:

```json
"reranker_model": "BAAI/bge-reranker-v2-m3"
```

Repeat with `2` for Qwen only if the model download is acceptable in the environment.
