# Interactive Reranker Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an interactive BGE/Qwen reranker choice to the launch scripts while keeping `/rerank` unchanged.

**Architecture:** The launch scripts always prompt for a reranker and pass the selected Hugging Face model id through `RERANKER_MODEL`. `bge-m3_server.py` keeps one `RerankerWrapper` interface and chooses either `FlagReranker` or `sentence_transformers.CrossEncoder` internally.

**Tech Stack:** FastAPI, FlagEmbedding, sentence-transformers CrossEncoder, Docker Compose, Windows batch, Bash.

---

## File Structure

- Modify `bge-m3_server.py`: add reranker model selection helpers, add Qwen backend inside `RerankerWrapper`, expose `reranker_model` in `/health`.
- Modify `start_server.bat`: add mandatory reranker prompt and pass `RERANKER_MODEL` to local and Docker runs.
- Modify `start_server.sh`: add mandatory reranker prompt and export `RERANKER_MODEL` to local and Docker runs.
- Modify `docker-compose.yml`: pass `RERANKER_MODEL` into the container.
- Modify `.env.example`: document the internal variable as launcher-managed.
- Create or extend focused tests if a local unit test harness is available; otherwise verify by import-level monkeypatch snippets.

### Task 1: Server Reranker Backend Selection

**Files:**
- Modify: `bge-m3_server.py`

- [ ] **Step 1: Add imports and constants**

Add `Callable` to the typing import and import `CrossEncoder`:

```python
from typing import Callable, Dict, List, Optional, Union

from sentence_transformers import CrossEncoder
```

Add constants near the server configuration section:

```python
BGE_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
QWEN_RERANKER_MODEL = "Qwen/Qwen3-Reranker-0.6B"
SUPPORTED_RERANKER_MODELS = {
    BGE_RERANKER_MODEL,
    QWEN_RERANKER_MODEL,
}
```

- [ ] **Step 2: Add model resolver**

Add below `_env_int_range`:

```python
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
        model_kwargs = {}
        if has_cuda:
            model_kwargs["torch_dtype"] = torch.float16

        logging.info(
            f"Initializing Qwen reranker '{self.model_name}' on '{device}'"
        )
        self.model = CrossEncoder(
            self.model_name,
            device=device,
            model_kwargs=model_kwargs,
        )
        self._score_fn = self._score_qwen

        if has_cuda:
            logging.info("Performing reranker warm-up...")
            _ = self.model.predict([("warm-up query", "warm-up passage")])
        logging.info("Reranker ready.")

    def score(self, pairs: List[List[str]], normalize: bool) -> List[float]:
        return self._score_fn(pairs, normalize)

    def _score_bge(self, pairs: List[List[str]], normalize: bool) -> List[float]:
        raw = self.model.compute_score(pairs, normalize=normalize)
        return self._coerce_scores(raw)

    def _score_qwen(self, pairs: List[List[str]], normalize: bool) -> List[float]:
        tuple_pairs = [(query, passage) for query, passage in pairs]
        raw = self.model.predict(tuple_pairs)
        scores = self._coerce_scores(raw)
        if normalize:
            return [float(torch.sigmoid(torch.tensor(score)).item()) for score in scores]
        return scores

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
    if not "%reranker_choice%"=="1" echo [WARNING] Invalid choice, defaulting to BGE
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

- [ ] **Step 4: Document launcher-managed environment variable**

Add to `.env.example` under model/inference:

```text
# Reranker model id selected by start_server.bat/start_server.sh.
# The launcher prompts for BGE or QWEN and sets this automatically.
# Defensive default: BAAI/bge-reranker-v2-m3
# RERANKER_MODEL=BAAI/bge-reranker-v2-m3
```

### Task 3: Verification

**Files:**
- Verify: `bge-m3_server.py`
- Verify: `start_server.sh`
- Verify: `start_server.bat`

- [ ] **Step 1: Run Bash syntax check**

Run:

```bash
bash -n ./start_server.sh
```

Expected: exit code `0`.

- [ ] **Step 2: Run Python syntax/import check without model load if possible**

Run a targeted compile check:

```powershell
python -m py_compile bge-m3_server.py
```

Expected: exit code `0`.

- [ ] **Step 3: Review batch script prompt flow**

Inspect `start_server.bat` and confirm the reranker prompt appears before the final summary and before both local and Docker branches.

- [ ] **Step 4: Smoke-run with existing server only if models are already cached**

Run:

```powershell
.\start_server.bat local cpu
```

Choose `1` for BGE and confirm `/health` returns:

```json
"reranker_model": "BAAI/bge-reranker-v2-m3"
```

Repeat with `2` for Qwen only if the model download is acceptable in the environment.
