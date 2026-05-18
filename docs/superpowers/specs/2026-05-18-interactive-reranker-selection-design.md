# Interactive Reranker Selection Design

## Goal

Add Qwen/Qwen3-Reranker-0.6B as an exclusive reranker option selected interactively by `start_server.bat` and `start_server.sh`, while keeping the public API and existing embedding behavior unchanged.

## Decision

The startup scripts always ask which reranker to use:

- `BGE` maps to `BAAI/bge-reranker-v2-m3`.
- `QWEN` maps to `Qwen/Qwen3-Reranker-0.6B`.

The scripts pass the chosen model to the server via `RERANKER_MODEL`. This variable is an internal handoff from the interactive launcher to the Python process or Docker Compose service, not a user-facing alternative to the prompt.

## Server Architecture

`bge-m3_server.py` keeps one active reranker at process startup. The existing `/rerank` endpoint, request schema, response schema, ordering, backpressure, timeout, metrics, and authentication behavior stay unchanged.

`RerankerWrapper` selects its backend from `model_name`:

- For `BAAI/bge-reranker-v2-m3`, it uses the current `FlagEmbedding.FlagReranker` code path.
- For `Qwen/Qwen3-Reranker-0.6B`, it uses `sentence_transformers.CrossEncoder`.

The wrapper keeps one `score(pairs, normalize)` method. For BGE, `normalize` is passed to `compute_score`. For Qwen, raw scores come from `CrossEncoder.predict`; when `normalize=True`, sigmoid is applied to map scores to `[0, 1]`, matching the Hugging Face model card guidance.

## Startup Flow

The reranker prompt runs before the final server summary in both launch scripts. Invalid answers default to BGE because it preserves current behavior.

For local mode, the scripts set/export `RERANKER_MODEL` before `uvicorn`.

For Docker mode, the scripts set/export `RERANKER_MODEL` before `docker compose build` and `docker compose up -d`; `docker-compose.yml` passes it into the container with a BGE default as a defensive fallback.

## Observability

`/health` includes the selected reranker model as `reranker_model`. The existing rerank response already returns `model_name`, so clients can also see the selected reranker per request.

## Testing

Use focused unit tests with monkeypatched fake reranker backends to avoid downloading models. Tests verify:

- Default reranker model is BGE when `RERANKER_MODEL` is unset or invalid.
- Qwen backend calls `CrossEncoder.predict`.
- Qwen `normalize=True` applies sigmoid.
- BGE backend still calls `FlagReranker.compute_score`.

Script syntax checks cover `start_server.sh` with `bash -n`; the batch file is reviewed structurally because Windows batch has no equivalent parser available in this environment.
