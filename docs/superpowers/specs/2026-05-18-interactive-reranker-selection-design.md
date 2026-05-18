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
- For `Qwen/Qwen3-Reranker-0.6B`, it uses `transformers.AutoTokenizer` and `transformers.AutoModelForCausalLM` with the yes/no logit scoring method from the Qwen model card.

The wrapper keeps one `score(pairs, normalize)` method. For BGE, `normalize` is passed to `compute_score`. For Qwen, the wrapper formats each `[query, passage]` pair with the Qwen instruction template, reads the final-token logits for `yes` and `no`, applies `log_softmax`, and returns the probability of `yes`. Because this scoring path already returns a probability in `[0, 1]`, `normalize` is accepted for API compatibility but does not change Qwen scores.

This avoids upgrading `sentence-transformers`. The Qwen repository currently includes Sentence Transformers integration metadata, but it was updated for newer Sentence Transformers releases; this project pins `sentence-transformers==2.7.0` to protect existing BGE reranker performance.

## Startup Flow

The reranker prompt runs before the final server summary in both launch scripts. Invalid answers default to BGE because it preserves current behavior.

For local mode, the scripts set/export `RERANKER_MODEL` before `uvicorn`.

For Docker mode, the scripts set/export `RERANKER_MODEL` before `docker compose build` and `docker compose up -d`; `docker-compose.yml` passes it into the container with a BGE default as a defensive fallback.

Qwen is not pre-downloaded during the Docker build. The first Qwen container startup may download the model into the existing Hugging Face cache volume, so Docker healthcheck timing must allow a longer first boot. This keeps BGE builds unchanged and avoids downloading both exclusive reranker options for every image build.

## Observability

`/health` includes the selected reranker model as `reranker_model`. The existing rerank response already returns `model_name`, so clients can also see the selected reranker per request.

## Testing

Use focused unit tests with monkeypatched fake reranker backends to avoid downloading models. Tests verify:

- Default reranker model is BGE when `RERANKER_MODEL` is unset or invalid.
- Qwen backend uses the tokenizer/model yes-no scorer path.
- Qwen `normalize=True` is API-compatible and returns the same probability scores.
- BGE backend still calls `FlagReranker.compute_score`.
- The Docker CPU overlay does not strip the base `RERANKER_MODEL` environment entry.
- Qwen first-boot Docker behavior is documented or healthcheck timing is adjusted.

Script syntax checks cover `start_server.sh` with `bash -n`; the batch file is reviewed structurally because Windows batch has no equivalent parser available in this environment.

## Documentation

Update README references that describe the reranker as exclusively `BAAI/bge-reranker-v2-m3`. The docs should state that embeddings always use `BAAI/bge-m3`, while reranking is selected interactively at startup.
