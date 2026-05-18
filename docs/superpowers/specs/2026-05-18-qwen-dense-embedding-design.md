# Qwen Dense Embedding Selection Design

## Goal

Add `Qwen/Qwen3-Embedding-0.6B` as an optional dense embedding backend selected at server startup. When selected, Qwen transparently replaces only the dense vectors returned by `/embeddings/`; `BAAI/bge-m3` remains loaded and continues to provide sparse lexical weights and ColBERT vectors.

## Decisions

The public `/embeddings/` request schema stays unchanged. Clients keep sending `return_dense`, `return_sparse`, `return_colbert`, `normalize_dense`, and `sparse_as_indices` exactly as they do today.

The response becomes more explicit while preserving `model_name`:

- `model_name` reports the active dense embedding model.
- `dense_model_name` reports the source for `embeddings.dense`.
- `sparse_model_name` reports the source for `embeddings.sparse`.
- `colbert_model_name` reports the source for `embeddings.colbert`.

If Qwen dense is enabled and a request asks for all embedding types, the response is hybrid:

- `dense` comes from `Qwen/Qwen3-Embedding-0.6B`.
- `sparse` comes from `BAAI/bge-m3`.
- `colbert` comes from `BAAI/bge-m3`.

`normalize_dense` keeps its current API meaning for both dense backends:

- `false` returns raw dense vectors from the selected dense backend.
- `true` applies L2 normalization before returning dense vectors.

## Non-Goals

This change does not add a new endpoint, remove BGE-M3, alter reranker selection, introduce query/document-specific embedding prompts, or change sparse and ColBERT semantics.

The Qwen dense embedding model is independent from the existing Qwen reranker. The launchers may select Qwen for dense embedding, reranking, both, or neither.

## Architecture

`bge-m3_server.py` should move embedding inference behind a small orchestrator:

- `BgeM3EmbeddingBackend` owns `BGEM3FlagModel` and produces BGE dense, sparse, and ColBERT outputs.
- `QwenDenseEmbeddingBackend` owns `AutoTokenizer` and `AutoModel` for `Qwen/Qwen3-Embedding-0.6B` and produces only dense vectors.
- `EmbeddingService` owns the BGE backend and the selected dense backend. It exposes one `embed(...)` method with the same signature currently used by `RequestProcessor`.

`RequestProcessor` continues to batch requests by option tuple and calls one `.embed(...)` method. The change is internal to the object passed into the processor.

The service should avoid unnecessary inference:

- Dense-only request with Qwen selected: call Qwen only.
- Sparse/ColBERT-only request with Qwen selected: call BGE only.
- Hybrid request with Qwen selected: call Qwen for dense and BGE for sparse/ColBERT, then merge results.
- Any request with BGE dense selected: call BGE with the requested outputs, matching current behavior.

## Startup and Configuration

Add a new environment variable:

```text
DENSE_EMBEDDING_MODEL
```

Supported values:

```text
BAAI/bge-m3
Qwen/Qwen3-Embedding-0.6B
```

When the value is unset, empty, or unsupported, the server logs a warning and uses `BAAI/bge-m3`. This mirrors the existing reranker resolver behavior.

`start_server.bat` and `start_server.sh` should ask for the dense embedding backend before the existing reranker prompt:

```text
[1] BGE  (BAAI/bge-m3)
[2] QWEN (Qwen/Qwen3-Embedding-0.6B)
```

The scripts pass the selection through `DENSE_EMBEDDING_MODEL`. Docker Compose exposes the same variable with a defensive BGE default, so direct Compose startup keeps current behavior.

## Qwen Dense Inference

Use the direct Transformers path to avoid changing the existing `sentence-transformers` pin:

- `AutoTokenizer.from_pretrained(QWEN_DENSE_EMBEDDING_MODEL, padding_side="left")`
- `AutoModel.from_pretrained(QWEN_DENSE_EMBEDDING_MODEL, ...)`
- batch tokenization with `padding=True`, `truncation=True`, `max_length=MAX_INPUT_LENGTH`, and `return_tensors="pt"`
- move tensors to `device`
- run inference under `torch.no_grad()`
- use last-token pooling from `outputs.last_hidden_state[:, -1, :]`
- assert the final vector dimension is `1024`
- apply L2 normalization only when `normalize_dense=true`
- return data under `dense_vecs`, matching the existing BGE result structure

The Hugging Face model card for `Qwen/Qwen3-Embedding-0.6B` reports a 1024-dimensional embedding size and requires a Transformers version at least 4.51.0. This project already pins `transformers==4.57.3`.

Reference: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B

## API and Observability

`/embeddings/` keeps the same request body and per-item `embeddings` shape. The response includes the new model metadata fields.

`/health` should expose:

```json
{
  "model": "BAAI/bge-m3",
  "dense_embedding_model": "Qwen/Qwen3-Embedding-0.6B",
  "reranker_model": "BAAI/bge-reranker-v2-m3"
}
```

Prometheus `server_info` should include the dense embedding model so metrics identify the active dense backend.

## Error Handling

Unsupported `DENSE_EMBEDDING_MODEL` values fall back to `BAAI/bge-m3` with a warning.

If Qwen dense is explicitly selected and the model cannot load, server startup fails. Silent fallback would make vector provenance unclear and could corrupt downstream vector indexes.

If one backend fails during a hybrid request, the whole request fails. The server should not return partial embeddings.

If Qwen returns a dense vector whose final dimension is not 1024, the backend raises an explicit runtime error. This catches accidental model mismatch before vectors are inserted into 1024-dimensional collections.

## Testing

Use focused unit tests with fake model modules so tests do not download Hugging Face models.

Required coverage:

- default dense model resolves to `BAAI/bge-m3`
- unsupported dense model falls back to `BAAI/bge-m3`
- Qwen dense selection initializes the Qwen dense backend
- dense-only Qwen request calls only Qwen
- sparse/ColBERT-only request with Qwen configured calls only BGE
- hybrid request merges Qwen dense with BGE sparse and ColBERT outputs
- `normalize_dense=true` is honored for Qwen dense
- Qwen dense vector dimension is validated as 1024
- `/embeddings/` response includes `model_name`, `dense_model_name`, `sparse_model_name`, and `colbert_model_name`
- `/health` includes `dense_embedding_model`
- `docker-compose.yml` passes `DENSE_EMBEDDING_MODEL`
- `.env.example`, `start_server.bat`, `start_server.sh`, and `README.md` document the new selection

## Documentation

README should explain that embeddings can now be hybrid:

- BGE dense means the current all-BGE behavior.
- Qwen dense means `dense` is Qwen while `sparse` and `colbert` remain BGE-M3.

The configuration table should document `DENSE_EMBEDDING_MODEL`. Startup examples should mention that dense embedding selection and reranker selection are independent prompts.
