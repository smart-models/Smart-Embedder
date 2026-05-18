# BGE-M3 Embedding & Reranker Server

High-performance FastAPI server for BGE-M3 embeddings and selectable reranking:

| Feature | Detail |
|---|---|
| **Embedding** | `BAAI/bge-m3` - dense, sparse, ColBERT vectors |
| **Reranking** | Interactive startup choice: `BAAI/bge-reranker-v2-m3` or `Qwen/Qwen3-Reranker-0.6B` |
| **Authentication** | Optional Bearer token on non-public endpoints |
| **Rate Limiting** | Token bucket, 3600 req/min per IP, burst 120 |
| **Backpressure** | Embedding queue max 200, rerank slots max 32, HTTP 503 on overflow |
| **Graceful Shutdown** | 30s drain for in-flight requests |
| **Prometheus Metrics** | Counters, histograms, gauges for both models |
| **Dynamic Batching** | Auto-tuned to GPU VRAM at startup |

## Quick Start

### 1. Setup

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Start the Server

`start_server.bat` and `start_server.sh` are parameterized and choose execution target and device:

```bat
start_server.bat [local|docker] [cpu|gpu|auto]
```

```bash
./start_server.sh [local|docker] [cpu|gpu|auto]
```

| Command | What it does |
|---|---|
| `start_server.bat` / `./start_server.sh` | Default: `docker auto` |
| `start_server.bat docker auto` | Docker startup with device auto-detection |
| `start_server.bat local gpu` | Local venv, CUDA auto-detect |
| `start_server.bat local cpu` | Local venv, forces CPU (`CUDA_VISIBLE_DEVICES=-1`) |
| `start_server.bat docker gpu` | `docker compose build && up -d` with NVIDIA runtime |
| `start_server.bat docker cpu` | Compose with override `docker-compose.cpu.yml` (no GPU) |

Arguments are case-insensitive. Built-in validation: unrecognized parameters print usage and exit with code 1.

Both startup scripts always prompt for the reranker before launching the server.
Choose BGE to preserve the original behavior, or QWEN to use `Qwen/Qwen3-Reranker-0.6B`.

**`local` mode**: requires `.venv` already created (see step 1). The script activates venv, checks for `uvicorn`, installs dependencies if missing, then starts the server.

**`docker` mode**: requires Docker Desktop / Engine in PATH. The script builds the image and starts the container in background. For logs:

```bat
docker compose logs -f embedder
```

In Docker Desktop the project appears as **bge-m3-embedder-reranker**.

Or directly without wrapper:

```bat
uvicorn bge-m3_server:app --host 0.0.0.0 --port 8000
```

Wait for these log lines:
```
INFO - Reranker ready.
INFO - Server ready to accept requests
```

### 3. Automatic Test

In a second terminal (with server running):

```bat
python test_server.py
```

Expected output: **16/16 tests passed**. With `--token` and `API_TOKEN`
configured, the authentication check is included and the expected output is
**17/17 tests passed**.

`test_server.py` accepts `--url` to point to a different host and `--token`
when `API_TOKEN` is configured:
```bat
python test_server.py --url http://localhost:8000
python test_server.py --token <token>
```

If payload limits were modified via env, pass expected values with
`--max-sentences`, `--max-sentence-chars`, `--max-total-chars`,
`--max-rerank-passages`, `--max-rerank-text-chars`,
`--max-rerank-total-chars`.

### 4. Benchmark

Measures latency (avg/p50/p95/p99) and throughput on `embed_dense`, `embed_full`, `rerank` scenarios:

```bat
python benchmark.py --concurrency 8 --requests 100 --batch-size 4
```

| Flag | Default | Description |
|---|---|---|
| `--url` | `http://localhost:8000` | Server target |
| `--token` | `API_TOKEN` env or empty | Bearer token if server requires auth |
| `--concurrency` | `8` | Concurrent requests in-flight |
| `--requests` | `100` | Requests per scenario |
| `--batch-size` | `4` | Sentences/passages per request |
| `--warmup` | `5` | Warmup requests (excluded from metrics) |
| `--timeout` | `60` | Timeout for single request |
| `--max-batch-size` | `128` | Local guardrail on payload limits; `0` disables |
| `--scenarios` | all | CSV: `embed_dense,embed_full,rerank` |
| `--sleep-between` | `0` | Pause between scenarios (use `65` if rate-limit active) |

> Note: Default rate limits (3600 req/min, burst 120) are tuned for benchmarks
> on a single client at `conc<=16`. For extreme stress testing:
> `RATE_LIMIT_REQUESTS_PER_MINUTE=1000000 docker compose up -d`.

Output: ASCII table with `Reqs / OK / Fail / Conc / Wall / Req/s / Units/s / Avg / P50 / P95 / P99 / Min / Max`.

**Latest measured run (RTX 4060 Laptop 8GB, batch=4, conc=8, `transformers==4.57.3`):**

| Scenario | Req/s | Units/s | P50 | P95 | P99 |
|---|---|---|---|---|---|
| `embed_dense` | 44.5 | 178 | 176.9ms | 185.4ms | 187.3ms |
| `embed_full` (dense+sparse+colbert) | 28.7 | 115 | 294.1ms | 350.6ms | 498.9ms |
| `rerank` | 37.8 | 151 | 205.7ms | 250.3ms | 263.8ms |

---

## Docker

### Prerequisites

- Docker Desktop / Docker Engine with Compose v2+
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

```bash
nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### First Startup

```bash
# Verify CUDA tag exists before building
docker pull nvidia/cuda:12.6.3-runtime-ubuntu22.04

# Build and GPU startup (first time: downloads embedding and selected reranker models)
docker compose up --build

# Or via bat wrapper (Windows)
start_server.bat docker gpu

# CPU execution (compose override)
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up --build
# Equivalent:
start_server.bat docker cpu
```

Wait for these log lines:
```
INFO - Reranker ready.
INFO - Server ready to accept requests
```

Server available at `http://localhost:8000`.  
Models are saved in the default named volume
`bge-m3-embedder-reranker-hf-cache` and mounted at `/app/model_cache`;
subsequent restarts do not re-download them.

The first Docker startup with QWEN selected downloads `Qwen/Qwen3-Reranker-0.6B`
into the Hugging Face cache volume. Startup can take longer than BGE on an empty
cache; later runs reuse the cached model.

### Useful Commands

```bash
# Startup in background
docker compose up -d

# Real-time logs
docker compose logs -f embedder

# Stop
docker compose down

# Rebuild after code changes (deps cached if requirements.txt unchanged)
docker compose up --build

# Complete rebuild from scratch
docker compose build --no-cache
```

### Verify GPU in Container

```bash
docker compose run --rm embedder python3 -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
"
```

### Exposure on Local Network

By default the server is bound to `127.0.0.1:8000` (localhost only).  
For LAN access modify `docker-compose.yml`:

```yaml
ports:
  - "8000:8000"
```

> Warning: If exposed on network, add a reverse proxy with authentication
> (nginx, Traefik).

---

## Project Files

| File | Description |
|---|---|
| `bge-m3_server.py` | Main server |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Docker image build (CUDA 12.6, non-root, hardened) |
| `docker-compose.yml` | Container orchestration with GPU and model volume |
| `docker-compose.cpu.yml` | Compose override: removes GPU reservation for CPU execution |
| `.env.example` | Environment variables template (copy to `.env` for local override) |
| `.dockerignore` | Excludes `.venv`, cache, docs from build context |
| `start_server.bat` | Windows startup script parameterized (`local\|docker` x `cpu\|gpu\|auto`) |
| `start_server.sh` | Unix shell startup script parameterized (`local\|docker` x `cpu\|gpu\|auto`) |
| `test_server.py` | Runtime test suite (16 checks, 17 with `--token`) |
| `benchmark.py` | Benchmark latency/throughput with summary table |

---

## API Endpoints

### `POST /embeddings/`

Generates embeddings for a list of texts.

**Request:**
```json
{
  "sentences": ["Hello world!", "Ciao mondo!"],
  "return_dense": true,
  "return_sparse": true,
  "return_colbert": true,
  "normalize_dense": false,
  "sparse_as_indices": false
}
```

**`sparse_as_indices` (default: `false`):** When `true`, sparse vectors are returned
in QDRANT-compatible format instead of the default token-id dict:

```json
"sparse": {"indices": [10, 1389, 2349], "values": [0.277, 0.292, 0.313]}
```

Use with `SparseVector(indices=..., values=...)` when upserting to QDRANT.

**Response:**
```json
{
  "data": [
    {
      "id": 0,
      "text": "Hello world!",
      "embeddings": {
        "dense": [0.021, -0.013, ...],
        "sparse": {"12": 0.08, "435": 0.12, ...},
        "colbert": [[0.01, ...], ...]
      }
    }
  ],
  "model_name": "BAAI/bge-m3",
  "processing_time_ms": 104.5
}
```

**cURL:**
```bash
curl -X POST "http://localhost:8000/embeddings/" \
  -H "Content-Type: application/json" \
  -d '{"sentences": ["Hello world!"], "return_dense": true}'
```

If `API_TOKEN` is set:
```bash
curl -X POST "http://localhost:8000/embeddings/" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"sentences": ["Hello world!"], "return_dense": true}'
```

---

### `POST /rerank`

Ranks a list of passages by relevance to a query.

**Request:**
```json
{
  "query": "What is machine learning?",
  "passages": [
    "Machine learning is a subset of AI.",
    "The weather is nice today.",
    "Deep learning uses neural networks."
  ],
  "normalize": true
}
```

**Response:**
```json
{
  "results": [
    {"index": 0, "passage": "Machine learning is a subset of AI.", "score": 0.987},
    {"index": 2, "passage": "Deep learning uses neural networks.", "score": 0.821},
    {"index": 1, "passage": "The weather is nice today.", "score": 0.003}
  ],
  "model_name": "BAAI/bge-reranker-v2-m3",
  "processing_time_ms": 52.2
}
```

- `normalize: true` returns a score in `[0, 1]` (sigmoid)
- `normalize: false` returns a raw score (negative values possible)
- With QWEN selected, scores are yes-probabilities and `normalize` is kept as an API-compatible no-op
- Do not compare BGE `normalize: false` raw logits directly with QWEN scores
- Passages are returned sorted by **descending** score
- The `index` field returns the original position in the input list
- `model_name` reports the reranker selected at startup

**cURL:**
```bash
curl -X POST "http://localhost:8000/rerank" \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning", "passages": ["ML is AI", "Nice weather"], "normalize": true}'
```

If `API_TOKEN` is set:
```bash
curl -X POST "http://localhost:8000/rerank" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning", "passages": ["ML is AI", "Nice weather"], "normalize": true}'
```

---

### `GET /health`

```bash
curl "http://localhost:8000/health"
```

Returns server status, GPU info, active model, batch size.

---

### `GET /stats`

```bash
curl "http://localhost:8000/stats"
```

Returns uptime, total requests, total sentences, total batches, rejected requests, hardware.

---

### `GET /metrics`

```bash
curl "http://localhost:8000/metrics"
```

Prometheus scraping endpoint in text/plain format.

---

### `GET /docs`

Interactive Swagger documentation: `http://localhost:8000/docs`

If `API_TOKEN` is configured, Swagger shows the lock on `POST` endpoints.
Use the **Authorize** button and enter only the token, without `Bearer` prefix.

---

## Configuration

Limits are tunable via **environment variable** (override in `docker-compose.yml` or shell before startup):

| Env var | Default | Description |
|---|---|---|
| `MAX_INPUT_LENGTH` | `2048` | Max tokens per sequence |
| `REQUEST_TIMEOUT` | `30` | Global HTTP timeout (sec) |
| `MAX_SENTENCES_PER_REQUEST` | `128` | Max sentences per embedding request |
| `MAX_SENTENCE_CHARS` | `20000` | Max chars per single embedding sentence |
| `MAX_TOTAL_CHARS_PER_REQUEST` | `250000` | Max total chars per embedding request |
| `MAX_RERANK_PASSAGES` | `128` | Max passages per rerank request |
| `MAX_RERANK_TEXT_CHARS` | `20000` | Max chars per rerank query/passage |
| `MAX_RERANK_TOTAL_CHARS` | `250000` | Max total chars per rerank request |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Reranker selected by launcher (`BAAI/bge-reranker-v2-m3` or `Qwen/Qwen3-Reranker-0.6B`) |
| `QWEN_RERANK_MAX_LENGTH` | `8192` | Max Qwen reranker token length when QWEN is selected |
| `API_TOKEN` | empty | Optional bearer token for non-public endpoints; empty disables authentication |
| `MAX_QUEUE_SIZE` | `200` | Max requests in queue `/embeddings/` (backpressure) |
| `RERANK_MAX_QUEUE` | `32` | Max concurrent slots for `/rerank` (backpressure) |
| `RERANK_GPU_TIMEOUT` | `15` | Hard timeout for a single rerank inference (sec) |
| `RATE_LIMIT_REQUESTS_PER_MINUTE` | `3600` | Rate limit per IP (60 req/s) |
| `RATE_LIMIT_BURST_SIZE` | `120` | Token bucket burst (~2s of traffic) |

With `API_TOKEN` set, all non-public endpoints require:

```http
Authorization: Bearer <token>
```

Service endpoints (`/health`, `/stats`, `/metrics`, `/docs`, `/redoc`, `/openapi.json`) remain accessible without token.

Defaults tuned for **NVIDIA RTX 4060 Laptop 8GB**: observed throughput
~29-45 req/s at `conc=8` depending on scenario (see benchmark table above).
Override:

```bash
# Ad-hoc (shell env)
RATE_LIMIT_REQUESTS_PER_MINUTE=10000 docker compose up -d

# Persistent - copy .env.example to .env and modify
cp .env.example .env
docker compose up -d
```

Compose automatically loads `.env` in the same directory. `.env` is in `.gitignore`; `.env.example` is the versioned template.

`MULTI_GPU_DEVICES = None` (in `bge-m3_server.py`) can be changed to
`['cuda:0', 'cuda:1']` for multi-GPU.

**Batch size** is automatically calculated from available VRAM:

| VRAM | batch_size | MAX_REQUESTS_IN_BATCH |
|---|---|---|
| > 8 GB | 128 | 64 |
| > 6 GB | 64 | 32 |
| > 4 GB | 32 | 16 |
| <= 4 GB | 16 | 16 |
| CPU | 1 | 8 |

---

## Prometheus Metrics

### Embedding

| Metric | Type | Label |
|---|---|---|
| `embedding_requests_total` | Counter | `status`, `endpoint` |
| `embedding_requests_rejected_total` | Counter | `reason` |
| `embedding_sentences_processed_total` | Counter | - |
| `embedding_request_duration_seconds` | Histogram | `endpoint` |
| `embedding_batch_size` | Histogram | - |
| `embedding_gpu_inference_duration_seconds` | Histogram | - |
| `embedding_queue_size` | Gauge | - |
| `embedding_active_requests` | Gauge | - |
| `embedding_gpu_memory_allocated_bytes` | Gauge | - |
| `embedding_server_info` | Info | `model`, `version`, `gpu_available`, `device` |

### Reranker

| Metric | Type | Label |
|---|---|---|
| `rerank_requests_total` | Counter | `status` |
| `rerank_requests_rejected_total` | Counter | `reason` |
| `rerank_pairs_processed_total` | Counter | - |
| `rerank_request_duration_seconds` | Histogram | - |
| `rerank_inference_duration_seconds` | Histogram | - |
| `rerank_active_requests` | Gauge | - |

### Useful PromQL Queries

```promql
# Throughput embedding (req/sec)
rate(embedding_requests_total[1m])

# Latency P95
histogram_quantile(0.95, rate(embedding_request_duration_seconds_bucket[5m]))

# Error rate (%)
rate(embedding_requests_total{status="error"}[5m]) / rate(embedding_requests_total[5m]) * 100

# GPU memory (GB)
embedding_gpu_memory_allocated_bytes / 1024 / 1024 / 1024

# Reranker throughput (pairs/sec)
rate(rerank_pairs_processed_total[1m])
```

### Setup Prometheus

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'bge-m3-embedder-reranker'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
```

### Grafana Dashboard - Recommended Panels

1. **Embedding Request Rate** - `rate(embedding_requests_total[1m])`
2. **Latency P50/P95/P99** - `histogram_quantile(0.X, ...)`
3. **Queue Size** - `embedding_queue_size`
4. **GPU Memory** - `embedding_gpu_memory_allocated_bytes`
5. **Rerank Request Rate** - `rate(rerank_requests_total[1m])`
6. **Batch Size Distribution** - `embedding_batch_size`

---

## Security and Limits

### Rate Limiting
- **Algorithm**: Token Bucket per IP
- **Limit**: `RATE_LIMIT_REQUESTS_PER_MINUTE=3600` req/min, `RATE_LIMIT_BURST_SIZE=120`
- **Response**: HTTP `429` with header `Retry-After: 60`

### Backpressure
- **/embeddings/ queue max**: `MAX_QUEUE_SIZE=200`
- **/rerank slots max**: `RERANK_MAX_QUEUE=32` (concurrency bound on reranker single-worker executor)
- **Acquire timeout**: 0.5s
- Rejections are reflected in both `/stats` (`rejected_requests`) and Prometheus (`embedding_requests_rejected_total` or `rerank_requests_rejected_total`, depending on endpoint).
- Rate limit uses direct connection IP (`request.client.host`). If the server is behind a trusted reverse proxy, update the middleware to extract IP from `X-Forwarded-For`.

### Timeout
- `REQUEST_TIMEOUT=30s` is the global HTTP timeout (504 to the caller).
- `GPU_PROCESS_TIMEOUT=15s` (CUDA) / `30s` (CPU) limits embedding batch inference on the thread pool.
- `RERANK_GPU_TIMEOUT=15s` limits rerank inference and should stay below `REQUEST_TIMEOUT`.
- Timeouts are tracked in Prometheus as `embedding_requests_total{status="timeout"}` or `rerank_requests_total{status="timeout"}`.

### Graceful Shutdown
- Blocks new requests (middleware)
- Waits for queue drain
- Completes in-flight requests (max 30s)
- Cancels processing loop and closes thread pools

---

## Troubleshooting

### Server Won't Start

```bat
python -c "import torch; print(torch.cuda.is_available())"
pip install -r requirements.txt --upgrade
```

### `429 Too Many Requests` Errors
Client exceeds rate limit. Increase `RATE_LIMIT_REQUESTS_PER_MINUTE` or reduce call frequency.

### `503 Service Unavailable` Errors
Queue is full. Increase `MAX_QUEUE_SIZE` or scale horizontally with a load balancer.

### `504 Gateway Timeout` Errors
Embedding inference exceeded `GPU_PROCESS_TIMEOUT` (15s on CUDA, 30s on CPU)
or rerank inference exceeded `RERANK_GPU_TIMEOUT`. Reduce batch size or check
GPU availability.

### Prometheus Metrics Not Visible
```bash
curl http://localhost:8000/metrics
```
Verify that target in `prometheus.yml` is reachable and that port 8000 is not blocked by firewall.

### Docker: GPU Not Detected in Container
```bash
# Verify NVIDIA Container Toolkit
docker run --rm --gpus all nvidia/cuda:12.6.3-runtime-ubuntu22.04 nvidia-smi
```
If it fails: reinstall NVIDIA Container Toolkit and restart Docker.

### Docker: CUDA Tag Not Found
```
Error: manifest for nvidia/cuda:12.6.3-runtime-ubuntu22.04 not found
```
Search correct tag on [hub.docker.com/r/nvidia/cuda/tags](https://hub.docker.com/r/nvidia/cuda/tags) and update first line of `Dockerfile`.

### Docker: Container Unhealthy on First Startup
Default Compose and Dockerfile healthchecks allow a 300s startup period for
first-run model downloads. On slow networks or empty caches, increase the
healthcheck start period above 300s in your custom Compose override:
```yaml
start_period: 300s
```

---

## References

- [BAAI/bge-m3 - Hugging Face](https://huggingface.co/BAAI/bge-m3)
- [BAAI/bge-reranker-v2-m3 - Hugging Face](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- [Qwen/Qwen3-Reranker-0.6B - Hugging Face](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)
- [FlagEmbedding - GitHub](https://github.com/FlagOpen/FlagEmbedding)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Prometheus Python Client](https://github.com/prometheus/client_python)

---

## License

Follows the selected model licenses (`BAAI/bge-m3`, `BAAI/bge-reranker-v2-m3`,
and optionally `Qwen/Qwen3-Reranker-0.6B`).
