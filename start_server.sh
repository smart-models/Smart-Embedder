#!/bin/bash
# BGE-M3 Embedding Server - Startup Script for Linux
# Usage: ./start_server.sh [local|docker] [cpu|gpu|auto]
# Default: docker with auto-detect (asks if GPU available)

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration
MODE="${1:-docker}"
DEVICE="${2:-auto}"
HOST="${HOST:-0.0.0.0}"
# PORT: shell env wins; else read from .env; else default 8000.
PORT="${PORT:-$(grep -E '^[[:space:]]*PORT=' .env 2>/dev/null | tail -n1 | cut -d= -f2 | sed 's/#.*//; s/[[:space:]]//g; s/"//g')}"
PORT="${PORT:-8000}"

# Normalize to lowercase
MODE=$(echo "$MODE" | tr '[:upper:]' '[:lower:]')
DEVICE=$(echo "$DEVICE" | tr '[:upper:]' '[:lower:]')

# Function to check if CUDA GPU is available
check_cuda() {
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null; then
            return 0  # CUDA GPU available
        fi
    fi
    return 1  # No CUDA GPU
}

# Function to ask user for GPU or CPU
ask_gpu_or_cpu() {
    echo "======================================" >&2
    echo "  CUDA GPU detected on this machine" >&2
    echo "======================================" >&2
    echo "" >&2
    echo "Do you want to run with:" >&2
    echo "  [1] GPU (faster inference)" >&2
    echo "  [2] CPU (compatible with all systems)" >&2
    echo "" >&2
    read -r -p "Enter choice (1 or 2): " choice >&2
    
    case "$choice" in
        1) echo "gpu" ;;
        2) echo "cpu" ;;
        *) 
            echo "[WARNING] Invalid choice, defaulting to CPU" >&2
            echo "cpu"
            ;;
    esac
}

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

get_gpu_memory_mib() {
    if ! command -v nvidia-smi &> /dev/null; then
        return 1
    fi

    nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null \
        | head -n 1 \
        | tr -d '[:space:]'
}

load_env_defaults_for_autotune() {
    if [[ ! -f ".env" ]]; then
        return
    fi

    while IFS='=' read -r key value; do
        case "$key" in
            QWEN_RERANK_MAX_LENGTH|QWEN_RERANK_BATCH_SIZE)
                if [[ -z "${!key:-}" ]]; then
                    value="${value%%#*}"
                    value="${value%\"}"
                    value="${value#\"}"
                    value="$(echo "$value" | xargs)"
                    if [[ -n "$value" ]]; then
                        export "$key=$value"
                    fi
                fi
                ;;
        esac
    done < <(grep -E '^[[:space:]]*(QWEN_RERANK_MAX_LENGTH|QWEN_RERANK_BATCH_SIZE)=' .env || true)
}

autotune_qwen_reranker_for_gpu() {
    if [[ "$DEVICE" != "gpu" || "$RERANKER_MODEL" != "Qwen/Qwen3-Reranker-0.6B" ]]; then
        return
    fi

    load_env_defaults_for_autotune

    local gpu_mem_mib
    gpu_mem_mib="$(get_gpu_memory_mib || true)"
    if [[ -z "$gpu_mem_mib" || ! "$gpu_mem_mib" =~ ^[0-9]+$ ]]; then
        echo "[WARNING] Could not detect GPU VRAM; keeping Qwen rerank defaults." >&2
        return
    fi

    local tuned_batch_size tuned_max_length
    if (( gpu_mem_mib <= 6144 )); then
        tuned_batch_size=4
        tuned_max_length=4096
    elif (( gpu_mem_mib <= 8192 )); then
        tuned_batch_size=8
        tuned_max_length=8192
    else
        tuned_batch_size=16
        tuned_max_length=8192
    fi

    if [[ -z "${QWEN_RERANK_BATCH_SIZE:-}" ]]; then
        export QWEN_RERANK_BATCH_SIZE="$tuned_batch_size"
        echo "[INFO] Auto-tuned QWEN_RERANK_BATCH_SIZE=$QWEN_RERANK_BATCH_SIZE for ${gpu_mem_mib}MiB VRAM"
    else
        echo "[INFO] Keeping QWEN_RERANK_BATCH_SIZE=$QWEN_RERANK_BATCH_SIZE (user/env override)"
    fi

    if [[ -z "${QWEN_RERANK_MAX_LENGTH:-}" ]]; then
        export QWEN_RERANK_MAX_LENGTH="$tuned_max_length"
        echo "[INFO] Auto-tuned QWEN_RERANK_MAX_LENGTH=$QWEN_RERANK_MAX_LENGTH for ${gpu_mem_mib}MiB VRAM"
    else
        echo "[INFO] Keeping QWEN_RERANK_MAX_LENGTH=$QWEN_RERANK_MAX_LENGTH (user/env override)"
    fi
}

autotune_qwen_reranker_for_cpu() {
    if [[ "$DEVICE" != "cpu" || "$RERANKER_MODEL" != "Qwen/Qwen3-Reranker-0.6B" ]]; then
        return
    fi

    load_env_defaults_for_autotune

    if [[ -z "${QWEN_RERANK_BATCH_SIZE:-}" ]]; then
        export QWEN_RERANK_BATCH_SIZE=1
        echo "[INFO] Auto-tuned QWEN_RERANK_BATCH_SIZE=$QWEN_RERANK_BATCH_SIZE for CPU mode"
    else
        echo "[INFO] Keeping QWEN_RERANK_BATCH_SIZE=$QWEN_RERANK_BATCH_SIZE (user/env override)"
    fi

    if [[ -z "${QWEN_RERANK_MAX_LENGTH:-}" ]]; then
        export QWEN_RERANK_MAX_LENGTH=2048
        echo "[INFO] Auto-tuned QWEN_RERANK_MAX_LENGTH=$QWEN_RERANK_MAX_LENGTH for CPU mode"
    else
        echo "[INFO] Keeping QWEN_RERANK_MAX_LENGTH=$QWEN_RERANK_MAX_LENGTH (user/env override)"
    fi
}

# Validate mode and device
if [[ "$MODE" != "local" && "$MODE" != "docker" ]]; then
    echo "[ERROR] Invalid mode: $MODE"
    echo "Usage: $0 [local|docker] [cpu|gpu|auto]"
    exit 1
fi

if [[ "$DEVICE" != "cpu" && "$DEVICE" != "gpu" && "$DEVICE" != "auto" ]]; then
    echo "[ERROR] Invalid device: $DEVICE"
    echo "Usage: $0 [local|docker] [cpu|gpu|auto]"
    exit 1
fi

DENSE_EMBEDDING_MODEL="$(ask_dense_embedding_model)"
export DENSE_EMBEDDING_MODEL

RERANKER_MODEL="$(ask_reranker)"
export RERANKER_MODEL

# Auto-detect device if not specified
if [[ "$DEVICE" == "auto" ]]; then
    if check_cuda; then
        DEVICE=$(ask_gpu_or_cpu)
    else
        echo "======================================"
        echo "  WARNING: No CUDA GPU detected"
        echo "======================================"
        echo ""
        echo "This machine does not have a CUDA-compatible GPU."
        echo "The server will run in CPU mode (slower but compatible)."
        echo ""
        read -r -p "Press Enter to continue..."
        DEVICE="cpu"
    fi
fi

autotune_qwen_reranker_for_gpu
autotune_qwen_reranker_for_cpu

echo "========================================"
echo "  BGE-M3 Embedding Server"
echo "  Mode:   $MODE"
echo "  Device: $DEVICE"
echo "  Dense:  $DENSE_EMBEDDING_MODEL"
echo "  Reranker: $RERANKER_MODEL"
if [[ "$RERANKER_MODEL" == "Qwen/Qwen3-Reranker-0.6B" ]]; then
    echo "  Qwen rerank batch: ${QWEN_RERANK_BATCH_SIZE:-16}"
    echo "  Qwen rerank max length: ${QWEN_RERANK_MAX_LENGTH:-8192}"
fi
echo "========================================"
echo ""

# Run local mode
if [[ "$MODE" == "local" ]]; then
    # Pick CPU-only requirements when running without GPU to avoid the CUDA torch wheel.
    REQ_FILE="requirements-gpu.txt"
    if [[ "$DEVICE" == "cpu" ]]; then
        REQ_FILE="requirements-cpu.txt"
    fi

    if [[ ! -f ".venv/bin/activate" ]]; then
        echo "[ERROR] Virtual environment not found!"
        echo ""
        echo "Create it first:"
        echo "  python -m venv .venv"
        echo "  source .venv/bin/activate"
        echo "  pip install -r $REQ_FILE"
        echo ""
        read -r -p "Press Enter to exit..."
        exit 1
    fi

    echo "[INFO] Activating virtual environment..."
    source .venv/bin/activate

    if ! python -c "import uvicorn, dotenv" 2>/dev/null; then
        echo "[ERROR] uvicorn or python-dotenv not found. Installing dependencies..."
        if ! pip install -r $REQ_FILE; then
            echo "[ERROR] Failed to install dependencies"
            read -r -p "Press Enter to exit..."
            exit 1
        fi
    fi

    if [[ "$DEVICE" == "cpu" ]]; then
        echo "[INFO] Forcing CPU mode (CUDA_VISIBLE_DEVICES=-1)"
        export CUDA_VISIBLE_DEVICES="-1"
    else
        echo "[INFO] GPU mode (CUDA auto-detect)"
        unset CUDA_VISIBLE_DEVICES
    fi

    UVICORN_ENV_ARGS=()
    if [[ -f ".env" ]]; then
        echo "[INFO] Loading environment from .env"
        UVICORN_ENV_ARGS=(--env-file .env)
    fi

    echo "[INFO] Binding host: $HOST"
    echo "[INFO] Starting server at http://localhost:$PORT"
    echo "[INFO] Docs:    http://localhost:$PORT/docs"
    echo "[INFO] Metrics: http://localhost:$PORT/metrics"
    echo "Press Ctrl+C to stop"
    echo ""

    uvicorn "${UVICORN_ENV_ARGS[@]}" bge-m3_server:app --host "$HOST" --port "$PORT"
    echo ""
    echo "[INFO] Server stopped"
    read -r -p "Press Enter to exit..."
    exit 0
fi

# Run docker mode
if [[ "$MODE" == "docker" ]]; then
    if ! command -v docker &> /dev/null; then
        echo "[ERROR] docker not found in PATH"
        read -r -p "Press Enter to exit..."
        exit 1
    fi

    COMPOSE_FILES="-f docker-compose.gpu.yml"
    
    if [[ "$DEVICE" == "cpu" ]]; then
        if [[ ! -f "docker-compose.cpu.yml" ]]; then
            echo "[ERROR] docker-compose.cpu.yml missing"
            read -r -p "Press Enter to exit..."
            exit 1
        fi
        COMPOSE_FILES="-f docker-compose.gpu.yml -f docker-compose.cpu.yml"
        echo "[INFO] Compose overlay: cpu"
    else
        echo "[INFO] Compose overlay: gpu (nvidia runtime)"
    fi

    echo "[INFO] Building image..."
    if ! docker compose $COMPOSE_FILES build; then
        echo "[ERROR] Build failed"
        read -r -p "Press Enter to exit..."
        exit 1
    fi

    echo "[INFO] Starting container..."
    if ! docker compose $COMPOSE_FILES up -d; then
        echo "[ERROR] Container start failed"
        read -r -p "Press Enter to exit..."
        exit 1
    fi

    echo ""
    echo "[INFO] Container started. Endpoints:"
    echo "  http://localhost:$PORT/health"
    echo "  http://localhost:$PORT/docs"
    echo "  http://localhost:$PORT/metrics"
    echo ""
    echo "[INFO] Tail logs: docker compose $COMPOSE_FILES logs -f"
    echo "[INFO] Stop:      docker compose $COMPOSE_FILES down"
    echo ""
    docker compose $COMPOSE_FILES ps
    exit 0
fi
