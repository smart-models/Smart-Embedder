# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04

LABEL org.opencontainers.image.title="bge-m3-embedder-reranker" \
      org.opencontainers.image.description="BGE-M3 embedding + reranker FastAPI server" \
      org.opencontainers.image.source="https://github.com/FlagOpen/FlagEmbedding"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/model_cache \
    HF_HUB_CACHE=/app/model_cache/hub

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

WORKDIR /app

# PyTorch isolated layer: heavy, cached separately from app dependencies
RUN python3 -m pip install \
        --extra-index-url https://download.pytorch.org/whl/cu126 \
        torch==2.7.0+cu126

COPY --chown=appuser:appuser requirements.txt .

RUN python3 -m pip install -r requirements.txt

COPY --chown=appuser:appuser bge-m3_server.py .

RUN mkdir -p /app/model_cache && chown -R appuser:appuser /app/model_cache

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "bge-m3_server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
