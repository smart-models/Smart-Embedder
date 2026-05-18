"""FastAPI server for BGE-M3 embeddings and reranking."""

import asyncio
import logging
import os
import secrets
import time
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Union
from uuid import uuid4

import numpy as np
import torch
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from FlagEmbedding import BGEM3FlagModel, FlagReranker
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)
from pydantic import BaseModel, Field, field_validator, model_validator
from starlette.status import HTTP_504_GATEWAY_TIMEOUT

# --- 1. Initial Configuration and Logging ---

# Configure structured logging instead of using print()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Server Configuration
# Multi-GPU configuration (optional - set to None for single GPU)
# Example: ['cuda:0', 'cuda:1'] for multi-GPU or None for single GPU
MULTI_GPU_DEVICES = None  # Change this to enable multi-GPU support
BGE_EMBEDDING_MODEL = "BAAI/bge-m3"
QWEN_DENSE_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
QWEN_DENSE_VECTOR_SIZE = 1024
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
SUPPORTED_DENSE_EMBEDDING_MODELS = {
    BGE_EMBEDDING_MODEL,
    QWEN_DENSE_EMBEDDING_MODEL,
}

# GPU detection and dynamic parameter configuration
has_cuda = torch.cuda.is_available()
device = "cuda" if has_cuda else "cpu"
available_gpus = torch.cuda.device_count() if has_cuda else 0


# Server configuration parameters (env-tunable)
def _env_int_range(
    name: str,
    default: int,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
) -> int:
    try:
        value = int(os.getenv(name, default))
    except ValueError:
        logging.warning(f"Invalid value for {name}, using default {default}")
        return default

    if min_value is not None and value < min_value:
        logging.warning(
            f"Value for {name} below minimum {min_value}, using default {default}"
        )
        return default

    if max_value is not None and value > max_value:
        logging.warning(
            f"Value for {name} above maximum {max_value}, using default {default}"
        )
        return default

    return value


MAX_INPUT_LENGTH = _env_int_range("MAX_INPUT_LENGTH", 2048, min_value=1, max_value=8192)
REQUEST_TIMEOUT = _env_int_range("REQUEST_TIMEOUT", 30, min_value=1, max_value=3600)
# Defaults tuned on RTX 4060 8GB: peak ~30-44 req/s @ conc=8 across scenarios
MAX_QUEUE_SIZE = _env_int_range("MAX_QUEUE_SIZE", 200, min_value=1, max_value=10000)
RATE_LIMIT_REQUESTS_PER_MINUTE = _env_int_range(
    "RATE_LIMIT_REQUESTS_PER_MINUTE", 3600, min_value=1, max_value=1000000
)
RATE_LIMIT_BURST_SIZE = _env_int_range(
    "RATE_LIMIT_BURST_SIZE", 120, min_value=1, max_value=1000000
)

# Payload limits protect the internal service from accidental oversized requests.
MAX_SENTENCES_PER_REQUEST = _env_int_range(
    "MAX_SENTENCES_PER_REQUEST", 128, min_value=1, max_value=10000
)
MAX_SENTENCE_CHARS = _env_int_range(
    "MAX_SENTENCE_CHARS", 20000, min_value=1, max_value=1000000
)
MAX_TOTAL_CHARS_PER_REQUEST = _env_int_range(
    "MAX_TOTAL_CHARS_PER_REQUEST", 250000, min_value=1, max_value=5000000
)
MAX_RERANK_PASSAGES = _env_int_range(
    "MAX_RERANK_PASSAGES", 128, min_value=1, max_value=10000
)
MAX_RERANK_TEXT_CHARS = _env_int_range(
    "MAX_RERANK_TEXT_CHARS", 20000, min_value=1, max_value=1000000
)
MAX_RERANK_TOTAL_CHARS = _env_int_range(
    "MAX_RERANK_TOTAL_CHARS", 250000, min_value=1, max_value=5000000
)
RERANK_MAX_QUEUE = _env_int_range("RERANK_MAX_QUEUE", 32, min_value=1, max_value=10000)
# Hard GPU timeout for a single rerank inference. Kept strictly below
# REQUEST_TIMEOUT so the executor surfaces a 504 before the HTTP layer would.
RERANK_GPU_TIMEOUT = _env_int_range(
    "RERANK_GPU_TIMEOUT", 15, min_value=1, max_value=3600
)
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


def _resolve_dense_embedding_model() -> str:
    configured = os.getenv("DENSE_EMBEDDING_MODEL", BGE_EMBEDDING_MODEL).strip()
    if configured in SUPPORTED_DENSE_EMBEDDING_MODELS:
        return configured

    logging.warning(
        f"Unsupported DENSE_EMBEDDING_MODEL '{configured}', "
        f"using {BGE_EMBEDDING_MODEL}"
    )
    return BGE_EMBEDDING_MODEL


# --- Authentication ---
# Bearer token for API access. Leave empty to disable authentication.
# When set, all non-public endpoints require: Authorization: Bearer <token>
# Health, stats, metrics, and API docs stay accessible for internal operations.
API_TOKEN = os.getenv("API_TOKEN", "").strip()
bearer_scheme = HTTPBearer(auto_error=False)
PUBLIC_AUTH_PATHS = {
    "/health",
    "/stats",
    "/metrics",
    "/docs",
    "/redoc",
    "/openapi.json",
}
PUBLIC_AUTH_PREFIXES = ("/docs/", "/redoc/")


def is_public_auth_path(path: str) -> bool:
    return path in PUBLIC_AUTH_PATHS or any(
        path.startswith(prefix) for prefix in PUBLIC_AUTH_PREFIXES
    )


async def require_bearer_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
):
    """Expose bearer authentication in OpenAPI and mirror middleware enforcement."""
    if not API_TOKEN:
        return

    if not credentials or not secrets.compare_digest(
        credentials.credentials, API_TOKEN
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )


if has_cuda:
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    logging.info(f"CUDA is available. GPU: {torch.cuda.get_device_name(0)}")
    logging.info(f"GPU Memory: {gpu_mem:.2f} GB")
    logging.info(f"Available GPUs: {available_gpus}")

    # Dynamically adjust batch size based on VRAM and max_length
    if MAX_INPUT_LENGTH <= 512:
        # Can handle larger batches with shorter sequences
        batch_size = (
            128 if gpu_mem > 8 else 64 if gpu_mem > 6 else 32 if gpu_mem > 4 else 16
        )
        MAX_REQUESTS_IN_BATCH = 64 if gpu_mem > 8 else 32 if gpu_mem > 6 else 16
        REQUEST_FLUSH_TIMEOUT = 0.01  # More aggressive batching
    else:
        # Original conservative settings for longer sequences
        batch_size = 12 if gpu_mem > 8 else 6 if gpu_mem > 4 else 3
        MAX_REQUESTS_IN_BATCH = 16
        REQUEST_FLUSH_TIMEOUT = 0.05

    GPU_PROCESS_TIMEOUT = 15
else:
    logging.info("CUDA not available. Using CPU.")
    batch_size = 1
    MAX_REQUESTS_IN_BATCH = 8
    REQUEST_FLUSH_TIMEOUT = 0.1
    GPU_PROCESS_TIMEOUT = 30


# --- Rate Limiter for Backpressure Protection ---


class RateLimiter:
    """Token bucket algorithm for rate limiting per client IP."""

    def __init__(self, requests_per_minute: int = 100, burst_size: int = 20):
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.buckets = defaultdict(
            lambda: {"tokens": burst_size, "last_update": datetime.now()}
        )

    def start_cleanup_task(self):
        """Schedule bucket cleanup inside a running event loop."""
        asyncio.create_task(self._cleanup_old_buckets())

    async def check_rate_limit(self, client_id: str) -> bool:
        """Check if client can make a request."""
        now = datetime.now()
        bucket = self.buckets[client_id]

        time_passed = (now - bucket["last_update"]).total_seconds()
        tokens_to_add = time_passed * (self.requests_per_minute / 60)

        bucket["tokens"] = min(self.burst_size, bucket["tokens"] + tokens_to_add)
        bucket["last_update"] = now

        if bucket["tokens"] >= 1:
            bucket["tokens"] -= 1
            return True
        return False

    async def _cleanup_old_buckets(self):
        """Remove unused buckets older than 1 hour."""
        while True:
            await asyncio.sleep(3600)
            now = datetime.now()
            cutoff = now - timedelta(hours=1)
            old_keys = [k for k, v in self.buckets.items() if v["last_update"] < cutoff]
            for k in old_keys:
                del self.buckets[k]


# --- 2. Model Wrapper ---


class BgeM3EmbeddingBackend:
    """Encapsulate the BGE-M3 model, handling loading and inference.

    This backend provides BGE-M3 dense, sparse, and ColBERT embeddings.
    """

    def __init__(
        self, model_name: str, device: str = device, devices: Optional[List[str]] = None
    ):
        self.model_name = model_name
        self.device = device
        self.devices = devices
        use_fp16 = self.device == "cuda" or (
            devices and any("cuda" in d for d in devices)
        )

        logging.info(
            f"Initializing model '{self.model_name}' on '{self.device}' "
            f"with FP16: {use_fp16}"
        )

        # Initialize with multiple devices if specified
        if devices:
            logging.info(f"Using multiple devices: {devices}")
            self.model = BGEM3FlagModel(
                self.model_name, devices=devices, use_fp16=use_fp16
            )
        else:
            self.model = BGEM3FlagModel(
                self.model_name, device=self.device, use_fp16=use_fp16
            )

        # Warm up CUDA kernels to keep the first request latency predictable.
        if self.device == "cuda" or (
            devices and any("cuda" in d for d in devices if d)
        ):
            logging.info("Performing model warm-up...")
            _ = self.model.encode(
                ["warm-up"],
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=True,
                max_length=MAX_INPUT_LENGTH,  # Use configured max length
            )
        logging.info("Model ready.")

        # Note: For PyTorch 2.0+, consider adding torch.compile for additional speedup:
        # if hasattr(torch, 'compile'):
        #     self.model.model = torch.compile(self.model.model)
        # This can provide 2-3x speedup but may increase startup time

    def embed(
        self,
        sentences: List[str],
        return_dense: bool = True,
        return_sparse: bool = True,
        return_colbert: bool = True,
        normalize_dense: bool = False,
    ) -> Dict[str, Union[np.ndarray, List]]:
        """Perform embedding of a list of sentences with configurable options.

        Generate embeddings for the provided sentences based on the specified options.
        Can selectively compute dense, sparse, and ColBERT embeddings, and optionally
        apply L2 normalization to dense vectors.

        Args:
            sentences: List of texts to process
            return_dense: If True, computes and returns dense vectors
            return_sparse: If True, computes and returns lexical weights (sparse)
            return_colbert: If True, computes and returns ColBERT vectors
            normalize_dense: If True, normalizes dense vectors (L2)

        Returns:
            Dictionary containing the requested embedding types
        """
        result = self.model.encode(
            sentences,
            batch_size=batch_size,
            max_length=MAX_INPUT_LENGTH,  # Use configured max length instead of 8192
            return_dense=return_dense,
            return_sparse=return_sparse,
            return_colbert_vecs=return_colbert,
        )

        if (
            return_dense
            and normalize_dense
            and "dense_vecs" in result
            and len(result["dense_vecs"]) > 0
        ):
            dense_vecs = np.array(result["dense_vecs"])
            norms = np.linalg.norm(dense_vecs, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)  # avoid division by zero
            result["dense_vecs"] = dense_vecs / norms

        return result


M3Wrapper = BgeM3EmbeddingBackend


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


# --- Prometheus Metrics ---

# Counters
requests_total = Counter(
    "embedding_requests_total",
    "Total number of embedding requests",
    ["status", "endpoint"],
)

requests_rejected = Counter(
    "embedding_requests_rejected_total",
    "Total rejected requests",
    ["reason"],
)

sentences_processed = Counter(
    "embedding_sentences_processed_total", "Total sentences processed"
)

# Histograms
request_duration = Histogram(
    "embedding_request_duration_seconds",
    "Request duration in seconds",
    ["endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

batch_size_histogram = Histogram(
    "embedding_batch_size",
    "Size of batches processed",
    buckets=[1, 2, 4, 8, 16, 32, 64, 128, 256],
)

gpu_inference_duration = Histogram(
    "embedding_gpu_inference_duration_seconds",
    "GPU inference time",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
)

# Gauges
queue_size = Gauge(
    "embedding_queue_size",
    (
        "Current number of embedding requests waiting in the batching queue "
        "(updated on enqueue and dequeue)"
    ),
)

active_requests = Gauge(
    "embedding_active_requests",
    (
        "Number of embedding requests currently queued or being processed "
        "(incremented on accept, decremented on completion)"
    ),
)

gpu_memory_allocated = Gauge(
    "embedding_gpu_memory_allocated_bytes", "GPU memory allocated in bytes"
)

# Info
server_info = Info("embedding_server", "Server information")

# --- Reranker Prometheus Metrics ---

rerank_requests_total = Counter(
    "rerank_requests_total",
    "Total number of rerank requests",
    ["status"],
)

rerank_pairs_processed = Counter(
    "rerank_pairs_processed_total", "Total query-passage pairs reranked"
)

rerank_request_duration = Histogram(
    "rerank_request_duration_seconds",
    "Rerank request duration in seconds",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

rerank_inference_duration = Histogram(
    "rerank_inference_duration_seconds",
    "Reranker model inference time",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
)

rerank_active_requests = Gauge(
    "rerank_active_requests",
    "Number of rerank requests currently holding a slot (queued plus in-flight)",
)

rerank_requests_rejected = Counter(
    "rerank_requests_rejected_total",
    "Total rejected rerank requests",
    ["reason"],
)


# --- 3. Pydantic Data Models for API (Structured Response) ---


class EmbedRequest(BaseModel):
    """Model for embedding request with configurable options.

    Defines the structure of the embedding request, including the text to embed
    and options for controlling which embedding types to compute and return.
    """

    sentences: List[str] = Field(..., max_length=MAX_SENTENCES_PER_REQUEST)
    return_dense: bool = True
    return_sparse: bool = True
    return_colbert: bool = True
    normalize_dense: bool = False
    sparse_as_indices: bool = False

    @field_validator("sentences")
    @classmethod
    def validate_sentences_size(cls, sentences: List[str]) -> List[str]:
        total_chars = 0

        for index, sentence in enumerate(sentences):
            sentence_length = len(sentence)
            if sentence_length > MAX_SENTENCE_CHARS:
                raise ValueError(
                    f"sentences[{index}] exceeds {MAX_SENTENCE_CHARS} characters"
                )
            total_chars += sentence_length

        if total_chars > MAX_TOTAL_CHARS_PER_REQUEST:
            raise ValueError(
                f"sentences total size exceeds {MAX_TOTAL_CHARS_PER_REQUEST} characters"
            )

        return sentences


class SparseIndicesVector(BaseModel):
    indices: List[int]
    values: List[float]


class EmbeddingVectors(BaseModel):
    """Group the three types of embedding vectors with optional fields.

    Contains the different embedding representations that can be generated for a text:
    dense vectors, sparse lexical weights, and ColBERT token-level embeddings.
    All fields are optional to support selective embedding generation.
    """

    dense: Optional[List[float]] = None
    sparse: Optional[Union[Dict[str, float], SparseIndicesVector]] = None
    colbert: Optional[List[List[float]]] = None


class SingleEmbeddingResponse(BaseModel):
    """Represent the complete result for a single text.

    Contains the original text, its position in the input list,
    and the generated embeddings in various formats.
    """

    id: int
    text: str
    embeddings: EmbeddingVectors


class EmbeddingsListResponse(BaseModel):
    """Main response object, with data and metadata.

    Contains a list of embedding results for all input texts,
    along with metadata about the model used and processing time.
    """

    model_config = {"protected_namespaces": ()}

    data: List[SingleEmbeddingResponse]
    model_name: str
    processing_time_ms: float


# --- Reranker Pydantic Models ---


class RerankRequest(BaseModel):
    """Model for a reranking request.

    Accepts a query and a list of passages (documents) to score.
    Returns passages sorted by relevance score descending.
    """

    query: str = Field(..., max_length=MAX_RERANK_TEXT_CHARS)
    passages: List[str] = Field(..., max_length=MAX_RERANK_PASSAGES)
    normalize: bool = False

    @field_validator("passages")
    @classmethod
    def validate_passages_size(cls, passages: List[str]) -> List[str]:
        for index, passage in enumerate(passages):
            if len(passage) > MAX_RERANK_TEXT_CHARS:
                raise ValueError(
                    f"passages[{index}] exceeds {MAX_RERANK_TEXT_CHARS} characters"
                )

        return passages

    @model_validator(mode="after")
    def validate_total_size(self):
        total_chars = len(self.query) + sum(len(passage) for passage in self.passages)
        if total_chars > MAX_RERANK_TOTAL_CHARS:
            raise ValueError(
                f"rerank payload total size exceeds {MAX_RERANK_TOTAL_CHARS} characters"
            )

        return self


class RankedPassage(BaseModel):
    """A single passage with its relevance score and original index."""

    index: int
    passage: str
    score: float


class RerankResponse(BaseModel):
    """Response for a reranking request.

    Contains passages sorted by relevance score (descending),
    along with metadata about the model and processing time.
    """

    model_config = {"protected_namespaces": ()}

    results: List[RankedPassage]
    model_name: str
    processing_time_ms: float


def lexical_weights_to_indices(lw: Dict[str, float]) -> SparseIndicesVector:
    """Convert BGE-M3 lexical_weights dict to QDRANT-compatible SparseIndicesVector.

    Uses list(items()) to guarantee indices and values iterate over the same
    ordered pairs (Python 3.7+ preserves dict insertion order, but explicit is safer).
    """
    items = list(lw.items())
    return SparseIndicesVector(
        indices=[int(k) for k, _ in items],
        values=[float(v) for _, v in items],
    )


# --- Reranker Wrapper ---


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
        logging.info(f"Initializing Qwen reranker '{self.model_name}' on '{device}'")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, padding_side="left"
        )
        # Qwen3-Reranker tokenizer ships without a pad_token; reuse EOS so
        # tokenizer.pad() does not raise. Required for batched scoring.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model_kwargs = {"dtype": torch.float16} if has_cuda else {}
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
        """Compute relevance scores for a list of [query, passage] pairs.

        Args:
            pairs: List of [query, passage] pairs to score
            normalize: Backend-compatible normalization flag

        Returns:
            List of float scores, one per pair
        """
        return self._score_fn(pairs, normalize)

    def _score_bge(self, pairs: List[List[str]], normalize: bool) -> List[float]:
        raw = self.model.compute_score(pairs, normalize=normalize)
        return self._coerce_scores(raw)

    def _score_qwen(self, pairs: List[List[str]], normalize: bool) -> List[float]:
        # Qwen returns yes-probabilities; keep normalize for API-compatible calls.
        _ = normalize
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

    def _tokenize_qwen_pairs(
        self, formatted_pairs: List[str]
    ) -> Dict[str, torch.Tensor]:
        max_pair_length = max(
            1,
            QWEN_RERANK_MAX_LENGTH
            - len(self.prefix_tokens)
            - len(self.suffix_tokens),
        )
        # Two-step encode+pad is required so prefix/suffix tokens can be
        # injected per-pair. Suppress the fast-tokenizer informational notice
        # that suggests collapsing it into a single __call__.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="You're using a Qwen2TokenizerFast tokenizer",
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
        # max_length is ignored by pad() without truncation; truncation was
        # already applied above, so omit it to silence the UserWarning.
        padded = self.tokenizer.pad(
            inputs,
            padding=True,
            return_tensors="pt",
        )
        return {key: value.to(device) for key, value in padded.items()}

    @staticmethod
    def _coerce_scores(raw) -> List[float]:
        if isinstance(raw, (int, float)):
            return [float(raw)]
        if hasattr(raw, "tolist"):
            raw = raw.tolist()
        return [float(score) for score in raw]


# --- 4. Asynchronous Request Processor with Batching ---


class RequestProcessor:
    """Manage dynamic batching of requests to maximize GPU throughput.

    Collects incoming embedding requests into batches to process them efficiently
    on the GPU. Implements asynchronous processing with configurable batch size
    and accumulation timeout.
    """

    def __init__(
        self,
        model_wrapper: Union[BgeM3EmbeddingBackend, EmbeddingService],
        max_batch_size: int,
        accumulation_timeout: float,
        stats=None,
        max_queue_size: int = MAX_QUEUE_SIZE,
    ):
        self.model_wrapper = model_wrapper
        self.max_batch_size = max_batch_size
        self.accumulation_timeout = accumulation_timeout
        self.queue = asyncio.Queue(
            maxsize=max_queue_size
        )  # BACKPRESSURE: Limited queue
        self.response_futures = {}
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.gpu_lock = asyncio.Semaphore(1)
        self.stats = stats
        self.is_shutting_down = False  # GRACEFUL SHUTDOWN: Flag

    async def start_processing_loop(self):
        """Start the main loop that processes the queue.

        Creates an asyncio task that continuously monitors the request queue
        and processes batches of requests.
        """
        logging.info("Starting the request processing loop.")
        self.processing_loop_task = asyncio.create_task(self._processing_loop())

    async def _processing_loop(self):
        """Collect requests and process them in batches.

        Main processing loop that accumulates requests up to the maximum batch size
        or until the accumulation timeout is reached, then processes the batch.
        """
        while not self.is_shutting_down:  # GRACEFUL SHUTDOWN: Check flag
            requests = []
            loop = asyncio.get_running_loop()
            start_time = loop.time()

            while len(requests) < self.max_batch_size:
                timeout = self.accumulation_timeout - (loop.time() - start_time)
                if timeout <= 0 or self.is_shutting_down:  # GRACEFUL SHUTDOWN
                    break
                try:
                    req_id, req_data = await asyncio.wait_for(
                        self.queue.get(), timeout=timeout
                    )
                    requests.append((req_id, req_data))
                    # PROMETHEUS: Reflect drained queue depth so the gauge does
                    # not stay stuck at the last enqueue-side value.
                    queue_size.set(self.queue.qsize())
                except asyncio.TimeoutError:
                    break

            if requests:
                # Group requests by their encoding options for more efficient processing
                option_groups = {}
                for req_id, req_data in requests:
                    options_key = (
                        req_data.return_dense,
                        req_data.return_sparse,
                        req_data.return_colbert,
                        req_data.normalize_dense,
                    )
                    if options_key not in option_groups:
                        option_groups[options_key] = []
                    option_groups[options_key].append((req_id, req_data))

                # Process each group with the same options
                for options_key, group_requests in option_groups.items():
                    all_sentences = [
                        s for _, req in group_requests for s in req.sentences
                    ]
                    request_ids = [req_id for req_id, _ in group_requests]

                    await self._run_model_on_batch(
                        all_sentences, request_ids, group_requests, options_key
                    )

    def _get_pending_future(self, request_id: str):
        future = self.response_futures.get(request_id)
        if future is None or future.cancelled() or future.done():
            return None
        return future

    def _set_response_exception(self, request_id: str, exc: Exception):
        future = self._get_pending_future(request_id)
        if future is not None:
            future.set_exception(exc)

    async def _run_model_on_batch(
        self, all_sentences, request_ids, requests, options_key
    ):
        """Run the model on the aggregated batch and distribute the results.

        Executes the model on a batch of sentences and distributes the results back
        to the individual requests. Uses the options from the options_key for the
        entire batch since all requests in this group have the same options.

        Args:
            all_sentences: List of all sentences from all requests in the batch
            request_ids: List of request IDs corresponding to the requests
            requests: List of (request_id, request_data) tuples
            options_key: Tuple of return_dense, return_sparse,
                return_colbert, and normalize_dense.
        """
        try:
            # Unpack the options from the key
            return_dense, return_sparse, return_colbert, normalize_dense = options_key
            options = {
                "return_dense": return_dense,
                "return_sparse": return_sparse,
                "return_colbert": return_colbert,
                "normalize_dense": normalize_dense,
            }

            # Track batch statistics if stats object is available
            if self.stats:
                self.stats.update_batch()

            # PROMETHEUS: Track batch size
            batch_size_histogram.observe(len(all_sentences))

            async with self.gpu_lock:
                # PROMETHEUS: Measure GPU time
                gpu_start = time.time()

                future = self.executor.submit(
                    self.model_wrapper.embed, all_sentences, **options
                )
                try:
                    embedding_data = await asyncio.wait_for(
                        asyncio.wrap_future(future), timeout=GPU_PROCESS_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    future.cancel()
                    raise

                # PROMETHEUS: Record GPU duration
                gpu_duration = time.time() - gpu_start
                gpu_inference_duration.observe(gpu_duration)

                # PROMETHEUS: Track GPU memory if CUDA
                if has_cuda:
                    gpu_memory_allocated.set(torch.cuda.memory_allocated())

            offset = 0
            for req_id, req_data in requests:
                num_sentences = len(req_data.sentences)
                future = self._get_pending_future(req_id)
                try:
                    if future is None:
                        continue

                    result_slice = {}

                    # Extract slices for this request only when data exists.
                    if "dense_vecs" in embedding_data and req_data.return_dense:
                        dense_slice = embedding_data["dense_vecs"][
                            offset : offset + num_sentences
                        ]
                        # Convert to list if it's a numpy array
                        result_slice["dense_vecs"] = (
                            dense_slice.tolist()
                            if hasattr(dense_slice, "tolist")
                            else dense_slice
                        )

                    if "lexical_weights" in embedding_data and req_data.return_sparse:
                        result_slice["lexical_weights"] = embedding_data[
                            "lexical_weights"
                        ][offset : offset + num_sentences]

                    if "colbert_vecs" in embedding_data and req_data.return_colbert:
                        colbert_slice = embedding_data["colbert_vecs"][
                            offset : offset + num_sentences
                        ]
                        # Convert to list if it's a numpy array
                        result_slice["colbert_vecs"] = (
                            colbert_slice.tolist()
                            if hasattr(colbert_slice, "tolist")
                            else colbert_slice
                        )

                    future.set_result(result_slice)
                except Exception as e:
                    logging.error(
                        f"Error preparing response for request {req_id}: {e}",
                        exc_info=True,
                    )
                    self._set_response_exception(req_id, e)
                finally:
                    offset += num_sentences

        except Exception as e:
            logging.error(f"Error during batch processing: {e}", exc_info=True)
            for req_id in request_ids:
                self._set_response_exception(req_id, e)

    async def process_request(self, request_data: EmbedRequest) -> Dict:
        """Add a new request to the queue and wait for its result.

        Creates a future for the request, adds it to the processing queue,
        and waits for the result to be set by the processing loop.

        Args:
            request_data: The embedding request to process

        Returns:
            Dictionary containing the embedding results
        """
        request_id = str(uuid4())
        self.response_futures[request_id] = asyncio.Future()

        # PROMETHEUS: Track active requests
        active_requests.inc()

        # BACKPRESSURE: Timeout on queue put
        try:
            await asyncio.wait_for(
                self.queue.put((request_id, request_data)), timeout=0.5
            )
        except asyncio.TimeoutError:
            # Queue full - reject request
            active_requests.dec()
            del self.response_futures[request_id]
            # PROMETHEUS: Track rejection
            requests_rejected.labels(reason="backpressure").inc()
            if self.stats:
                self.stats.increment_rejected()
            raise HTTPException(
                status_code=503, detail="Server is overloaded. Please retry later."
            )

        # PROMETHEUS: Track queue size and sentences
        queue_size.set(self.queue.qsize())
        sentences_processed.inc(len(request_data.sentences))

        # Track request statistics if stats object is available
        if self.stats:
            self.stats.update_request(len(request_data.sentences))

        try:
            return await self.response_futures[request_id]
        finally:
            if request_id in self.response_futures:
                del self.response_futures[request_id]
            # PROMETHEUS: Decrement active
            active_requests.dec()

    async def graceful_shutdown(self, timeout: float = 30.0):
        """Graceful shutdown with timeout."""
        logging.info("Starting graceful shutdown...")

        # 1. Signal to stop accepting new requests
        self.is_shutting_down = True

        # 2. Wait for queue to empty
        start_time = time.time()
        while not self.queue.empty():
            if time.time() - start_time > timeout:
                logging.warning(
                    f"Shutdown timeout: {self.queue.qsize()} requests dropped"
                )
                break
            await asyncio.sleep(0.1)

        # 3. Wait for active requests to complete
        active_futures = len(self.response_futures)
        if active_futures > 0:
            logging.info(f"Waiting for {active_futures} active requests...")
            remaining_time = timeout - (time.time() - start_time)
            if remaining_time > 0:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(
                            *self.response_futures.values(), return_exceptions=True
                        ),
                        timeout=remaining_time,
                    )
                except asyncio.TimeoutError:
                    logging.warning("Some requests did not complete in time")

        # 4. Cancel processing loop
        if hasattr(self, "processing_loop_task"):
            self.processing_loop_task.cancel()
            try:
                await self.processing_loop_task
            except asyncio.CancelledError:
                pass

        # 5. Shutdown thread pool
        logging.info("Shutting down thread pool executor...")
        self.executor.shutdown(wait=False, cancel_futures=True)

        logging.info("Graceful shutdown completed")


# --- 5. Server Statistics ---


class ServerStats:
    """Collect statistics on server usage.

    Tracks metrics about server usage including request counts,
    batch processing, and uptime for monitoring and diagnostics.
    """

    def __init__(self):
        self.start_time = time.time()
        self.total_requests = 0
        self.total_sentences = 0
        self.total_batches = 0
        self.gpu_info = self._get_gpu_info() if has_cuda else "CPU"
        self.rejected_requests = 0  # BACKPRESSURE: Track rejections

    def _get_gpu_info(self) -> str:
        """Get information about the GPU.

        Retrieves the GPU name and memory information if available.

        Returns:
            String containing GPU name and memory or status message
        """
        try:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            return f"{gpu_name} ({gpu_mem:.2f} GB)"
        except Exception:
            return "GPU not available"

    def update_request(self, num_sentences: int = 1):
        """Update statistics for a new request.

        Increments the request counter and adds to the total sentence count.

        Args:
            num_sentences: Number of sentences in the request
        """
        self.total_requests += 1
        self.total_sentences += num_sentences

    def update_batch(self):
        """Update the batch counter.

        Increments the counter tracking the number of batches processed.
        """
        self.total_batches += 1

    def increment_rejected(self):
        """Track rejected requests."""
        self.rejected_requests += 1

    def get_stats(self) -> Dict:
        """Return current server statistics.

        Calculates uptime and returns a dictionary with all tracked metrics.

        Returns:
            Dictionary containing server statistics
        """
        uptime_seconds = time.time() - self.start_time
        hours, remainder = divmod(uptime_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        return {
            "uptime": f"{int(hours)}h {int(minutes)}m {int(seconds)}s",
            "uptime_seconds": int(uptime_seconds),
            "total_requests": self.total_requests,
            "total_sentences": self.total_sentences,
            "total_batches": self.total_batches,
            "rejected_requests": self.rejected_requests,
            "hardware": self.gpu_info,
        }


# --- 6. FastAPI Application Instance and Lifecycle ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage server startup and shutdown within a single async context."""
    logging.info("Starting up server...")
    await processor.start_processing_loop()
    rate_limiter.start_cleanup_task()
    logging.info("Server ready to accept requests")
    yield
    logging.info("Shutting down server...")
    await processor.graceful_shutdown(timeout=30.0)
    reranker.executor.shutdown(wait=False, cancel_futures=True)


app = FastAPI(
    title="BGE-M3 Embedder & Reranker Server",
    version="1.0.0",
    lifespan=lifespan,
)

# RATE LIMITING: Initialize rate limiter
rate_limiter = RateLimiter(
    requests_per_minute=RATE_LIMIT_REQUESTS_PER_MINUTE,
    burst_size=RATE_LIMIT_BURST_SIZE,
)

# Load the models and request processor at startup.
model = M3Wrapper("BAAI/bge-m3", devices=MULTI_GPU_DEVICES)
reranker = RerankerWrapper(_resolve_reranker_model())
stats = ServerStats()
processor = RequestProcessor(
    model,
    max_batch_size=MAX_REQUESTS_IN_BATCH,
    accumulation_timeout=REQUEST_FLUSH_TIMEOUT,
    stats=stats,
)

# RERANK BACKPRESSURE: bound concurrent /rerank requests so the single-worker
# executor cannot accumulate unbounded waiting HTTP tasks under multi-client load.
rerank_slots = asyncio.Semaphore(RERANK_MAX_QUEUE)

# PROMETHEUS: Set server info
server_info.info(
    {
        "model": "BAAI/bge-m3",
        "version": "1.0.0",
        "gpu_available": str(has_cuda),
        "device": device,
    }
)


# AUTHENTICATION: Optional Bearer token for write/API endpoints
@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if not API_TOKEN or is_public_auth_path(request.url.path):
        return await call_next(request)

    auth_header = request.headers.get("Authorization", "")
    scheme, _, token = auth_header.partition(" ")

    if scheme.lower() != "bearer" or not secrets.compare_digest(token, API_TOKEN):
        requests_rejected.labels(reason="auth").inc()
        return JSONResponse(
            status_code=401,
            content={"detail": "Invalid or missing bearer token."},
            headers={"WWW-Authenticate": "Bearer"},
        )

    return await call_next(request)


# RATE LIMITING: Middleware for rate limiting
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    # Skip rate limiting for health and metrics endpoints
    if request.url.path in ["/health", "/stats", "/metrics"]:
        return await call_next(request)

    # Use the direct connection IP for rate limiting.
    # X-Forwarded-For is ignored because clients can spoof it to bypass limits.
    # If this server runs behind a trusted reverse proxy, replace this with the
    # verified first IP from X-Forwarded-For after validating the proxy address.
    client_ip = request.client.host if request.client else "unknown"

    # Check rate limit
    allowed = await rate_limiter.check_rate_limit(client_ip)

    if not allowed:
        # PROMETHEUS: Track rate limit rejection on the endpoint-specific counter
        # so dashboards split /rerank vs /embeddings rejections accurately.
        if request.url.path.startswith("/rerank"):
            rerank_requests_rejected.labels(reason="rate_limit").inc()
        else:
            requests_rejected.labels(reason="rate_limit").inc()
        # STATS: Keep /stats rejected_requests aligned with Prometheus counters
        # so operational dashboards reading either source see the same total.
        stats.increment_rejected()
        return JSONResponse(
            status_code=429,
            content={
                "detail": (
                    f"Rate limit exceeded. Max "
                    f"{RATE_LIMIT_REQUESTS_PER_MINUTE} requests per minute."
                ),
                "retry_after": 60,
            },
            headers={"Retry-After": "60"},
        )

    return await call_next(request)


# GRACEFUL SHUTDOWN: Middleware to block requests during shutdown
@app.middleware("http")
async def shutdown_middleware(request: Request, call_next):
    if processor.is_shutting_down:
        return JSONResponse(
            status_code=503, content={"detail": "Server is shutting down"}
        )
    return await call_next(request)


# Middleware for request timeout
@app.middleware("http")
async def timeout_middleware(request: Request, call_next):
    try:
        return await asyncio.wait_for(call_next(request), timeout=REQUEST_TIMEOUT)
    except asyncio.TimeoutError:
        return JSONResponse(
            {"detail": "Request timed out."},
            status_code=HTTP_504_GATEWAY_TIMEOUT,
        )


# --- 7. API Endpoints ---


@app.get("/health", status_code=200)
async def health_check():
    """Check the server status and GPU availability.

    Endpoint that verifies the server is running and provides information
    about the GPU availability and configuration.

    Returns:
        Dict: A dictionary with the server status and GPU information
    """
    gpu_status = {
        "available": has_cuda,
        "device_info": torch.cuda.get_device_name(0) if has_cuda else "CPU",
        "memory_gb": round(
            torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
        )
        if has_cuda
        else 0,
        "device_count": available_gpus if has_cuda else 0,
        "multi_gpu_enabled": MULTI_GPU_DEVICES is not None,
        "configured_devices": MULTI_GPU_DEVICES if MULTI_GPU_DEVICES else [device],
    }

    return {
        "status": "healthy",
        "gpu": gpu_status,
        "model": model.model_name,
        "reranker_model": reranker.model_name,
        "max_input_length": MAX_INPUT_LENGTH,
        "batch_size": batch_size,
        "max_requests_in_batch": MAX_REQUESTS_IN_BATCH,
    }


@app.get("/stats", status_code=200)
async def server_stats():
    """Return statistics on server usage.

    Provides metrics about server usage including request counts,
    batch processing efficiency, and uptime.

    Returns:
        Dict: A dictionary containing server usage statistics
    """
    return stats.get_stats()


@app.get("/metrics")
async def metrics():
    """PROMETHEUS: Endpoint for Prometheus metrics scraping."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post(
    "/embeddings/",
    response_model=EmbeddingsListResponse,
    status_code=200,
    dependencies=[Depends(require_bearer_token)],
)
async def get_embeddings(request: EmbedRequest):
    """Generate embeddings from input texts.

    Main endpoint that processes a list of texts and returns their embeddings.
    Supports configurable options for which embedding types to compute and return.

    Args:
        request: The embedding request containing texts and options

    Returns:
        EmbeddingsListResponse: Structured response with embeddings and metadata

    Raises:
        HTTPException: For validation errors, timeouts, or internal errors
    """
    # PROMETHEUS: Measure total request time
    with request_duration.labels(endpoint="/embeddings").time():
        if not request.sentences:
            # PROMETHEUS: Track error
            requests_total.labels(status="error", endpoint="/embeddings").inc()
            raise HTTPException(
                status_code=400, detail="The 'sentences' list cannot be empty."
            )

        try:
            start_time = time.time()

            # Process the request
            embedding_data = await processor.process_request(request)

            end_time = time.time()
            processing_time_ms = (
                end_time - start_time
            ) * 1000  # Convert to milliseconds

            # Build structured response with only the requested embedding types
            response_data = []
            for i, sentence in enumerate(request.sentences):
                # Create a dictionary with the requested embedding types
                embedding_kwargs = {}

                if request.return_dense and "dense_vecs" in embedding_data:
                    embedding_kwargs["dense"] = embedding_data["dense_vecs"][i]

                if request.return_sparse and "lexical_weights" in embedding_data:
                    lw = embedding_data["lexical_weights"][i]
                    embedding_kwargs["sparse"] = (
                        lexical_weights_to_indices(lw)
                        if request.sparse_as_indices
                        else lw
                    )

                if request.return_colbert and "colbert_vecs" in embedding_data:
                    embedding_kwargs["colbert"] = embedding_data["colbert_vecs"][i]

                vectors = EmbeddingVectors(**embedding_kwargs)

                response_data.append(
                    SingleEmbeddingResponse(id=i, text=sentence, embeddings=vectors)
                )

            # PROMETHEUS: Track success
            requests_total.labels(status="success", endpoint="/embeddings").inc()

            return EmbeddingsListResponse(
                data=response_data,
                model_name=model.model_name,
                processing_time_ms=round(processing_time_ms, 2),
            )

        except asyncio.TimeoutError:
            # PROMETHEUS: Track timeout
            requests_total.labels(status="timeout", endpoint="/embeddings").inc()
            raise HTTPException(
                status_code=504, detail="Timeout during GPU processing."
            )
        except HTTPException:
            # Re-raise HTTP exceptions (including backpressure 503)
            raise
        except Exception as e:
            logging.error(
                f"Unexpected error in /embeddings/ endpoint: {e}", exc_info=True
            )
            # PROMETHEUS: Track error
            requests_total.labels(status="error", endpoint="/embeddings").inc()
            raise HTTPException(status_code=500, detail="Internal server error.")


@app.post(
    "/rerank",
    response_model=RerankResponse,
    status_code=200,
    dependencies=[Depends(require_bearer_token)],
    description=(
        f"Score and rank passages by relevance to a query using "
        f"`{reranker.model_name}`.\n\n"
        "Accepts a query string and a list of candidate passages. "
        "Returns all passages sorted by descending relevance score.\n\n"
        "- BGE backend: `normalize=true` applies sigmoid to map raw logits "
        "into `[0, 1]`; `normalize=false` returns the raw logit "
        "(negative values possible).\n"
        "- Qwen backend: scores are yes-probabilities in `[0, 1]`; "
        "`normalize` is kept as an API-compatible no-op.\n\n"
        "Errors: HTTP 400 if query or passages list is empty; HTTP 503 on "
        "rerank queue backpressure; HTTP 504 on GPU timeout; HTTP 500 on "
        "internal model errors."
    ),
)
async def rerank(request: RerankRequest):
    """Score and rank passages using the reranker selected at startup."""
    with rerank_request_duration.time():
        if not request.query.strip():
            rerank_requests_total.labels(status="error").inc()
            raise HTTPException(
                status_code=400, detail="The 'query' field cannot be empty."
            )

        if not request.passages:
            rerank_requests_total.labels(status="error").inc()
            raise HTTPException(
                status_code=400, detail="The 'passages' list cannot be empty."
            )

        # RERANK BACKPRESSURE: bound concurrent slots so the single-worker
        # executor cannot accumulate unbounded HTTP tasks. Mirror /embeddings
        # 503 semantics so callers see consistent overload behavior.
        try:
            await asyncio.wait_for(rerank_slots.acquire(), timeout=0.5)
        except asyncio.TimeoutError:
            rerank_requests_rejected.labels(reason="backpressure").inc()
            stats.increment_rejected()
            rerank_requests_total.labels(status="error").inc()
            raise HTTPException(
                status_code=503, detail="Server is overloaded. Please retry later."
            )

        rerank_active_requests.inc()
        try:
            start_time = time.time()

            pairs = [[request.query, passage] for passage in request.passages]

            # PROMETHEUS: Track pairs
            rerank_pairs_processed.inc(len(pairs))

            # Run inference in the thread pool to avoid blocking the event loop.
            # Bound the GPU work explicitly so a runaway inference does not
            # outlive the HTTP request timeout; symmetric to the embedding
            # path's GPU_PROCESS_TIMEOUT in _run_model_on_batch.
            inference_start = time.time()
            loop = asyncio.get_running_loop()
            try:
                scores = await asyncio.wait_for(
                    loop.run_in_executor(
                        reranker.executor,
                        lambda: reranker.score(pairs, request.normalize),
                    ),
                    timeout=RERANK_GPU_TIMEOUT,
                )
            except asyncio.TimeoutError:
                rerank_requests_total.labels(status="timeout").inc()
                raise HTTPException(
                    status_code=504,
                    detail="Timeout during rerank GPU processing.",
                )
            rerank_inference_duration.observe(time.time() - inference_start)

            # Build response sorted by score descending
            ranked = sorted(
                [
                    RankedPassage(index=i, passage=passage, score=scores[i])
                    for i, passage in enumerate(request.passages)
                ],
                key=lambda r: r.score,
                reverse=True,
            )

            processing_time_ms = (time.time() - start_time) * 1000
            rerank_requests_total.labels(status="success").inc()

            return RerankResponse(
                results=ranked,
                model_name=reranker.model_name,
                processing_time_ms=round(processing_time_ms, 2),
            )

        except HTTPException:
            raise
        except Exception as e:
            logging.error(f"Unexpected error in /rerank endpoint: {e}", exc_info=True)
            rerank_requests_total.labels(status="error").inc()
            raise HTTPException(status_code=500, detail="Internal server error.")
        finally:
            rerank_active_requests.dec()
            rerank_slots.release()
