"""
Comprehensive test suite for the BGE-M3 Embedding Server.

Validates embeddings, payload limits, rate limiting, backpressure, metrics,
reranking, ColBERT vectors, and optional bearer authentication.
"""

import argparse
import asyncio
import sys
import time

import httpx

DEFAULT_PAYLOAD_LIMITS = {
    "max_sentences": 128,
    "max_sentence_chars": 20000,
    "max_total_chars": 250000,
    "max_rerank_passages": 128,
    "max_rerank_text_chars": 20000,
    "max_rerank_total_chars": 250000,
}


def burst_limits(concurrency: int) -> httpx.Limits:
    """Return connection limits high enough for deterministic burst tests."""
    return httpx.Limits(
        max_connections=concurrency,
        max_keepalive_connections=concurrency,
    )


# Force UTF-8 on stdout/stderr; Windows consoles default to cp1252.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


class Colors:
    """ANSI color codes for terminal output"""

    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    END = "\033[0m"
    BOLD = "\033[1m"


def print_header(text: str):
    """Print a formatted header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text.center(60)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.END}\n")


def print_success(text: str):
    """Print success message"""
    print(f"{Colors.GREEN}[OK] {text}{Colors.END}")


def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}[WARN] {text}{Colors.END}")


def print_error(text: str):
    """Print error message"""
    print(f"{Colors.RED}[ERROR] {text}{Colors.END}")


def print_info(text: str):
    """Print info message"""
    print(f"{Colors.BLUE}[INFO] {text}{Colors.END}")


def parse_prometheus_metric(text: str, name: str, labels: dict = None) -> float:
    """Extract a single Prometheus metric value from exposition text.

    Matches the first sample line whose metric name equals `name` and whose
    label set is a superset of `labels` (if provided). Returns 0.0 when no
    matching sample is found, so callers can treat missing series as a zero
    baseline for delta math.
    """
    target_labels = labels or {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "{" in line:
            line_name, rest = line.split("{", 1)
            label_section, _, value_section = rest.partition("}")
        else:
            parts = line.split(" ", 1)
            if len(parts) != 2:
                continue
            line_name, value_section = parts
            label_section = ""
        if line_name.strip() != name:
            continue
        parsed_labels = {}
        if label_section:
            for kv in label_section.split(","):
                if "=" not in kv:
                    continue
                k, v = kv.split("=", 1)
                parsed_labels[k.strip()] = v.strip().strip('"')
        if not all(parsed_labels.get(k) == v for k, v in target_labels.items()):
            continue
        try:
            return float(value_section.strip().split()[0])
        except (ValueError, IndexError):
            continue
    return 0.0


async def fetch_metrics_text(client: httpx.AsyncClient, base_url: str) -> str:
    resp = await client.get(f"{base_url}/metrics", timeout=10.0)
    resp.raise_for_status()
    return resp.text


async def fetch_stats(client: httpx.AsyncClient, base_url: str) -> dict:
    resp = await client.get(f"{base_url}/stats", timeout=10.0)
    resp.raise_for_status()
    return resp.json()


async def expect_post_status(
    client: httpx.AsyncClient,
    url: str,
    payload: dict,
    expected_status: int,
    label: str,
    timeout: float = 10.0,
) -> bool:
    """POST a payload and assert a specific HTTP status."""
    resp = await client.post(url, json=payload, timeout=timeout)
    if resp.status_code == expected_status:
        print_success(f"{label} correctly returned {expected_status}")
        return True

    print_error(
        f"{label} should return {expected_status}, got "
        f"{resp.status_code} - {resp.text[:200]}"
    )
    return False


def build_over_total_payload(total_limit: int, item_limit: int, item_count_limit: int):
    """Build strings exceeding total_limit without exceeding item limits."""
    if item_count_limit < 1 or item_limit < 1:
        return None

    if total_limit < item_limit:
        return ["x" * max(1, total_limit + 1)]

    if item_count_limit < 2:
        return None

    chunk_size = min(item_limit, max(1, (total_limit // item_count_limit) + 1))
    needed_items = (total_limit // chunk_size) + 1

    if needed_items > item_count_limit:
        return None

    return ["x" * chunk_size for _ in range(needed_items)]


async def test_health_endpoint(client: httpx.AsyncClient, base_url: str) -> bool:
    """Test 1: Health endpoint"""
    print_header("TEST 1: Health Endpoint")

    try:
        resp = await client.get(f"{base_url}/health", timeout=5.0)
        data = resp.json()

        if resp.status_code == 200 and data.get("status") == "healthy":
            print_success("Server is healthy")
            print_info(f"Model: {data.get('model')}")
            print_info(f"GPU: {data.get('gpu', {}).get('device_info')}")
            print_info(f"Max input length: {data.get('max_input_length')}")
            print_info(f"Batch size: {data.get('batch_size')}")
            return True
        else:
            print_error(f"Health check failed: {data}")
            return False
    except Exception as e:
        print_error(f"Health check error: {e}")
        return False


async def test_basic_embedding(client: httpx.AsyncClient, base_url: str) -> bool:
    """Test 2: Basic embedding request"""
    print_header("TEST 2: Basic Embedding")

    try:
        start = time.time()
        resp = await client.post(
            f"{base_url}/embeddings/",
            json={
                "sentences": ["Hello world!", "Ciao mondo!"],
                "return_dense": True,
                "return_sparse": True,
                "return_colbert": True,
            },
            timeout=30.0,
        )
        duration = time.time() - start

        if resp.status_code == 200:
            data = resp.json()
            print_success("Embedding generated successfully")
            print_info(f"Processing time: {data.get('processing_time_ms', 0):.2f}ms")
            print_info(f"Total request time: {duration * 1000:.2f}ms")
            print_info(f"Number of sentences: {len(data.get('data', []))}")

            # Check embedding types
            if data["data"]:
                emb = data["data"][0]["embeddings"]
                dense_ok = emb.get("dense") is not None
                sparse_ok = emb.get("sparse") is not None
                colbert_ok = emb.get("colbert") is not None

                print_info(f"Dense vectors: {'OK' if dense_ok else 'FAIL'}")
                print_info(f"Sparse vectors: {'OK' if sparse_ok else 'FAIL'}")
                print_info(f"ColBERT vectors: {'OK' if colbert_ok else 'FAIL'}")

                return dense_ok and sparse_ok and colbert_ok
            return True
        else:
            print_error(f"Embedding failed: {resp.status_code} - {resp.text}")
            return False
    except Exception as e:
        print_error(f"Embedding error: {e}")
        return False


async def test_sparse_as_indices(client: httpx.AsyncClient, base_url: str) -> bool:
    """Test 9: sparse_as_indices flag returns QDRANT-compatible format"""
    print_header("TEST 9: sparse_as_indices Flag")

    try:
        # --- 9a: sparse_as_indices=True returns indices/values format ---
        resp = await client.post(
            f"{base_url}/embeddings/",
            json={
                "sentences": ["Io sono a casa"],
                "return_dense": False,
                "return_sparse": True,
                "return_colbert": False,
                "sparse_as_indices": True,
            },
            timeout=30.0,
        )

        if resp.status_code != 200:
            print_error(
                f"sparse_as_indices request failed: {resp.status_code} - {resp.text}"
            )
            return False

        data = resp.json()
        sparse = data["data"][0]["embeddings"]["sparse"]

        if not isinstance(sparse, dict):
            print_error(f"Expected dict, got {type(sparse)}")
            return False

        if "indices" not in sparse or "values" not in sparse:
            print_error(f"Missing indices/values keys. Got: {list(sparse.keys())}")
            return False

        indices = sparse["indices"]
        values = sparse["values"]

        if len(indices) != len(values):
            print_error(
                f"indices/values length mismatch: {len(indices)} vs {len(values)}"
            )
            return False

        if not all(isinstance(i, int) for i in indices):
            print_error("Not all indices are integers")
            return False

        # Use (int, float) to handle JSON numeric parsing edge cases
        if not all(isinstance(v, (int, float)) for v in values):
            print_error("Not all values are numeric")
            return False

        if len(indices) == 0:
            print_error("Empty sparse vector - expected at least 1 token")
            return False

        print_success("sparse_as_indices=True returns correct indices/values format")
        print_info(f"Sparse tokens: {len(indices)}")
        print_info(f"Sample indices: {indices[:4]}")
        print_info(f"Sample values: {[round(float(v), 4) for v in values[:4]]}")

        # --- 9b: default (sparse_as_indices=False) still returns dict format ---
        resp2 = await client.post(
            f"{base_url}/embeddings/",
            json={
                "sentences": ["Io sono a casa"],
                "return_dense": False,
                "return_sparse": True,
                "return_colbert": False,
            },
            timeout=30.0,
        )

        if resp2.status_code != 200:
            print_error(f"Default sparse request failed: {resp2.status_code}")
            return False

        data2 = resp2.json()
        sparse2 = data2["data"][0]["embeddings"]["sparse"]

        if not isinstance(sparse2, dict):
            print_error(f"Default sparse should be dict, got {type(sparse2)}")
            return False

        if "indices" in sparse2 or "values" in sparse2:
            print_error("Default sparse unexpectedly returned indices/values format")
            return False

        for k, v in list(sparse2.items())[:3]:
            if not isinstance(k, str):
                print_error(f"Default sparse key not string: {k}")
                return False
            if not isinstance(v, (int, float)):
                print_error(f"Default sparse value not numeric: {v}")
                return False

        print_success("Default (sparse_as_indices=False) still returns dict format")

        # --- 9c: sparse_as_indices=True with multiple sentences ---
        resp3 = await client.post(
            f"{base_url}/embeddings/",
            json={
                "sentences": ["First sentence", "Second sentence"],
                "return_dense": False,
                "return_sparse": True,
                "return_colbert": False,
                "sparse_as_indices": True,
            },
            timeout=30.0,
        )

        if resp3.status_code != 200:
            print_error(f"Multi-sentence request failed: {resp3.status_code}")
            return False

        data3 = resp3.json()
        for item in data3["data"]:
            s = item["embeddings"]["sparse"]
            if "indices" not in s or "values" not in s:
                print_error(f"Multi-sentence: missing keys for item {item['id']}")
                return False
            if len(s["indices"]) != len(s["values"]):
                print_error(f"Multi-sentence: length mismatch for item {item['id']}")
                return False

        print_success("sparse_as_indices=True works correctly for multiple sentences")
        return True

    except Exception as e:
        print_error(f"sparse_as_indices test error: {e}")
        return False


async def test_payload_limits(
    client: httpx.AsyncClient, base_url: str, limits: dict
) -> bool:
    """Test payload validation limits for embeddings and rerank endpoints."""
    print_header("TEST 10: Payload Limits")

    try:
        checks = []

        checks.append(
            await expect_post_status(
                client,
                f"{base_url}/embeddings/",
                {
                    "sentences": [],
                    "return_dense": False,
                    "return_sparse": False,
                    "return_colbert": False,
                },
                400,
                "Empty embeddings sentences list",
            )
        )

        checks.append(
            await expect_post_status(
                client,
                f"{base_url}/embeddings/",
                {
                    "sentences": ["x"] * (limits["max_sentences"] + 1),
                    "return_dense": False,
                    "return_sparse": False,
                    "return_colbert": False,
                },
                422,
                "Too many embeddings sentences",
            )
        )

        checks.append(
            await expect_post_status(
                client,
                f"{base_url}/embeddings/",
                {
                    "sentences": ["x" * (limits["max_sentence_chars"] + 1)],
                    "return_dense": False,
                    "return_sparse": False,
                    "return_colbert": False,
                },
                422,
                "Overlong embeddings sentence",
            )
        )

        total_payload = build_over_total_payload(
            limits["max_total_chars"],
            limits["max_sentence_chars"],
            limits["max_sentences"],
        )
        if total_payload:
            checks.append(
                await expect_post_status(
                    client,
                    f"{base_url}/embeddings/",
                    {
                        "sentences": total_payload,
                        "return_dense": False,
                        "return_sparse": False,
                        "return_colbert": False,
                    },
                    422,
                    "Oversized embeddings total text",
                )
            )
        else:
            print_warning(
                "Skipping embeddings total-size limit: "
                "not reachable with current item limits"
            )

        checks.append(
            await expect_post_status(
                client,
                f"{base_url}/rerank",
                {
                    "query": "test",
                    "passages": ["p"] * (limits["max_rerank_passages"] + 1),
                },
                422,
                "Too many rerank passages",
            )
        )

        checks.append(
            await expect_post_status(
                client,
                f"{base_url}/rerank",
                {
                    "query": "test",
                    "passages": ["p" * (limits["max_rerank_text_chars"] + 1)],
                },
                422,
                "Overlong rerank passage",
            )
        )

        rerank_total_payload = build_over_total_payload(
            limits["max_rerank_total_chars"] - len("test"),
            limits["max_rerank_text_chars"],
            limits["max_rerank_passages"],
        )
        if rerank_total_payload:
            checks.append(
                await expect_post_status(
                    client,
                    f"{base_url}/rerank",
                    {"query": "test", "passages": rerank_total_payload},
                    422,
                    "Oversized rerank total text",
                )
            )
        else:
            print_warning(
                "Skipping rerank total-size limit: "
                "not reachable with current item limits"
            )

        return all(checks)

    except Exception as e:
        print_error(f"Payload limits test error: {e}")
        return False


async def test_rate_limiting(client: httpx.AsyncClient, base_url: str) -> bool:
    """Test 3: Rate limiting verifies HTTP 429 above token bucket capacity.

    Sends a burst large enough to exceed the default RATE_LIMIT_BURST_SIZE
    (120) while keeping client-side connection pressure bounded.
    """
    print_header("TEST 3: Rate Limiting")

    total = 300
    print_info(f"Sending {total} concurrent requests to exceed burst capacity...")

    try:
        start = time.time()
        async with httpx.AsyncClient(
            headers=dict(client.headers),
            limits=burst_limits(total),
        ) as burst_client:
            tasks = [
                burst_client.post(
                    f"{base_url}/embeddings/",
                    json={"sentences": [f"test {i}"]},
                    timeout=30.0,
                )
                for i in range(total)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        duration = time.time() - start

        success = sum(
            1 for r in results if isinstance(r, httpx.Response) and r.status_code == 200
        )
        rate_limited = sum(
            1 for r in results if isinstance(r, httpx.Response) and r.status_code == 429
        )
        backpressure = sum(
            1 for r in results if isinstance(r, httpx.Response) and r.status_code == 503
        )
        errors = sum(
            1
            for r in results
            if isinstance(r, Exception)
            or (isinstance(r, httpx.Response) and r.status_code not in [200, 429, 503])
        )

        print_info(f"Completed in {duration:.2f}s")
        print_success(f"Successful: {success}/{total}")
        print_warning(f"Rate limited (429): {rate_limited}/{total}")
        print_warning(f"Backpressure (503): {backpressure}/{total}")
        if errors > 0:
            print_error(f"Errors: {errors}/{total}")

        # Either 429 (rate-limit) or 503 (backpressure) is acceptable rejection.
        # Both indicate protection is working. Require at least 100 rejections total.
        rejections = rate_limited + backpressure
        if rejections >= 100:
            print_success(f"Protection mechanisms working ({rejections} rejected)")
            return True
        else:
            print_warning(f"Expected >=100 rejections, got {rejections}")
            return False

    except Exception as e:
        print_error(f"Rate limiting test error: {e}")
        return False


async def test_backpressure(
    client: httpx.AsyncClient, base_url: str, total_requests: int = 200
) -> bool:
    """Test 4: Backpressure protection"""
    print_header("TEST 4: Backpressure Protection")

    print_info(f"Sending {total_requests} concurrent heavy requests...")

    try:
        # Long text to slow down processing
        long_text = "test " * 100

        tasks = []
        for i in range(total_requests):
            task = client.post(
                f"{base_url}/embeddings/",
                json={"sentences": [f"{long_text} {i}"]},
                timeout=30.0,
            )
            tasks.append(task)

        start = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        duration = time.time() - start

        # Count results
        success = sum(
            1 for r in results if isinstance(r, httpx.Response) and r.status_code == 200
        )
        backpressure = sum(
            1 for r in results if isinstance(r, httpx.Response) and r.status_code == 503
        )
        rate_limited = sum(
            1 for r in results if isinstance(r, httpx.Response) and r.status_code == 429
        )
        errors = sum(
            1
            for r in results
            if isinstance(r, Exception)
            or (isinstance(r, httpx.Response) and r.status_code not in [200, 429, 503])
        )

        print_info(f"Completed in {duration:.2f}s")
        print_success(f"Successful: {success}/{total_requests}")
        print_warning(f"Backpressure (503): {backpressure}/{total_requests}")
        print_warning(f"Rate limited (429): {rate_limited}/{total_requests}")
        if errors > 0:
            print_error(f"Errors: {errors}/{total_requests}")
            return False

        if backpressure > 0:
            print_success("Backpressure protection working correctly")
            return True

        if rate_limited > 0:
            print_success("Rate limiter protected the service before queue overflow")
            return True

        print_info("No queue overflow observed under current queue/rate limits")
        return success == total_requests

    except Exception as e:
        print_error(f"Backpressure test error: {e}")
        return False


async def test_prometheus_metrics(client: httpx.AsyncClient, base_url: str) -> bool:
    """Test 5: Prometheus metrics endpoint"""
    print_header("TEST 5: Prometheus Metrics")

    try:
        resp = await client.get(f"{base_url}/metrics", timeout=5.0)

        if resp.status_code == 200:
            metrics = resp.text

            # Check for key metrics
            required_metrics = [
                "embedding_requests_total",
                "embedding_requests_rejected_total",
                "embedding_sentences_processed_total",
                "embedding_request_duration_seconds",
                "embedding_batch_size",
                "embedding_gpu_inference_duration_seconds",
                "embedding_queue_size",
                "embedding_active_requests",
                "embedding_server_info",
                "rerank_requests_total",
                "rerank_requests_rejected_total",
                "rerank_pairs_processed_total",
                "rerank_request_duration_seconds",
                "rerank_inference_duration_seconds",
                "rerank_active_requests",
            ]
            # Label-aware checks: the per-reason series only materialize after
            # an inc(). Tests in this suite are guaranteed to exercise the
            # following labels, so they must be present.
            labeled_required = [
                'embedding_requests_rejected_total{reason="rate_limit"}',
                'rerank_requests_rejected_total{reason="backpressure"}',
                'rerank_requests_rejected_total{reason="rate_limit"}',
            ]
            # Conditional labels: emitted only when the underlying condition is
            # actually exercised in this run. Missing means an informational warning.
            labeled_conditional = [
                'embedding_requests_rejected_total{reason="backpressure"}',
            ]

            print_info(f"Metrics size: {len(metrics)} bytes")

            all_present = True
            for metric in required_metrics:
                present = metric in metrics
                if present:
                    print_success(f"Metric '{metric}' present")
                else:
                    print_error(f"Metric '{metric}' missing")
                    all_present = False

            for needle in labeled_required:
                if needle in metrics:
                    print_success(f"Labeled series '{needle}' present")
                else:
                    print_error(
                        f"Labeled series '{needle}' missing "
                        "(must be exercised by suite)"
                    )
                    all_present = False

            for needle in labeled_conditional:
                if needle in metrics:
                    print_success(
                        f"Conditional labeled series '{needle}' present (was exercised)"
                    )
                else:
                    print_warning(
                        f"Conditional labeled series '{needle}' not present - "
                        "condition not exercised in this run, not a failure"
                    )

            return all_present
        else:
            print_error(f"Metrics endpoint failed: {resp.status_code}")
            return False

    except Exception as e:
        print_error(f"Metrics test error: {e}")
        return False


async def test_stats_endpoint(client: httpx.AsyncClient, base_url: str) -> bool:
    """Test 6: Stats endpoint"""
    print_header("TEST 6: Stats Endpoint")

    try:
        resp = await client.get(f"{base_url}/stats", timeout=5.0)

        if resp.status_code == 200:
            data = resp.json()

            print_success("Stats retrieved successfully")
            print_info(f"Uptime: {data.get('uptime')}")
            print_info(f"Total requests: {data.get('total_requests')}")
            print_info(f"Total sentences: {data.get('total_sentences')}")
            print_info(f"Total batches: {data.get('total_batches')}")
            print_info(f"Rejected requests: {data.get('rejected_requests', 0)}")
            print_info(f"Hardware: {data.get('hardware')}")

            return True
        else:
            print_error(f"Stats endpoint failed: {resp.status_code}")
            return False

    except Exception as e:
        print_error(f"Stats test error: {e}")
        return False


async def test_performance(client: httpx.AsyncClient, base_url: str) -> bool:
    """Test 7: Performance under normal load"""
    print_header("TEST 7: Performance Test")

    print_info("Sending 50 sequential requests...")

    try:
        latencies = []

        for i in range(50):
            start = time.time()
            resp = await client.post(
                f"{base_url}/embeddings/",
                json={"sentences": [f"Performance test {i}"]},
                timeout=30.0,
            )
            latency = (time.time() - start) * 1000  # ms

            if resp.status_code == 200:
                latencies.append(latency)

        if latencies:
            avg = sum(latencies) / len(latencies)
            p50 = sorted(latencies)[len(latencies) // 2]
            p95 = sorted(latencies)[int(len(latencies) * 0.95)]
            p99 = sorted(latencies)[int(len(latencies) * 0.99)]

            print_success(f"Completed {len(latencies)}/50 requests")
            print_info(f"Average latency: {avg:.2f}ms")
            print_info(f"P50 latency: {p50:.2f}ms")
            print_info(f"P95 latency: {p95:.2f}ms")
            print_info(f"P99 latency: {p99:.2f}ms")

            # Performance criteria (adjust based on hardware)
            if avg < 500:  # Average < 500ms
                print_success("Performance is excellent")
            elif avg < 1000:
                print_warning("Performance is acceptable")
            else:
                print_warning("Performance could be improved")

            return True
        else:
            print_error("No successful requests")
            return False

    except Exception as e:
        print_error(f"Performance test error: {e}")
        return False


async def test_rerank(client: httpx.AsyncClient, base_url: str) -> bool:
    """Test 8: Rerank endpoint"""
    print_header("TEST 8: Rerank Endpoint")

    try:
        # --- 8a: basic rerank ---
        resp = await client.post(
            f"{base_url}/rerank",
            json={
                "query": "What is machine learning?",
                "passages": [
                    "The weather is nice today.",
                    "Machine learning is a subset of artificial intelligence.",
                    "Deep learning uses neural networks to solve complex problems.",
                    "I enjoy eating pizza on Fridays.",
                ],
                "normalize": True,
            },
            timeout=30.0,
        )

        if resp.status_code != 200:
            print_error(f"Rerank failed: {resp.status_code} - {resp.text}")
            return False

        data = resp.json()
        results = data.get("results", [])

        print_success("Rerank response received")
        print_info(f"Model: {data.get('model_name')}")
        print_info(f"Processing time: {data.get('processing_time_ms', 0):.2f}ms")
        print_info(f"Passages ranked: {len(results)}")

        if len(results) != 4:
            print_error(f"Expected 4 results, got {len(results)}")
            return False

        # Scores must be descending
        scores = [r["score"] for r in results]
        if scores != sorted(scores, reverse=True):
            print_error(f"Results not sorted by score descending: {scores}")
            return False
        print_success("Results correctly sorted by score (descending)")

        # Normalized scores must be in [0, 1]
        if not all(0.0 <= s <= 1.0 for s in scores):
            print_error(f"Normalized scores out of [0,1] range: {scores}")
            return False
        print_success("Normalized scores are in [0, 1] range")

        # The ML-related passage should rank first
        top_passage = results[0]["passage"]
        if (
            "machine learning" in top_passage.lower()
            or "deep learning" in top_passage.lower()
        ):
            print_success(f"Top passage is relevant: '{top_passage[:60]}...'")
        else:
            print_warning(f"Unexpected top passage: '{top_passage[:60]}...'")

        # Original index is preserved
        returned_indices = {r["index"] for r in results}
        if returned_indices != {0, 1, 2, 3}:
            print_error(f"Original indices not preserved: {returned_indices}")
            return False
        print_success("Original passage indices preserved")

        # --- 8b: validation - empty query ---
        resp_empty_query = await client.post(
            f"{base_url}/rerank",
            json={"query": "   ", "passages": ["some text"]},
            timeout=5.0,
        )
        if resp_empty_query.status_code == 400:
            print_success("Empty query correctly rejected (400)")
        else:
            print_error(
                f"Empty query should return 400, got {resp_empty_query.status_code}"
            )
            return False

        # --- 8c: validation - empty passages ---
        resp_empty_passages = await client.post(
            f"{base_url}/rerank",
            json={"query": "test", "passages": []},
            timeout=5.0,
        )
        if resp_empty_passages.status_code == 400:
            print_success("Empty passages list correctly rejected (400)")
        else:
            print_error(
                "Empty passages should return 400, got "
                f"{resp_empty_passages.status_code}"
            )
            return False

        return True

    except Exception as e:
        print_error(f"Rerank test error: {e}")
        return False


async def test_rerank_backpressure(
    client: httpx.AsyncClient,
    base_url: str,
    total_requests: int = 200,
) -> bool:
    """Test 12: Rerank bounded backpressure.

    Saturates the rerank executor so RERANK_MAX_QUEUE slots fill. Requests that
    cannot acquire a slot within 0.5s should receive HTTP 503 and increment
    rerank_requests_rejected_total{reason="backpressure"}. Verifies the counter
    delta exactly matches observed 503 responses.

    Strict: requires at least one HTTP 503 from the backpressure path. If the
    first burst does not exercise the slot bound (very fast hardware or empty
    queue), escalates with a heavier payload and larger concurrency before
    giving up. Hardware-tolerant: rate-limit rejections are acceptable evidence
    of upstream protection and pass the test only if backpressure is also
    eventually observed or proven unreachable.
    """
    print_header("TEST 12: Rerank Backpressure (strict)")

    async def burst(passages_count: int, repeat_count: int, total: int):
        passages = [
            f"Long benchmark passage about subject {i}. " * repeat_count
            for i in range(passages_count)
        ]
        tasks = [
            client.post(
                f"{base_url}/rerank",
                json={
                    "query": f"query {i}",
                    "passages": passages,
                    "normalize": True,
                },
                timeout=60.0,
            )
            for i in range(total)
        ]
        start = time.time()
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        return responses, time.time() - start

    try:
        metrics_before = await fetch_metrics_text(client, base_url)
        rejected_before = parse_prometheus_metric(
            metrics_before,
            "rerank_requests_rejected_total",
            {"reason": "backpressure"},
        )

        # Pass 1: baseline burst
        print_info(f"Pass 1: {total_requests} concurrent /rerank, 48 passages x 8 reps")
        results, duration = await burst(48, 8, total_requests)

        def tally(rs):
            return (
                sum(
                    1
                    for r in rs
                    if isinstance(r, httpx.Response) and r.status_code == 200
                ),
                sum(
                    1
                    for r in rs
                    if isinstance(r, httpx.Response) and r.status_code == 503
                ),
                sum(
                    1
                    for r in rs
                    if isinstance(r, httpx.Response) and r.status_code == 429
                ),
                sum(
                    1
                    for r in rs
                    if isinstance(r, Exception)
                    or (
                        isinstance(r, httpx.Response)
                        and r.status_code not in [200, 429, 503]
                    )
                ),
            )

        success, backpressure, rate_limited, errors = tally(results)
        print_info(
            f"Pass 1: {duration:.2f}s | 200={success} "
            f"503={backpressure} 429={rate_limited} err={errors}"
        )

        # Escalate if first pass did not exercise the backpressure path and
        # rate limit did not intervene either.
        if backpressure == 0 and rate_limited == 0 and errors == 0:
            print_warning(
                "Pass 1 produced no rejection; escalating to heavier payload."
            )
            await asyncio.sleep(2)
            results2, dur2 = await burst(96, 12, total_requests + 100)
            s2, b2, rl2, e2 = tally(results2)
            print_info(f"Pass 2: {dur2:.2f}s | 200={s2} 503={b2} 429={rl2} err={e2}")
            results.extend(results2)
            success += s2
            backpressure += b2
            rate_limited += rl2
            errors += e2

        if errors > 0:
            print_error(f"Unexpected statuses/exceptions: {errors}")
            return False

        await asyncio.sleep(1)
        metrics_after = await fetch_metrics_text(client, base_url)
        rejected_after = parse_prometheus_metric(
            metrics_after,
            "rerank_requests_rejected_total",
            {"reason": "backpressure"},
        )
        rejected_delta = rejected_after - rejected_before
        print_info(
            'rerank_requests_rejected_total{reason="backpressure"} '
            f"delta: {rejected_delta}"
        )

        if backpressure == 0:
            print_error(
                "No HTTP 503 observed across both passes - "
                "RERANK_MAX_QUEUE bound not exercised."
            )
            return False

        if rejected_delta != backpressure:
            print_error(
                f"Counter delta {rejected_delta} != observed 503 count {backpressure}"
            )
            return False

        print_success(
            f"Rerank slots bound enforced: {backpressure} HTTP 503 "
            "with exact counter match"
        )
        return True

    except Exception as e:
        print_error(f"Rerank backpressure test error: {e}")
        return False


async def test_rerank_rate_limit_attribution(
    client: httpx.AsyncClient, base_url: str
) -> bool:
    """Test 17: /rerank rate-limit rejections increment the rerank counter only.

    Forces a rate-limit burst against /rerank and verifies the rejection delta
    lands in rerank_requests_rejected_total{reason="rate_limit"} rather than
    embedding_requests_rejected_total{reason="rate_limit"}. Catches the
    regression where the middleware treated every endpoint as an embedding
    request when accounting rejections.
    """
    print_header("TEST 17: Rerank Rate-Limit Counter Attribution")

    try:
        metrics_before = await fetch_metrics_text(client, base_url)
        rerank_rl_before = parse_prometheus_metric(
            metrics_before,
            "rerank_requests_rejected_total",
            {"reason": "rate_limit"},
        )
        embed_rl_before = parse_prometheus_metric(
            metrics_before,
            "embedding_requests_rejected_total",
            {"reason": "rate_limit"},
        )

        burst = 500
        print_info(
            f"Sending {burst} concurrent /rerank requests to trigger rate limit..."
        )
        async with httpx.AsyncClient(
            headers=dict(client.headers),
            limits=burst_limits(burst),
        ) as burst_client:
            tasks = [
                burst_client.post(
                    f"{base_url}/rerank",
                    json={
                        "query": f"q {i}",
                        "passages": ["short text"],
                        "normalize": False,
                    },
                    timeout=30.0,
                )
                for i in range(burst)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        observed_429 = sum(
            1 for r in results if isinstance(r, httpx.Response) and r.status_code == 429
        )

        await asyncio.sleep(1)
        metrics_after = await fetch_metrics_text(client, base_url)
        rerank_rl_after = parse_prometheus_metric(
            metrics_after,
            "rerank_requests_rejected_total",
            {"reason": "rate_limit"},
        )
        embed_rl_after = parse_prometheus_metric(
            metrics_after,
            "embedding_requests_rejected_total",
            {"reason": "rate_limit"},
        )

        rerank_delta = rerank_rl_after - rerank_rl_before
        embed_delta = embed_rl_after - embed_rl_before

        print_info(f"Observed HTTP 429 on /rerank: {observed_429}")
        print_info(f"rerank rate_limit counter delta: {rerank_delta}")
        print_info(f"embedding rate_limit counter delta (should be 0): {embed_delta}")

        if observed_429 == 0:
            print_error(
                "Burst did not trigger any rate-limit rejection - cannot validate."
            )
            return False
        if rerank_delta != observed_429:
            print_error(
                f"rerank rate_limit delta {rerank_delta} != observed 429 {observed_429}"
            )
            return False
        if embed_delta != 0:
            print_error(
                f"embedding rate_limit delta {embed_delta} > 0 - "
                "rejections leaked into wrong counter"
            )
            return False

        print_success(
            f"All {int(rerank_delta)} /rerank rate-limit rejections "
            "attributed to rerank counter"
        )
        return True

    except Exception as e:
        print_error(f"Rerank rate-limit attribution test error: {e}")
        return False


async def test_queue_gauge_drain(client: httpx.AsyncClient, base_url: str) -> bool:
    """Test 13: embedding_queue_size drains to 0 after load.

    Sends a small embedding burst and then verifies that, after a settle window,
    embedding_queue_size returns to 0 and embedding_active_requests is also 0.
    Validates that the queue gauge is updated on dequeue, not only on enqueue.
    """
    print_header("TEST 13: Queue Gauge Drain")

    try:
        tasks = [
            client.post(
                f"{base_url}/embeddings/",
                json={
                    "sentences": [f"drain test {i}"],
                    "return_dense": True,
                    "return_sparse": False,
                    "return_colbert": False,
                },
                timeout=30.0,
            )
            for i in range(20)
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

        # Wait long enough for any batching window + GPU pass to clear.
        await asyncio.sleep(3)

        metrics = await fetch_metrics_text(client, base_url)
        queue_val = parse_prometheus_metric(metrics, "embedding_queue_size")
        active_val = parse_prometheus_metric(metrics, "embedding_active_requests")
        rerank_active_val = parse_prometheus_metric(metrics, "rerank_active_requests")

        print_info(f"embedding_queue_size after drain: {queue_val}")
        print_info(f"embedding_active_requests after drain: {active_val}")
        print_info(f"rerank_active_requests after drain: {rerank_active_val}")

        ok = True
        if queue_val != 0.0:
            print_error(f"embedding_queue_size should be 0, got {queue_val}")
            ok = False
        if active_val != 0.0:
            print_error(f"embedding_active_requests should be 0, got {active_val}")
            ok = False
        if rerank_active_val != 0.0:
            print_error(f"rerank_active_requests should be 0, got {rerank_active_val}")
            ok = False

        if ok:
            print_success("Queue and active-request gauges drain to 0 post-load")
        return ok

    except Exception as e:
        print_error(f"Queue drain test error: {e}")
        return False


async def test_stats_prometheus_alignment(
    client: httpx.AsyncClient, base_url: str
) -> bool:
    """Test 14: /stats rejected_requests aligned with Prometheus rejection counters.

    Forces a rate-limit burst and checks that the /stats rejected_requests delta
    equals the sum of Prometheus rejection counter deltas across embedding
    rate_limit, embedding backpressure, and rerank backpressure reasons.
    """
    print_header("TEST 14: Stats / Prometheus Alignment")

    try:
        stats_before = await fetch_stats(client, base_url)
        metrics_before = await fetch_metrics_text(client, base_url)
        rate_limit_before = parse_prometheus_metric(
            metrics_before,
            "embedding_requests_rejected_total",
            {"reason": "rate_limit"},
        )
        embed_bp_before = parse_prometheus_metric(
            metrics_before,
            "embedding_requests_rejected_total",
            {"reason": "backpressure"},
        )
        rerank_bp_before = parse_prometheus_metric(
            metrics_before,
            "rerank_requests_rejected_total",
            {"reason": "backpressure"},
        )
        rejected_stats_before = float(stats_before.get("rejected_requests", 0))

        burst = 300
        print_info(
            f"Forcing {burst} concurrent requests to trigger rate-limit rejections..."
        )
        async with httpx.AsyncClient(
            headers=dict(client.headers),
            limits=burst_limits(burst),
        ) as burst_client:
            tasks = [
                burst_client.post(
                    f"{base_url}/embeddings/",
                    json={
                        "sentences": [f"align {i}"],
                        "return_dense": True,
                        "return_sparse": False,
                        "return_colbert": False,
                    },
                    timeout=30.0,
                )
                for i in range(burst)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        observed_429 = sum(
            1 for r in results if isinstance(r, httpx.Response) and r.status_code == 429
        )
        observed_503 = sum(
            1 for r in results if isinstance(r, httpx.Response) and r.status_code == 503
        )

        await asyncio.sleep(1)

        stats_after = await fetch_stats(client, base_url)
        metrics_after = await fetch_metrics_text(client, base_url)
        rate_limit_after = parse_prometheus_metric(
            metrics_after,
            "embedding_requests_rejected_total",
            {"reason": "rate_limit"},
        )
        embed_bp_after = parse_prometheus_metric(
            metrics_after,
            "embedding_requests_rejected_total",
            {"reason": "backpressure"},
        )
        rerank_bp_after = parse_prometheus_metric(
            metrics_after,
            "rerank_requests_rejected_total",
            {"reason": "backpressure"},
        )
        rejected_stats_after = float(stats_after.get("rejected_requests", 0))

        rate_limit_delta = rate_limit_after - rate_limit_before
        embed_bp_delta = embed_bp_after - embed_bp_before
        rerank_bp_delta = rerank_bp_after - rerank_bp_before
        stats_delta = rejected_stats_after - rejected_stats_before
        prometheus_delta = rate_limit_delta + embed_bp_delta + rerank_bp_delta

        print_info(f"Observed 429: {observed_429}, 503: {observed_503}")
        print_info(f"Prometheus rate_limit delta: {rate_limit_delta}")
        print_info(f"Prometheus embedding backpressure delta: {embed_bp_delta}")
        print_info(f"Prometheus rerank backpressure delta: {rerank_bp_delta}")
        print_info(f"/stats rejected_requests delta: {stats_delta}")

        if observed_429 == 0 and observed_503 == 0:
            print_error(
                "Burst produced 0 rejections - alignment claim untestable; "
                "raise burst size or check rate-limit config."
            )
            return False

        if stats_delta != prometheus_delta:
            print_error(
                f"/stats delta {stats_delta} != sum of Prometheus deltas "
                f"{prometheus_delta}"
            )
            return False

        print_success(
            "/stats and Prometheus rejection counters aligned "
            f"({int(stats_delta)} rejections)"
        )
        return True

    except Exception as e:
        print_error(f"Stats/Prometheus alignment test error: {e}")
        return False


async def test_colbert_vectors(client: httpx.AsyncClient, base_url: str) -> bool:
    """Test 15: ColBERT vectors, shape, dtype, multi-sentence, and consistency.

    Validates BGE-M3 ColBERT (token-level) embeddings beyond bare presence:
    - Single-sentence: returns List[List[float]] with positive token count and
      consistent inner dimension across tokens.
    - Multi-sentence: each item produces its own non-empty ColBERT tensor.
    - Negative: return_colbert=False omits the colbert field (or leaves None).
    - Consistency: identical input yields identical ColBERT vectors.
    """
    print_header("TEST 15: ColBERT Vectors")

    try:
        # --- 15a: single sentence, only ColBERT ---
        resp = await client.post(
            f"{base_url}/embeddings/",
            json={
                "sentences": [
                    "Token-level embeddings are useful for late interaction."
                ],
                "return_dense": False,
                "return_sparse": False,
                "return_colbert": True,
            },
            timeout=30.0,
        )
        if resp.status_code != 200:
            print_error(
                f"ColBERT-only request failed: {resp.status_code} - {resp.text}"
            )
            return False
        data = resp.json()
        emb = data["data"][0]["embeddings"]
        colbert = emb.get("colbert")

        if colbert is None:
            print_error("colbert field missing on ColBERT-only request")
            return False
        if not isinstance(colbert, list):
            print_error(f"colbert should be list, got {type(colbert)}")
            return False
        if len(colbert) == 0:
            print_error("colbert empty - expected >=1 token vector")
            return False
        if not all(isinstance(tok, list) for tok in colbert):
            print_error("colbert rows must each be a list")
            return False
        inner_dims = {len(tok) for tok in colbert}
        if len(inner_dims) != 1:
            print_error(f"colbert inner dims inconsistent across tokens: {inner_dims}")
            return False
        token_dim = inner_dims.pop()
        if token_dim <= 0:
            print_error(f"colbert token dim non-positive: {token_dim}")
            return False
        sample_values = colbert[0][:4]
        if not all(isinstance(v, (int, float)) for tok in colbert for v in tok):
            print_error("colbert contains non-numeric values")
            return False
        print_success("ColBERT single-sentence shape and dtype valid")
        print_info(f"Tokens: {len(colbert)}, dim/token: {token_dim}")
        print_info(f"Sample row[0][:4]: {[round(float(v), 4) for v in sample_values]}")

        # Negative side: dense and sparse must be absent on this response
        if emb.get("dense") is not None:
            print_error("dense field unexpectedly present when return_dense=False")
            return False
        if emb.get("sparse") is not None:
            print_error("sparse field unexpectedly present when return_sparse=False")
            return False
        print_success("dense/sparse correctly omitted when their flags are False")

        # --- 15b: multi-sentence ---
        resp_multi = await client.post(
            f"{base_url}/embeddings/",
            json={
                "sentences": [
                    "First short sentence.",
                    "A longer second sentence with more tokens to embed properly.",
                    "Terzo testo in italiano per variare la tokenizzazione.",
                ],
                "return_dense": False,
                "return_sparse": False,
                "return_colbert": True,
            },
            timeout=30.0,
        )
        if resp_multi.status_code != 200:
            print_error(f"ColBERT multi-sentence failed: {resp_multi.status_code}")
            return False
        multi = resp_multi.json()["data"]
        if len(multi) != 3:
            print_error(f"Expected 3 items, got {len(multi)}")
            return False
        per_item_tokens = []
        for item in multi:
            cb = item["embeddings"].get("colbert")
            if not isinstance(cb, list) or not cb:
                print_error(f"item {item['id']} colbert missing or empty")
                return False
            if {len(tok) for tok in cb} != {token_dim}:
                print_error(
                    f"item {item['id']} token dim differs from "
                    f"single-sentence ({token_dim})"
                )
                return False
            per_item_tokens.append(len(cb))
        print_success(
            "Multi-sentence ColBERT valid "
            f"(tokens per item: {per_item_tokens}, dim={token_dim})"
        )

        # --- 15c: negative - return_colbert=False should omit colbert ---
        resp_no_cb = await client.post(
            f"{base_url}/embeddings/",
            json={
                "sentences": ["ColBERT must be absent here."],
                "return_dense": True,
                "return_sparse": False,
                "return_colbert": False,
            },
            timeout=30.0,
        )
        if resp_no_cb.status_code != 200:
            print_error(f"Negative ColBERT request failed: {resp_no_cb.status_code}")
            return False
        neg_emb = resp_no_cb.json()["data"][0]["embeddings"]
        if neg_emb.get("colbert") is not None:
            print_error("colbert field present despite return_colbert=False")
            return False
        if neg_emb.get("dense") is None:
            print_error("dense missing despite return_dense=True")
            return False
        print_success("return_colbert=False correctly omits colbert field")

        # --- 15d: consistency across identical requests ---
        payload = {
            "sentences": ["Deterministic check for ColBERT vectors."],
            "return_dense": False,
            "return_sparse": False,
            "return_colbert": True,
        }
        first = await client.post(f"{base_url}/embeddings/", json=payload, timeout=30.0)
        second = await client.post(
            f"{base_url}/embeddings/", json=payload, timeout=30.0
        )
        if first.status_code != 200 or second.status_code != 200:
            print_error("Consistency requests did not both return 200")
            return False
        cb1 = first.json()["data"][0]["embeddings"]["colbert"]
        cb2 = second.json()["data"][0]["embeddings"]["colbert"]
        if len(cb1) != len(cb2) or len(cb1[0]) != len(cb2[0]):
            print_error(
                "Shape mismatch across identical calls: "
                f"({len(cb1)},{len(cb1[0])}) vs "
                f"({len(cb2)},{len(cb2[0])})"
            )
            return False
        max_abs_diff = 0.0
        for row1, row2 in zip(cb1, cb2):
            for a, b in zip(row1, row2):
                d = abs(float(a) - float(b))
                if d > max_abs_diff:
                    max_abs_diff = d
        # FP16 inference: tolerate small drift
        tolerance = 1e-3
        if max_abs_diff > tolerance:
            print_error(
                "ColBERT not deterministic enough: "
                f"max abs diff {max_abs_diff} > {tolerance}"
            )
            return False
        print_success(
            f"ColBERT deterministic within tolerance (max abs diff {max_abs_diff:.2e})"
        )

        return True

    except Exception as e:
        print_error(f"ColBERT test error: {e}")
        return False


async def test_all_three_vectors(client: httpx.AsyncClient, base_url: str) -> bool:
    """Test 16: Combined dense + sparse + colbert in a single multi-sentence call.

    Verifies that requesting all three embedding types simultaneously yields a
    response where each item exposes a well-formed dense vector, non-empty
    sparse lexical weights, and a non-empty ColBERT tensor with consistent
    dimensionality across items.
    """
    print_header("TEST 16: All Three Vectors Combined")

    try:
        sentences = [
            "Combined dense, sparse, and ColBERT in a single batch.",
            "Una seconda frase per validare la batch size.",
            "Third sentence with different token mix and punctuation!",
        ]
        resp = await client.post(
            f"{base_url}/embeddings/",
            json={
                "sentences": sentences,
                "return_dense": True,
                "return_sparse": True,
                "return_colbert": True,
                "normalize_dense": True,
            },
            timeout=30.0,
        )
        if resp.status_code != 200:
            print_error(f"Combined request failed: {resp.status_code} - {resp.text}")
            return False
        data = resp.json()
        items = data.get("data", [])
        if len(items) != len(sentences):
            print_error(f"Expected {len(sentences)} items, got {len(items)}")
            return False

        dense_dims = set()
        colbert_token_dims = set()
        for i, item in enumerate(items):
            emb = item["embeddings"]
            dense = emb.get("dense")
            sparse = emb.get("sparse")
            colbert = emb.get("colbert")

            if not isinstance(dense, list) or len(dense) == 0:
                print_error(f"item {i}: dense missing or empty")
                return False
            if not all(isinstance(v, (int, float)) for v in dense):
                print_error(f"item {i}: dense contains non-numeric")
                return False
            dense_dims.add(len(dense))

            if not isinstance(sparse, dict) or len(sparse) == 0:
                print_error(f"item {i}: sparse missing or empty")
                return False
            for k, v in sparse.items():
                if not isinstance(k, str) or not isinstance(v, (int, float)):
                    print_error(f"item {i}: sparse entry malformed: {k}={v}")
                    return False

            if not isinstance(colbert, list) or len(colbert) == 0:
                print_error(f"item {i}: colbert missing or empty")
                return False
            inner = {len(tok) for tok in colbert}
            if len(inner) != 1:
                print_error(f"item {i}: inconsistent colbert inner dims {inner}")
                return False
            colbert_token_dims.update(inner)

        if len(dense_dims) != 1:
            print_error(f"Dense dim inconsistent across items: {dense_dims}")
            return False
        if len(colbert_token_dims) != 1:
            print_error(
                f"ColBERT token dim inconsistent across items: {colbert_token_dims}"
            )
            return False

        dense_dim = dense_dims.pop()
        colbert_dim = colbert_token_dims.pop()

        # Verify normalize_dense=True produced unit-norm dense vectors.
        import math

        norms = [
            math.sqrt(sum(float(v) * float(v) for v in item["embeddings"]["dense"]))
            for item in items
        ]
        max_norm_err = max(abs(n - 1.0) for n in norms)
        if max_norm_err > 1e-3:
            print_error(
                "normalize_dense=True did not produce unit vectors "
                f"(max err {max_norm_err})"
            )
            return False

        print_success("All three vector types present and well-formed for every item")
        print_info(f"dense dim: {dense_dim}, colbert token dim: {colbert_dim}")
        print_info(f"normalize_dense unit-norm max err: {max_norm_err:.2e}")
        print_info(f"model_name: {data.get('model_name')}")
        print_info(f"processing_time_ms: {data.get('processing_time_ms')}")
        return True

    except Exception as e:
        print_error(f"All-three-vectors test error: {e}")
        return False


async def test_authentication(base_url: str) -> bool:
    """Test bearer auth behavior when the server is configured with API_TOKEN."""
    print_header("TEST 11: Authentication")

    try:
        async with httpx.AsyncClient() as anonymous_client:
            public_resp = await anonymous_client.get(f"{base_url}/health", timeout=5.0)
            if public_resp.status_code != 200:
                print_error(
                    f"Public /health should return 200, got {public_resp.status_code}"
                )
                return False
            print_success("Public /health works without token")

            protected_resp = await anonymous_client.post(
                f"{base_url}/embeddings/",
                json={
                    "sentences": ["auth check"],
                    "return_dense": False,
                    "return_sparse": False,
                    "return_colbert": False,
                },
                timeout=5.0,
            )
            if protected_resp.status_code != 401:
                print_error(
                    "Protected /embeddings/ without token should return 401, "
                    f"got {protected_resp.status_code}"
                )
                return False
            print_success("Protected /embeddings/ rejects missing token")

        return True

    except Exception as e:
        print_error(f"Authentication test error: {e}")
        return False


async def run_all_tests(
    base_url: str = "http://localhost:8000",
    api_token: str = "",
    payload_limits: dict = None,
    backpressure_requests: int = 200,
):
    """Run all tests"""
    print_header("BGE-M3 EMBEDDING SERVER - TEST SUITE")
    print_info(f"Target: {base_url}")
    print_info(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    results = {}
    payload_limits = payload_limits or DEFAULT_PAYLOAD_LIMITS
    headers = {"Authorization": f"Bearer {api_token}"} if api_token else {}

    async with httpx.AsyncClient(headers=headers) as client:
        # Run tests
        results["health"] = await test_health_endpoint(client, base_url)
        await asyncio.sleep(1)

        if api_token:
            results["auth"] = await test_authentication(base_url)
            await asyncio.sleep(1)

        results["embedding"] = await test_basic_embedding(client, base_url)
        await asyncio.sleep(1)

        results["sparse_as_indices"] = await test_sparse_as_indices(client, base_url)
        await asyncio.sleep(1)

        results["colbert_vectors"] = await test_colbert_vectors(client, base_url)
        await asyncio.sleep(1)

        results["all_three_vectors"] = await test_all_three_vectors(client, base_url)
        await asyncio.sleep(1)

        results["payload_limits"] = await test_payload_limits(
            client, base_url, payload_limits
        )
        await asyncio.sleep(1)

        results["performance"] = await test_performance(client, base_url)
        await asyncio.sleep(2)

        results["rerank"] = await test_rerank(client, base_url)
        await asyncio.sleep(2)

        results["rerank_backpressure"] = await test_rerank_backpressure(
            client, base_url
        )
        # Let token bucket refill before the rate-limit blast.
        await asyncio.sleep(4)

        results["rate_limiting"] = await test_rate_limiting(client, base_url)
        await asyncio.sleep(2)

        results["backpressure"] = await test_backpressure(
            client, base_url, backpressure_requests
        )
        await asyncio.sleep(4)

        results["queue_drain"] = await test_queue_gauge_drain(client, base_url)
        await asyncio.sleep(2)

        results["stats_alignment"] = await test_stats_prometheus_alignment(
            client, base_url
        )
        await asyncio.sleep(3)

        results[
            "rerank_rate_limit_attribution"
        ] = await test_rerank_rate_limit_attribution(client, base_url)
        await asyncio.sleep(3)

        results["metrics"] = await test_prometheus_metrics(client, base_url)
        await asyncio.sleep(1)

        results["stats"] = await test_stats_endpoint(client, base_url)

    # Summary
    print_header("TEST SUMMARY")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = (
            f"{Colors.GREEN}PASS{Colors.END}"
            if result
            else f"{Colors.RED}FAIL{Colors.END}"
        )
        print(f"{test_name.upper().ljust(20)}: {status}")

    print(f"\n{Colors.BOLD}Total: {passed}/{total} tests passed{Colors.END}")

    if passed == total:
        print(f"\n{Colors.GREEN}{Colors.BOLD}ALL TESTS PASSED!{Colors.END}")
        return True
    else:
        print(f"\n{Colors.YELLOW}{Colors.BOLD}SOME TESTS FAILED{Colors.END}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BGE-M3 server test suite")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL of the server (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--token",
        default="",
        help="Bearer token to use when API_TOKEN is configured on the server",
    )
    parser.add_argument(
        "--backpressure-requests",
        type=int,
        default=200,
        help="Concurrent heavy requests for the backpressure test",
    )
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=DEFAULT_PAYLOAD_LIMITS["max_sentences"],
        help="Expected MAX_SENTENCES_PER_REQUEST configured on the server",
    )
    parser.add_argument(
        "--max-sentence-chars",
        type=int,
        default=DEFAULT_PAYLOAD_LIMITS["max_sentence_chars"],
        help="Expected MAX_SENTENCE_CHARS configured on the server",
    )
    parser.add_argument(
        "--max-total-chars",
        type=int,
        default=DEFAULT_PAYLOAD_LIMITS["max_total_chars"],
        help="Expected MAX_TOTAL_CHARS_PER_REQUEST configured on the server",
    )
    parser.add_argument(
        "--max-rerank-passages",
        type=int,
        default=DEFAULT_PAYLOAD_LIMITS["max_rerank_passages"],
        help="Expected MAX_RERANK_PASSAGES configured on the server",
    )
    parser.add_argument(
        "--max-rerank-text-chars",
        type=int,
        default=DEFAULT_PAYLOAD_LIMITS["max_rerank_text_chars"],
        help="Expected MAX_RERANK_TEXT_CHARS configured on the server",
    )
    parser.add_argument(
        "--max-rerank-total-chars",
        type=int,
        default=DEFAULT_PAYLOAD_LIMITS["max_rerank_total_chars"],
        help="Expected MAX_RERANK_TOTAL_CHARS configured on the server",
    )
    args = parser.parse_args()
    payload_limits = {
        "max_sentences": args.max_sentences,
        "max_sentence_chars": args.max_sentence_chars,
        "max_total_chars": args.max_total_chars,
        "max_rerank_passages": args.max_rerank_passages,
        "max_rerank_text_chars": args.max_rerank_text_chars,
        "max_rerank_total_chars": args.max_rerank_total_chars,
    }
    try:
        success = asyncio.run(
            run_all_tests(
                args.url,
                api_token=args.token,
                payload_limits=payload_limits,
                backpressure_requests=args.backpressure_requests,
            )
        )
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Tests interrupted by user{Colors.END}")
        exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}Fatal error: {e}{Colors.END}")
        exit(1)
