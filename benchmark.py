"""
BGE-M3 server benchmark.

Measures latency (avg/p50/p95/p99) and throughput (req/s, sent/s) across
configurable scenarios. Pure stdlib + httpx (already in requirements).

Usage:
    python benchmark.py --url http://localhost:8000 --concurrency 8 --requests 200
    python benchmark.py --scenarios embed_dense,rerank --concurrency 4 --requests 100
    python benchmark.py --token <token>
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, List

import httpx

DEFAULT_SENTENCES = [
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks to solve complex problems.",
    "Natural language processing enables machines to understand text.",
    "Vector databases power semantic search at scale.",
    "Retrieval-augmented generation grounds LLMs with external context.",
    "Transformers revolutionized sequence modeling.",
    "Embeddings map text to high-dimensional vector spaces.",
    "Cross-encoders score query-document pairs jointly.",
]

DEFAULT_PASSAGES = [
    "Machine learning is a subset of artificial intelligence.",
    "The weather forecast predicts rain tomorrow.",
    "Deep learning uses neural networks for complex tasks.",
    "Cooking pasta requires boiling water and salt.",
    "Vector databases enable efficient similarity search.",
    "The Eiffel Tower is located in Paris, France.",
    "Cross-encoders rerank retrieved candidates by relevance.",
    "Photosynthesis converts sunlight into chemical energy.",
]


@dataclass
class Sample:
    latency_ms: float
    ok: bool
    status: int = 0
    error: str = ""


@dataclass
class ScenarioResult:
    name: str
    samples: List[Sample] = field(default_factory=list)
    wall_time_s: float = 0.0
    total_units: int = 0  # sentences embedded or pairs reranked

    @property
    def ok_latencies(self) -> List[float]:
        return [s.latency_ms for s in self.samples if s.ok]

    @property
    def success(self) -> int:
        return sum(1 for s in self.samples if s.ok)

    @property
    def failed(self) -> int:
        return len(self.samples) - self.success

    @property
    def ok_percent(self) -> float:
        return (self.success / len(self.samples) * 100.0) if self.samples else 0.0


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = max(
        0, min(len(sorted_vals) - 1, int(round(p / 100.0 * (len(sorted_vals) - 1))))
    )
    return sorted_vals[k]


def fmt_ms(v: float) -> str:
    return f"{v:.1f}" if v < 1000 else f"{v / 1000:.2f}s"


def fmt_int(v: int | float) -> str:
    return f"{v:,}".replace(",", " ")


def section(title: str) -> str:
    return f"{title}\n{'-' * len(title)}"


def sample_status_label(sample: Sample) -> str:
    if sample.status:
        return str(sample.status)
    return f"EXC:{sample.error or 'client_error'}"


def status_counts(result: ScenarioResult) -> Dict[str, int]:
    counts = Counter(sample_status_label(sample) for sample in result.samples)

    def sort_key(item: tuple[str, int]) -> tuple[int, str]:
        key = item[0]
        if key.isdigit():
            return (0, f"{int(key):04d}")
        return (1, key)

    return dict(sorted(counts.items(), key=sort_key))


def format_status_counts(result: ScenarioResult) -> str:
    counts = status_counts(result)
    return ", ".join(f"{status}={count}" for status, count in counts.items()) or "-"


def render_table(rows: List[List[str]], headers: List[str]) -> str:
    cols = list(zip(*([headers] + rows)))
    widths = [max(len(str(c)) for c in col) for col in cols]
    sep = "+".join("-" * (w + 2) for w in widths)
    sep = f"+{sep}+"

    def row(cells: List[str]) -> str:
        padded = [f" {c:<{w}} " for c, w in zip(cells, widths)]
        return f"|{'|'.join(padded)}|"

    out = [sep, row(headers), sep]
    for r in rows:
        out.append(row([str(c) for c in r]))
    out.append(sep)
    return "\n".join(out)


# --- Scenarios --------------------------------------------------------------

ScenarioFn = Callable[[httpx.AsyncClient, str], Awaitable[Sample]]


def make_embed_request(payload: Dict, timeout: float) -> ScenarioFn:
    async def fn(client: httpx.AsyncClient, base_url: str) -> Sample:
        start = time.perf_counter()
        try:
            resp = await client.post(
                f"{base_url}/embeddings/", json=payload, timeout=timeout
            )
            latency = (time.perf_counter() - start) * 1000
            return Sample(
                latency_ms=latency, ok=resp.status_code == 200, status=resp.status_code
            )
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            return Sample(
                latency_ms=latency,
                ok=False,
                status=0,
                error=type(e).__name__,
            )

    return fn


def make_rerank_request(payload: Dict, timeout: float) -> ScenarioFn:
    async def fn(client: httpx.AsyncClient, base_url: str) -> Sample:
        start = time.perf_counter()
        try:
            resp = await client.post(
                f"{base_url}/rerank", json=payload, timeout=timeout
            )
            latency = (time.perf_counter() - start) * 1000
            return Sample(
                latency_ms=latency, ok=resp.status_code == 200, status=resp.status_code
            )
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            return Sample(
                latency_ms=latency, ok=False, status=0, error=type(e).__name__
            )

    return fn


def build_scenarios(
    batch_size: int, timeout: float
) -> Dict[str, tuple[ScenarioFn, int]]:
    sents = (DEFAULT_SENTENCES * ((batch_size // len(DEFAULT_SENTENCES)) + 1))[
        :batch_size
    ]
    passages = (DEFAULT_PASSAGES * ((batch_size // len(DEFAULT_PASSAGES)) + 1))[
        :batch_size
    ]

    embed_dense = {
        "sentences": sents,
        "return_dense": True,
        "return_sparse": False,
        "return_colbert": False,
    }
    embed_full = {
        "sentences": sents,
        "return_dense": True,
        "return_sparse": True,
        "return_colbert": True,
    }
    rerank_payload = {
        "query": "What is machine learning?",
        "passages": passages,
        "normalize": True,
    }

    return {
        "embed_dense": (make_embed_request(embed_dense, timeout), batch_size),
        "embed_full": (make_embed_request(embed_full, timeout), batch_size),
        "rerank": (make_rerank_request(rerank_payload, timeout), batch_size),
    }


# --- Runner -----------------------------------------------------------------


async def run_scenario(
    client: httpx.AsyncClient,
    base_url: str,
    name: str,
    fn: ScenarioFn,
    units_per_req: int,
    requests: int,
    concurrency: int,
    warmup: int,
) -> ScenarioResult:
    # Warmup
    if warmup > 0:
        warm = [fn(client, base_url) for _ in range(warmup)]
        await asyncio.gather(*warm, return_exceptions=True)

    sem = asyncio.Semaphore(concurrency)

    async def bounded() -> Sample:
        async with sem:
            return await fn(client, base_url)

    start = time.perf_counter()
    samples = await asyncio.gather(*[bounded() for _ in range(requests)])
    wall = time.perf_counter() - start

    return ScenarioResult(
        name=name,
        samples=samples,
        wall_time_s=wall,
        total_units=units_per_req * sum(1 for s in samples if s.ok),
    )


def summarize(results: List[ScenarioResult], concurrency: int) -> str:
    headers = [
        "Scenario",
        "Result",
        "Reqs",
        "OK",
        "OK%",
        "Fail",
        "Conc",
        "Wall(s)",
        "Req/s",
        "Units/s",
        "Avg(ms)",
        "P50",
        "P95",
        "P99",
        "Min",
        "Max",
        "Statuses",
    ]
    rows: List[List[str]] = []
    for r in results:
        lat = r.ok_latencies
        rps = r.success / r.wall_time_s if r.wall_time_s > 0 else 0
        ups = r.total_units / r.wall_time_s if r.wall_time_s > 0 else 0
        if lat:
            avg = statistics.fmean(lat)
            p50 = percentile(lat, 50)
            p95 = percentile(lat, 95)
            p99 = percentile(lat, 99)
            mn = min(lat)
            mx = max(lat)
        else:
            avg = p50 = p95 = p99 = mn = mx = 0.0

        rows.append(
            [
                r.name,
                "PASS" if r.failed == 0 else "FAIL",
                fmt_int(len(r.samples)),
                fmt_int(r.success),
                f"{r.ok_percent:.1f}%",
                fmt_int(r.failed),
                str(concurrency),
                f"{r.wall_time_s:.2f}",
                f"{rps:.1f}",
                f"{ups:.0f}",
                fmt_ms(avg),
                fmt_ms(p50),
                fmt_ms(p95),
                fmt_ms(p99),
                fmt_ms(mn),
                fmt_ms(mx),
                format_status_counts(r),
            ]
        )
    return render_table(rows, headers)


def render_verdict(results: List[ScenarioResult]) -> str:
    total = sum(len(r.samples) for r in results)
    failed = sum(r.failed for r in results)
    if failed == 0:
        return (
            "Verdict: PASS - "
            f"{fmt_int(total)} measured requests completed successfully."
        )

    failed_names = ", ".join(r.name for r in results if r.failed)
    return (
        f"Verdict: FAIL - {fmt_int(failed)}/{fmt_int(total)} measured requests failed "
        f"across: {failed_names}."
    )


def render_server_info(info: Dict) -> str:
    gpu = info.get("gpu", {}) if isinstance(info.get("gpu"), dict) else {}
    rows = [
        ["Status", str(info.get("status", "unknown"))],
        ["Model", str(info.get("model", "unknown"))],
        ["GPU", str(gpu.get("device_info", "unknown"))],
        ["GPU available", str(gpu.get("available", "unknown"))],
        ["Max input length", str(info.get("max_input_length", "unknown"))],
        ["Batch size", str(info.get("batch_size", "unknown"))],
        ["Max requests/batch", str(info.get("max_requests_in_batch", "unknown"))],
    ]
    return render_table(rows, ["Field", "Value"])


def extract_metric_lines(metrics_text: str) -> List[str]:
    prefixes = (
        "embedding_requests_total",
        "embedding_requests_rejected_total",
        "embedding_queue_size",
        "embedding_active_requests",
        "rerank_requests_total",
        "rerank_requests_rejected_total",
        "rerank_active_requests",
    )
    return [
        line
        for line in metrics_text.splitlines()
        if line.startswith(prefixes) and not line.startswith("#")
    ]


def validate_args(args: argparse.Namespace) -> List[str]:
    errors = []

    if args.concurrency < 1:
        errors.append("--concurrency must be >= 1")
    if args.requests < 1:
        errors.append("--requests must be >= 1")
    if args.batch_size < 1:
        errors.append("--batch-size must be >= 1")
    if args.warmup < 0:
        errors.append("--warmup must be >= 0")
    if args.sleep_between < 0:
        errors.append("--sleep-between must be >= 0")
    if args.timeout <= 0:
        errors.append("--timeout must be > 0")
    if args.max_batch_size < 0:
        errors.append("--max-batch-size must be >= 0")
    if args.max_batch_size and args.batch_size > args.max_batch_size:
        errors.append(
            f"--batch-size exceeds --max-batch-size ({args.max_batch_size}); "
            "raise --max-batch-size or set it to 0 if the server allows larger payloads"
        )

    return errors


async def main_async(args: argparse.Namespace) -> int:
    arg_errors = validate_args(args)
    if arg_errors:
        print("Invalid arguments:")
        for error in arg_errors:
            print(f"  - {error}")
        return 2

    scenarios = build_scenarios(args.batch_size, args.timeout)
    requested = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    unknown = [s for s in requested if s not in scenarios]
    if not requested:
        print(f"No scenarios selected. Available: {list(scenarios)}")
        return 2
    if unknown:
        print(f"Unknown scenarios: {unknown}. Available: {list(scenarios)}")
        return 2

    print(section("Benchmark plan"))
    print(
        render_table(
            [
                ["Target", args.url],
                ["Scenarios", ", ".join(requested)],
                ["Concurrency", str(args.concurrency)],
                ["Requests/scenario", str(args.requests)],
                ["Batch size", f"{args.batch_size} sentences/passages per request"],
                ["Warmup", f"{args.warmup} requests per scenario"],
                ["Timeout", f"{args.timeout:.1f}s per request"],
                ["Auth", "Bearer token provided" if args.token else "disabled"],
            ],
            ["Setting", "Value"],
        )
    )
    print()

    limits = httpx.Limits(
        max_connections=args.concurrency * 2,
        max_keepalive_connections=args.concurrency * 2,
    )
    headers = {"Authorization": f"Bearer {args.token}"} if args.token else {}
    async with httpx.AsyncClient(headers=headers, limits=limits) as client:
        # Health check
        try:
            r = await client.get(f"{args.url}/health", timeout=10.0)
            if r.status_code != 200:
                print(f"Health check failed: {r.status_code}")
                return 1
            info = r.json()
            print(section("Pre-flight health"))
            print(render_server_info(info))
            print()
        except Exception as e:
            print(f"Cannot reach {args.url}: {e}")
            return 1

        results: List[ScenarioResult] = []
        for idx, name in enumerate(requested):
            if idx > 0 and args.sleep_between > 0:
                print(f"Cooling down {args.sleep_between}s (rate-limit refill)...")
                await asyncio.sleep(args.sleep_between)
            fn, units = scenarios[name]
            print(f"Running: {name} ...")
            res = await run_scenario(
                client=client,
                base_url=args.url,
                name=name,
                fn=fn,
                units_per_req=units,
                requests=args.requests,
                concurrency=args.concurrency,
                warmup=args.warmup,
            )
            results.append(res)
            print(
                f"  {name}: {'PASS' if res.failed == 0 else 'FAIL'} "
                f"({res.success}/{len(res.samples)} OK, "
                f"{res.wall_time_s:.2f}s, statuses: {format_status_counts(res)})"
            )

        post_health = None
        post_metrics = ""
        if not args.no_post_check:
            try:
                post_health_resp = await client.get(f"{args.url}/health", timeout=10.0)
                if post_health_resp.status_code == 200:
                    post_health = post_health_resp.json()
            except Exception as e:
                post_metrics += f"post-health-error={type(e).__name__}: {e}\n"
            try:
                metrics_resp = await client.get(f"{args.url}/metrics", timeout=10.0)
                if metrics_resp.status_code == 200:
                    post_metrics += metrics_resp.text
                else:
                    post_metrics += f"post-metrics-status={metrics_resp.status_code}\n"
            except Exception as e:
                post_metrics += f"post-metrics-error={type(e).__name__}: {e}\n"

    print()
    print(section("Scenario results"))
    print(summarize(results, args.concurrency))
    print()
    print(render_verdict(results))
    print()

    # Per-scenario error breakdown if any
    errors = []
    for r in results:
        if r.failed:
            status_counts: Dict[str, int] = {}
            for s in r.samples:
                if not s.ok:
                    label = sample_status_label(s)
                    status_counts[label] = status_counts.get(label, 0) + 1
            errors.append((r.name, status_counts))
    if errors:
        print(section("Failure details"))
        for name, sc in errors:
            detail = ", ".join(f"{status}={count}" for status, count in sc.items())
            print(f"  {name}: {detail}")
        print()

    if not args.no_post_check:
        print(section("Post-run health and key metrics"))
        if post_health:
            print(
                f"Health: {post_health.get('status', 'unknown')} "
                f"model={post_health.get('model', 'unknown')}"
            )
        else:
            print("Health: unavailable")

        metric_lines = extract_metric_lines(post_metrics)
        if metric_lines:
            for line in metric_lines:
                print(f"  {line}")
        else:
            print("  No key Prometheus metrics found or /metrics unavailable.")
        print()

    any_fail = any(r.failed > 0 for r in results)
    return 1 if any_fail else 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BGE-M3 server benchmark")
    p.add_argument("--url", default="http://localhost:8000", help="Server base URL")
    p.add_argument(
        "--token",
        default=os.getenv("API_TOKEN", ""),
        help="Bearer token to use when API_TOKEN is configured on the server",
    )
    p.add_argument(
        "--concurrency", type=int, default=8, help="Concurrent in-flight requests"
    )
    p.add_argument("--requests", type=int, default=100, help="Requests per scenario")
    p.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Sentences/passages per request (input batch)",
    )
    p.add_argument("--warmup", type=int, default=5, help="Warmup requests per scenario")
    p.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Per-request timeout in seconds",
    )
    p.add_argument(
        "--max-batch-size",
        type=int,
        default=128,
        help="Fail-fast guard for default payload item limits; set 0 to disable",
    )
    p.add_argument(
        "--scenarios",
        default="embed_dense,embed_full,rerank",
        help="Comma-separated scenarios: embed_dense, embed_full, rerank",
    )
    p.add_argument(
        "--sleep-between",
        type=int,
        default=0,
        help=(
            "Seconds to sleep between scenarios (use ~65s if rate-limit is 100 req/min)"
        ),
    )
    p.add_argument(
        "--no-post-check",
        action="store_true",
        help="Skip post-run /health and /metrics diagnostics",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        exit(asyncio.run(main_async(args)))
    except KeyboardInterrupt:
        print("\nInterrupted")
        exit(130)
