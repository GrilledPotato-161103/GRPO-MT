"""Measure SAC throughput and retry recovery; synthetic transport by default."""

import argparse
import asyncio
import json
import logging
import math
import os
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

import httpx

from src.feedback.backends import APIChatBackend, FeedbackError
from src.feedback.sac import SACScorer
from src.utils.credentials import environment_secret


async def measure(
    concurrency: int,
    requests: int,
    latency_ms: float,
    malformed_every: int,
    live: bool,
    model: str,
    api_key_env: str,
) -> dict[str, float]:
    """Score distinct synthetic translations once without a disk cache."""
    attempts: Counter[str] = Counter()

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(json.loads(request.content)["messages"][-1]["content"])
        source = payload["source"]
        index = int(source.split()[3])
        attempts[source] += 1
        await asyncio.sleep(latency_ms / 1000)
        malformed = (
            malformed_every > 0
            and index % malformed_every == 0
            and attempts[source] == 1
        )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "Full match." if malformed else "Full match"
                        }
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 2},
            },
        )

    backend = APIChatBackend(
        model=model,
        api_key_env=api_key_env,
        concurrency=concurrency,
        temperature=None,
        json_mode=False,
        max_tokens=32,
        transport=None if live else httpx.MockTransport(handler),
    )
    scorer = SACScorer(evaluator=backend)
    began = time.perf_counter()
    scores = await scorer.evaluate_async(
        [f"The package number {i} is ready." for i in range(requests)],
        [f"Kiện hàng số {i} đã sẵn sàng." for i in range(requests)],
        ["English"] * requests,
        ["Vietnamese"] * requests,
    )
    elapsed = time.perf_counter() - began
    assert len(scores) == requests and all(
        score in {0.1, 0.3, 0.7, 1.0} for score in scores
    )
    return {
        "seconds": elapsed,
        "scores_per_second": requests / elapsed,
        **scorer.last_metrics,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concurrency", type=int, nargs="+")
    parser.add_argument("--requests", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--latency-ms", type=float, default=20)
    parser.add_argument(
        "--malformed-every",
        type=int,
        default=8,
        help="Inject one malformed first response per N mock requests; 0 disables faults",
    )
    parser.add_argument(
        "--live", action="store_true", help="Send paid synthetic requests to OpenRouter"
    )
    parser.add_argument("--model", default="openai/gpt-6-luna")
    parser.add_argument("--api-key-env", default="ANTHROPIC_AUTH_TOKEN")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.concurrency is None:
        args.concurrency = [1, 4, 8] if args.live else [1, 4, 8, 16]
    if args.requests is None:
        args.requests = 8 if args.live else 64
    if args.repeats is None:
        args.repeats = 1 if args.live else 3
    if (
        min(*args.concurrency, args.requests, args.repeats) < 1
        or args.latency_ms < 0
        or not math.isfinite(args.latency_ms)
        or args.malformed_every < 0
    ):
        parser.error(
            "Concurrency, requests and repeats must be positive; latency/fault interval cannot be negative"
        )
    key_name = args.api_key_env if args.live else "SAC_BENCHMARK_MOCK_KEY"
    key = environment_secret(key_name) if args.live else "synthetic-only-not-a-secret"
    if not key:
        parser.error(f"Set {key_name} before a live benchmark")
    previous = os.environ.get(key_name)
    os.environ[key_name] = key.strip()
    del key
    logger = logging.getLogger("src.feedback.backends")
    previous_level = logger.level
    logger.setLevel(logging.ERROR)
    rows = []
    try:
        for concurrency in args.concurrency:
            samples = [
                asyncio.run(
                    measure(
                        concurrency,
                        args.requests,
                        args.latency_ms,
                        args.malformed_every,
                        args.live,
                        args.model,
                        key_name,
                    )
                )
                for _ in range(args.repeats)
            ]
            seconds = statistics.median(s["seconds"] for s in samples)
            rows.append(
                {
                    "concurrency": concurrency,
                    "median_seconds": round(seconds, 4),
                    "scores_per_second": round(args.requests / seconds, 2),
                    "peak_concurrency": max(s["api_peak_concurrency"] for s in samples),
                    "api_requests_per_repeat": statistics.mean(
                        s["api_requests"] for s in samples
                    ),
                    "parse_retries_per_repeat": statistics.mean(
                        s["parse_retries"] for s in samples
                    ),
                    "mean_request_seconds": statistics.mean(
                        s["api_request_seconds"] / s["api_requests"] for s in samples
                    ),
                }
            )
        serial = next(
            (r["median_seconds"] for r in rows if r["concurrency"] == 1), None
        )
        for row in rows:
            row["speedup_vs_serial"] = (
                round(serial / row["median_seconds"], 2) if serial else None
            )
        report: dict[str, Any] = {
            "mode": "live_openrouter" if args.live else "simulated_http_transport",
            "model": args.model,
            "requests_per_repeat": args.requests,
            "repeats": args.repeats,
            "injected_malformed_every": None if args.live else args.malformed_every,
            "simulated_latency_ms": None if args.live else args.latency_ms,
            "results": rows,
        }
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
        print(json.dumps(report, indent=2))
        return 0
    except FeedbackError as error:
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(
                    {
                        "mode": "live_openrouter"
                        if args.live
                        else "simulated_http_transport",
                        "status": "failed",
                        "error": str(error),
                        "results": rows,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        print(f"SAC benchmark failed: {error}")
        return 1
    finally:
        logger.setLevel(previous_level)
        if previous is None:
            os.environ.pop(key_name, None)
        else:
            os.environ[key_name] = previous


if __name__ == "__main__":
    raise SystemExit(main())
