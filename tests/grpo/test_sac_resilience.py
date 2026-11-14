"""SAC parse retries, independent concurrent requests, and valid-result preservation."""

import asyncio
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from email.utils import format_datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from src.feedback.backends import APIChatBackend, FeedbackError
from src.feedback.sac import SACScorer


def envelope(text: Any) -> dict[str, Any]:
    return {
        "choices": [{"message": {"content": text}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2},
    }


def api(handler: Any, **options: Any) -> APIChatBackend:
    return APIChatBackend(
        model="openai/gpt-6-luna",
        temperature=None,
        json_mode=False,
        transport=httpx.MockTransport(handler),
        **options,
    )


def score(scorer: SACScorer, candidates: list[str]) -> list[float]:
    return scorer.evaluate(
        ["source"] * len(candidates),
        candidates,
        ["English"] * len(candidates),
        ["Vietnamese"] * len(candidates),
    )


def test_only_malformed_response_is_requested_again(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLAUDE_AUTH_KEY", "sac-test-secret")
    calls: Counter[str] = Counter()

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        candidate = json.loads(body["messages"][-1]["content"])["candidate"]
        calls[candidate] += 1
        return httpx.Response(
            200,
            json=envelope(
                "Full match."
                if candidate == "retry" and calls[candidate] == 1
                else "Full match"
            ),
        )

    scorer = SACScorer(evaluator=api(handler, max_retries=0), cache_dir=str(tmp_path))
    assert score(scorer, ["good", "retry", "good"]) == [1.0, 1.0, 1.0]
    assert calls == {"good": 1, "retry": 2}
    assert scorer.last_metrics["parse_retries"] == 1
    assert scorer.last_metrics["api_prompt_tokens"] == 30


def test_valid_results_are_cached_when_another_response_exhausts_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLAUDE_AUTH_KEY", "sac-test-secret")

    def handler(request: httpx.Request) -> httpx.Response:
        candidate = json.loads(json.loads(request.content)["messages"][-1]["content"])[
            "candidate"
        ]
        return httpx.Response(
            200, json=envelope("Full match" if candidate == "good" else "invalid")
        )

    scorer = SACScorer(evaluator=api(handler, max_retries=0), cache_dir=str(tmp_path))
    with pytest.raises(FeedbackError):
        score(scorer, ["good", "bad"])
    assert len(list(tmp_path.glob("*.json"))) == 1
    assert score(scorer, ["good"]) == [1.0]
    assert scorer.last_metrics["api_requests"] == 0


def test_parse_retry_does_not_wait_for_a_slower_peer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLAUDE_AUTH_KEY", "sac-test-secret")

    async def scenario() -> None:
        retry_arrived = asyncio.Event()
        calls: Counter[str] = Counter()

        async def handler(request: httpx.Request) -> httpx.Response:
            candidate = json.loads(
                json.loads(request.content)["messages"][-1]["content"]
            )["candidate"]
            calls[candidate] += 1
            if candidate == "slow":
                await retry_arrived.wait()
            elif calls[candidate] == 1:
                return httpx.Response(200, json=envelope("not a label"))
            else:
                retry_arrived.set()
            return httpx.Response(200, json=envelope("Full match"))

        scorer = SACScorer(evaluator=api(handler, concurrency=2))
        result = await asyncio.wait_for(
            scorer.evaluate_async(
                ["s", "s"], ["slow", "retry"], ["en"] * 2, ["vi"] * 2
            ),
            timeout=1,
        )
        assert result == [1, 1] and calls == {"slow": 1, "retry": 2}

    asyncio.run(scenario())


def test_network_backoff_releases_the_only_request_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLAUDE_AUTH_KEY", "sac-test-secret")

    async def scenario() -> None:
        healthy_completed = asyncio.Event()
        order = []
        calls: Counter[str] = Counter()

        async def delay(_: float) -> None:
            await healthy_completed.wait()

        monkeypatch.setattr(asyncio, "sleep", delay)

        def handler(request: httpx.Request) -> httpx.Response:
            candidate = json.loads(
                json.loads(request.content)["messages"][-1]["content"]
            )["candidate"]
            calls[candidate] += 1
            order.append((candidate, calls[candidate]))
            if candidate == "retry" and calls[candidate] == 1:
                return httpx.Response(429)
            if candidate == "healthy":
                healthy_completed.set()
            return httpx.Response(200, json=envelope("Full match"))

        scorer = SACScorer(evaluator=api(handler, concurrency=1))
        await asyncio.wait_for(
            scorer.evaluate_async(
                ["s"] * 2, ["retry", "healthy"], ["en"] * 2, ["vi"] * 2
            ),
            timeout=1,
        )
        assert order == [("retry", 1), ("healthy", 1), ("retry", 2)]
        assert scorer.last_metrics["api_transport_retries"] == 1

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "bad",
    [
        None,
        "",
        3,
        {},
        "Full match.",
        "Full match because correct",
        '{"label":"Full match"}',
    ],
)
def test_invalid_content_gets_a_fresh_request(
    bad: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CLAUDE_AUTH_KEY", "sac-test-secret")
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return httpx.Response(
            200, json=envelope(bad if len(calls) == 1 else "Full match")
        )

    scorer = SACScorer(evaluator=api(handler))
    assert score(scorer, ["x"]) == [1]
    assert len(calls) == 2
    assert calls[0]["messages"][-1] == calls[1]["messages"][-1]
    assert "invalid format" in calls[1]["messages"][0]["content"]
    assert scorer.last_metrics["parse_failures"] == 1


@pytest.mark.parametrize("body", [[], {}, {"choices": []}, {"choices": [{}]}])
def test_invalid_envelope_is_retried(
    body: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CLAUDE_AUTH_KEY", "sac-test-secret")
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=body if calls == 1 else envelope("Full match"))

    assert score(SACScorer(evaluator=api(handler)), ["x"]) == [1]
    assert calls == 2


@pytest.mark.parametrize("retries", [0, 1, 3])
def test_parse_exhaustion_is_bounded_and_never_creates_a_reward(
    retries: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLAUDE_AUTH_KEY", "sac-test-secret")
    backend = api(lambda _: httpx.Response(200, json=envelope("sac-test-secret")))
    scorer = SACScorer(
        evaluator=backend, max_parse_retries=retries, cache_dir=str(tmp_path)
    )
    with pytest.raises(FeedbackError, match="validation failed") as error:
        score(scorer, ["x"])
    assert backend.requests == retries + 1
    assert scorer.last_metrics["parse_retries"] == retries
    assert scorer.last_metrics["parse_exhausted"] == 1
    assert "sac-test-secret" not in str(error.value)
    assert not list(tmp_path.glob("*.json")) and not backend._limiters


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404])
def test_permanent_http_errors_are_not_format_retried(
    status: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CLAUDE_AUTH_KEY", "sac-test-secret")
    backend = api(lambda _: httpx.Response(status, text="sac-test-secret"))
    scorer = SACScorer(evaluator=backend, max_parse_retries=5)
    with pytest.raises(FeedbackError, match=f"HTTP {status}") as error:
        score(scorer, ["x"])
    assert backend.requests == 1 and "sac-test-secret" not in str(error.value)


@pytest.mark.parametrize("status", [408, 429, 500, 503])
def test_transient_http_errors_retry_independently(
    status: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CLAUDE_AUTH_KEY", "sac-test-secret")
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return (
            httpx.Response(status)
            if calls == 1
            else httpx.Response(200, json=envelope("Full match"))
        )

    scorer = SACScorer(evaluator=api(handler, retry_backoff=0))
    assert score(scorer, ["x"]) == [1]
    assert calls == 2 and scorer.last_metrics["api_transport_retries"] == 1


def test_overlapping_batches_share_concurrency_and_keep_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLAUDE_AUTH_KEY", "sac-test-secret")
    active = peak = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(0.01)
            value = json.loads(request.content)["messages"][-1]["content"]
            return httpx.Response(200, json=envelope(value))
        finally:
            active -= 1

    backend = api(handler, concurrency=2)

    async def scenario() -> None:
        conversations = [[{"role": "user", "content": str(i)}] for i in range(8)]
        a, b = await asyncio.gather(
            backend.generate(conversations), backend.generate(conversations)
        )
        assert a == b == [str(i) for i in range(8)]

    asyncio.run(scenario())
    assert peak == 2 and active == 0 and not backend._limiters


def test_cancelled_batch_cleans_up_requests_and_loop_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLAUDE_AUTH_KEY", "sac-test-secret")

    async def scenario() -> None:
        started = asyncio.Event()
        forever = asyncio.Event()
        active = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal active
            active += 1
            started.set()
            try:
                await forever.wait()
                return httpx.Response(200, json=envelope("Full match"))
            finally:
                active -= 1

        backend = api(handler, concurrency=2)
        task = asyncio.create_task(
            backend.generate([[{"role": "user", "content": "x"}]] * 5)
        )
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert active == 0 and not backend._limiters

    asyncio.run(scenario())


def test_self_hosted_fallback_retries_only_failed_rows(tmp_path: Path) -> None:
    class Backend:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        async def generate(self, conversations: list[Any]) -> list[str]:
            names = [json.loads(m[-1]["content"])["candidate"] for m in conversations]
            self.calls.append(names)
            return [
                "bad format"
                if name == "retry" and len(self.calls) == 1
                else "Full match"
                for name in names
            ]

    backend = Backend()
    scorer = SACScorer(evaluator=backend, cache_dir=str(tmp_path))
    assert score(scorer, ["good", "retry"]) == [1, 1]
    assert backend.calls == [["good", "retry"], ["retry"]]
    assert scorer.last_metrics["parse_retries"] == 1


def test_corrupt_cache_is_refreshed_online_but_rejected_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CLAUDE_AUTH_KEY", "sac-test-secret")
    backend = api(lambda _: httpx.Response(200, json=envelope("Full match")))
    scorer = SACScorer(evaluator=backend, cache_dir=str(tmp_path))
    assert score(scorer, ["x"]) == [1]
    path = next(tmp_path.glob("*.json"))
    path.write_text('{"label":null}')
    with pytest.raises(FeedbackError, match="cache"):
        score(
            SACScorer(evaluator=backend, cache_dir=str(tmp_path), cache_only=True),
            ["x"],
        )
    assert score(scorer, ["x"]) == [1]
    assert backend.requests == 2 and scorer.last_metrics["cache_errors"] == 1


def test_concurrency_changes_reuse_semantic_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CLAUDE_AUTH_KEY", "sac-test-secret")
    backend = api(lambda _: httpx.Response(200, json=envelope("Full match")))
    assert score(
        SACScorer(
            backend={"concurrency": 1}, evaluator=backend, cache_dir=str(tmp_path)
        ),
        ["x"],
    ) == [1]
    assert score(
        SACScorer(
            backend={"concurrency": 16},
            evaluator=backend,
            cache_dir=str(tmp_path),
            cache_only=True,
        ),
        ["x"],
    ) == [1]
    assert backend.requests == 1


def test_previous_cache_format_is_read_without_api_call(tmp_path: Path) -> None:
    scorer = SACScorer(evaluator=object(), cache_dir=str(tmp_path))
    identity = dict(scorer.identity, version=1)
    payload = {
        "source": "source",
        "candidate": "x",
        "source_lang": "English",
        "target_lang": "Vietnamese",
    }
    key = hashlib.sha256(
        json.dumps([identity, payload], sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    (tmp_path / f"{key}.json").write_text('{"label":"Full match"}')
    assert score(scorer, ["x"]) == [1]
    assert scorer.last_metrics["cache_hits"] == 1


@pytest.mark.parametrize(
    "header,expected",
    [("3", 3), ("9999", 30), ("-1", 0), ("bad", 1), ("NaN", 1), ("date", 5)],
)
def test_retry_after_backoff(
    header: str, expected: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CLAUDE_AUTH_KEY", "sac-test-secret")
    monkeypatch.setattr("src.feedback.backends.time.time", lambda: 1000.0)
    waits = []

    async def sleep(delay: float) -> None:
        waits.append(delay)

    monkeypatch.setattr(asyncio, "sleep", sleep)
    if header == "date":
        header = format_datetime(datetime.fromtimestamp(1005, UTC), usegmt=True)
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return (
            httpx.Response(429, headers={"Retry-After": header})
            if calls == 1
            else httpx.Response(200, json=envelope("Full match"))
        )

    assert score(SACScorer(evaluator=api(handler)), ["x"]) == [1]
    assert waits == [expected]


@pytest.mark.parametrize("value", [-1, True, 1.2, None])
def test_invalid_parse_budget_is_rejected(value: Any) -> None:
    with pytest.raises(ValueError, match="max_parse_retries"):
        SACScorer(max_parse_retries=value)


@pytest.mark.parametrize("http_status", [200, 402])
def test_only_explicit_in_flight_spending_budget_is_retried(
    http_status: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CLAUDE_AUTH_KEY", "sac-test-secret")
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                http_status,
                headers={"Retry-After": "0"},
                json={
                    "error": {
                        "code": 402,
                        "metadata": {"limit_source": "openrouter_in_flight_budget"},
                    }
                },
            )
        return httpx.Response(200, json=envelope("Full match"))

    scorer = SACScorer(evaluator=api(handler))
    assert score(scorer, ["x"]) == [1]
    assert calls == 2 and scorer.last_metrics["api_transport_retries"] == 1


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"error": {"metadata": {"limit_source": "openrouter_key_limit"}}},
        {"error": None},
    ],
)
def test_credit_402_is_not_retried_even_with_retry_header(
    payload: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CLAUDE_AUTH_KEY", "sac-test-secret")
    backend = api(
        lambda _: httpx.Response(402, headers={"Retry-After": "0"}, json=payload)
    )
    with pytest.raises(FeedbackError, match="402"):
        score(SACScorer(evaluator=backend), ["x"])
    assert backend.requests == 1


@pytest.mark.parametrize("code", [401, 402, 429, 503])
def test_http_200_provider_errors_use_their_embedded_code(
    code: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CLAUDE_AUTH_KEY", "sac-test-secret")
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={"error": {"code": code}} if calls == 1 else envelope("Full match"),
        )

    scorer = SACScorer(evaluator=api(handler, retry_backoff=0))
    if code in {401, 402}:
        with pytest.raises(FeedbackError, match=str(code)):
            score(scorer, ["x"])
        assert calls == 1
    else:
        assert score(scorer, ["x"]) == [1]
        assert calls == 2 and scorer.last_metrics["parse_retries"] == 0


def test_retry_budgets_are_additive_not_multiplicative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLAUDE_AUTH_KEY", "sac-test-secret")
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls in {1, 3}:
            raise httpx.ReadTimeout("sac-test-secret", request=request)
        return httpx.Response(200, json=envelope("invalid"))

    backend = api(handler, retry_backoff=0)
    scorer = SACScorer(evaluator=backend)
    with pytest.raises(FeedbackError):
        score(scorer, ["x"])
    assert calls == 5
    assert scorer.last_metrics["parse_retries"] == 2
    assert scorer.last_metrics["api_transport_retries"] == 2


def test_cache_write_failure_does_not_spend_more_api_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLAUDE_AUTH_KEY", "sac-test-secret")
    backend = api(lambda _: httpx.Response(200, json=envelope("Full match")))
    scorer = SACScorer(evaluator=backend)

    def fail(*args: Any) -> None:
        raise OSError("cache unavailable")

    monkeypatch.setattr(scorer, "_cache_label", fail)
    with pytest.raises(OSError, match="cache"):
        score(scorer, ["x"])
    assert backend.requests == 1 and scorer.last_metrics["parse_retries"] == 0


def test_synchronous_bridge_works_inside_a_notebook_loop() -> None:
    class Backend:
        async def generate(self, conversations: list[Any]) -> list[str]:
            return ["Full match"] * len(conversations)

    async def notebook() -> None:
        assert score(SACScorer(evaluator=Backend()), ["x"]) == [1]

    asyncio.run(notebook())


def test_mock_benchmark_checks_recovery_and_restores_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.benchmarks.sac import main

    monkeypatch.delenv("SAC_BENCHMARK_MOCK_KEY", raising=False)
    output = tmp_path / "speed.json"
    assert (
        main(
            [
                "--requests",
                "4",
                "--repeats",
                "1",
                "--concurrency",
                "1",
                "2",
                "--latency-ms",
                "1",
                "--malformed-every",
                "4",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads(output.read_text())
    assert all(
        row["api_requests_per_repeat"] == 5 and row["parse_retries_per_repeat"] == 1
        for row in report["results"]
    )
    import os

    assert "SAC_BENCHMARK_MOCK_KEY" not in os.environ


def test_generic_api_validation_adds_format_instruction_without_mutating_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.feedback.sac import parse_sac

    monkeypatch.setenv("CLAUDE_AUTH_KEY", "sac-test-secret")
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content)["messages"])
        return httpx.Response(
            200, json=envelope("bad" if len(calls) == 1 else "Full match")
        )

    conversations = [[{"role": "user", "content": "original texts"}]]
    result = asyncio.run(
        api(handler).generate(conversations, validator=parse_sac, validation_retries=1)
    )
    assert result == ["Full match"]
    assert conversations == [[{"role": "user", "content": "original texts"}]]
    assert calls[1][0]["role"] == "system" and calls[1][-1] == conversations[0][0]


@pytest.mark.parametrize(
    "options",
    [
        {"concurrency": True},
        {"concurrency": 1.5},
        {"max_retries": -1},
        {"max_tokens": 0},
        {"timeout": 0},
        {"timeout": float("nan")},
        {"retry_backoff": -1},
        {"max_retry_delay": float("inf")},
    ],
)
def test_invalid_request_controls_fail_before_any_request(
    options: dict[str, Any]
) -> None:
    with pytest.raises(ValueError):
        APIChatBackend(**options)


def test_sac_cache_only_and_self_hosted_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CLAUDE_AUTH_KEY", raising=False)
    with pytest.raises(ValueError, match="cache_dir"):
        SACScorer(cache_only=True)
    with pytest.raises(ValueError, match="kind"):
        SACScorer(backend={"kind": "invalid"})
    scorer = SACScorer(
        backend={"kind": "self_hosted", "model_name_or_path": "unused-local-path"},
        cache_only=True,
        cache_dir=str(tmp_path),
    )
    scorer.check_ready()
    with pytest.raises(FeedbackError, match="cache miss"):
        score(scorer, ["x"])
    with pytest.raises(FeedbackError, match="CLAUDE_AUTH_KEY"):
        SACScorer().check_ready()
    with pytest.raises(ValueError, match="aligned"):
        SACScorer().evaluate(["x"], [], [], [])


def test_self_hosted_bad_response_count_is_not_assigned_to_a_candidate() -> None:
    class Backend:
        async def generate(self, conversations: list[Any]) -> list[str]:
            return []

    with pytest.raises(FeedbackError, match="count"):
        score(SACScorer(evaluator=Backend()), ["x"])


def test_production_v1_cache_survives_new_retry_options(tmp_path: Path) -> None:
    from src.feedback.sac import SAC_PROMPT

    options = {
        "model": "openai/gpt-6-luna",
        "concurrency": 8,
        "timeout": 60,
        "max_retries": 2,
        "temperature": None,
        "json_mode": False,
        "max_tokens": 32,
    }
    identity = {
        "version": 1,
        "rubric": SAC_PROMPT,
        "kind": "api",
        **options,
        "endpoint": "https://openrouter.ai/api/v1",
        "revision": None,
    }
    payload = {
        "source": "source",
        "candidate": "x",
        "source_lang": "English",
        "target_lang": "Vietnamese",
    }
    key = hashlib.sha256(
        json.dumps([identity, payload], sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    (tmp_path / f"{key}.json").write_text('{"label":"Full match"}')
    scorer = SACScorer(
        backend={**options, "retry_backoff": 1.0, "max_retry_delay": 30.0},
        cache_dir=str(tmp_path),
    )
    assert score(scorer, ["x"]) == [1]
    assert scorer.last_metrics["api_requests"] == 0
