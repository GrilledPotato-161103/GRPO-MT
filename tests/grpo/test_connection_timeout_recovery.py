"""Connection recovery must preserve the SAC label and identify sanitized failures."""
import asyncio

import httpx
import pytest

from src.feedback.backends import APIChatBackend, FeedbackError
from src.feedback.sac import SACScorer


def test_connection_timeout_recovers_with_separate_response_budget(monkeypatch, caplog):
    monkeypatch.setenv("OPENROUTER_API_KEY", "private-value")
    calls = 0

    def respond(request):
        nonlocal calls
        calls += 1
        assert request.extensions["timeout"]["connect"] == 30
        assert request.extensions["timeout"]["read"] == 120
        if calls == 1:
            raise httpx.ConnectTimeout("private-value", request=request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "Full match"}}]})

    scorer = SACScorer(evaluator=APIChatBackend(
        api_key_env="OPENROUTER_API_KEY", timeout=120, connect_timeout=30,
        max_retries=4, retry_backoff=0, transport=httpx.MockTransport(respond),
    ))
    assert scorer.evaluate(["source"], ["candidate"], ["English"], ["Vietnamese"]) == [1]
    assert calls == 2 and scorer.last_metrics["api_transport_retries"] == 1
    assert "ConnectTimeout" in caplog.text and "private-value" not in caplog.text


def test_exhausted_connection_timeout_is_typed_and_sanitized(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "private-value")
    calls = 0

    def respond(request):
        nonlocal calls
        calls += 1
        raise httpx.ConnectTimeout("private-value", request=request)

    backend = APIChatBackend(api_key_env="OPENROUTER_API_KEY", max_retries=2,
                             retry_backoff=0, transport=httpx.MockTransport(respond))
    with pytest.raises(FeedbackError, match="ConnectTimeout; item 0; attempts 3") as caught:
        asyncio.run(backend.generate([[{"role": "user", "content": "x"}]]))
    assert calls == 3 and "private-value" not in str(caught.value)


@pytest.mark.parametrize("timeout", [0, -1, True, float("nan"), "30"])
def test_invalid_connection_budgets_are_rejected(timeout):
    with pytest.raises(ValueError):
        APIChatBackend(connect_timeout=timeout)


def test_connection_budget_does_not_invalidate_semantic_cache():
    first = SACScorer(backend={"connect_timeout": 15, "timeout": 60})
    second = SACScorer(backend={"connect_timeout": 30, "timeout": 120})
    assert first.identity == second.identity
