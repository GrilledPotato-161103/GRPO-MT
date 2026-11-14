"""Feedback transport, schema, retry, and local inference tests without paid calls."""

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import torch
from transformers import AutoTokenizer

from scripts.verify.assets import create_assets
from src.feedback.backends import APIChatBackend, FeedbackError, TransformersChatBackend
from src.feedback.judge import TranslationJudge, TranslationTask, parse_feedback


def envelope(text: str) -> dict[str, Any]:
    return {
        "choices": [{"message": {"content": text}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 3},
    }


def test_api_defaults_and_alignment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAUDE_AUTH_KEY", "  test-only-key  ")

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://openrouter.ai/api/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer test-only-key"
        payload = json.loads(request.content)
        assert payload["model"] == "deepseek/deepseek-v4.1-flash-1017"
        assert payload["response_format"] == {"type": "json_object"}
        task = json.loads(payload["messages"][-1]["content"])
        return httpx.Response(
            200,
            json=envelope(
                json.dumps({"valid": True, "score": 0.9, "feedback": task["source"]})
            ),
        )

    backend = APIChatBackend(transport=httpx.MockTransport(handler))
    judge = TranslationJudge(backend)
    tasks = [
        TranslationTask("Hello", "English", "Vietnamese"),
        TranslationTask("Good morning", "English", "Vietnamese"),
    ]
    result = judge.evaluate(tasks, ["Xin chào", "Chào buổi sáng"])
    assert [r.feedback for r in result] == ["Hello", "Good morning"]
    assert (
        backend.requests == 2
        and backend.prompt_tokens == 20
        and backend.completion_tokens == 6
    )
    assert "test-only-key" not in repr(vars(backend))
    assert (
        APIChatBackend(base_url="http://localhost:8000/v1").resolved_model
        == "deepseek_v4.1_flash_1017"
    )
    assert asyncio.run(backend.generate([])) == []


@pytest.mark.parametrize("status", [429, 503])
def test_transient_api_retry(monkeypatch: pytest.MonkeyPatch, status: int) -> None:
    monkeypatch.setenv("CLAUDE_AUTH_KEY", "test-key")
    calls = []

    async def no_wait(delay: float) -> None:
        pass

    monkeypatch.setattr(asyncio, "sleep", no_wait)

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return (
            httpx.Response(status)
            if len(calls) == 1
            else httpx.Response(200, json=envelope("done"))
        )

    backend = APIChatBackend(transport=httpx.MockTransport(handler), max_retries=1)
    assert asyncio.run(backend.generate([[{"role": "user", "content": "x"}]])) == [
        "done"
    ]
    assert len(calls) == 2


@pytest.mark.parametrize("mode", ["auth", "server", "network", "envelope"])
def test_feedback_failures_are_explicit_and_do_not_leak_key(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    monkeypatch.setenv("CLAUDE_AUTH_KEY", "sensitive-test-value")

    def handler(request: httpx.Request) -> httpx.Response:
        if mode == "network":
            raise httpx.ConnectError("sensitive-test-value", request=request)
        if mode == "envelope":
            return httpx.Response(200, json={})
        return httpx.Response(
            401 if mode == "auth" else 503, text="sensitive-test-value"
        )

    backend = APIChatBackend(transport=httpx.MockTransport(handler), max_retries=0)
    with pytest.raises(FeedbackError) as exc:
        asyncio.run(backend.generate([[{"role": "user", "content": "x"}]]))
    assert "sensitive-test-value" not in str(exc.value)


def test_missing_credentials_and_invalid_backend_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CLAUDE_AUTH_KEY", raising=False)
    with pytest.raises(FeedbackError, match="CLAUDE_AUTH_KEY"):
        asyncio.run(APIChatBackend().generate([[{"role": "user", "content": "x"}]]))
    for kwargs in (
        {"concurrency": 0},
        {"base_url": "ftp://host"},
        {"base_url": "https://key:secret@host"},
    ):
        with pytest.raises(ValueError):
            APIChatBackend(**kwargs)


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        "[]",
        '{"valid":1,"score":0.5}',
        '{"valid":true,"score":true}',
        '{"valid":true,"score":2}',
        '{"valid":true,"score":NaN}',
        '{"valid":true,"score":0.5,"feedback":3}',
    ],
)
def test_malformed_judge_output_fails_closed(text: str) -> None:
    with pytest.raises(FeedbackError):
        parse_feedback(text)


def test_json_fence_and_notebook_event_loop() -> None:
    assert parse_feedback('```json\n{"valid":true,"score":0.5}\n```').score == 0.5

    class Backend:
        async def generate(self, conversations: list[Any]) -> list[str]:
            return ['{"valid":true,"score":1}'] * len(conversations)

    judge = TranslationJudge(Backend())
    task = TranslationTask("x", "English", "Vietnamese")

    async def notebook() -> None:
        assert judge.evaluate([task])[0].valid

    asyncio.run(notebook())
    with pytest.raises(ValueError):
        judge.evaluate([task], [])


def test_wrong_response_count() -> None:
    class Backend:
        async def generate(self, conversations: list[Any]) -> list[str]:
            return []

    with pytest.raises(FeedbackError):
        TranslationJudge(Backend()).evaluate([TranslationTask("x", "en", "vi")])


def test_self_hosted_transformers_backend(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    asset = create_assets(tmp_path / "judge", "causal")
    tokenizer = AutoTokenizer.from_pretrained(asset)
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['content'] + ' ' }}{% endfor %}"
    )
    tokenizer.save_pretrained(asset)
    backend = TransformersChatBackend(str(asset), max_new_tokens=2, max_input_tokens=8)
    conversations = [[{"role": "user", "content": "Hello"}]]
    assert len(asyncio.run(backend.generate(conversations))) == 1
    assert all(not p.requires_grad for p in backend.model.parameters())
    assert not backend.model.training
    assert asyncio.run(backend.generate([])) == []
    with pytest.raises(FeedbackError, match="truncate"):
        asyncio.run(backend.generate([[{"role": "user", "content": "Hello " * 20}]]))
    backend.tokenizer.chat_template = None
    with pytest.raises(FeedbackError, match="chat template"):
        asyncio.run(backend.generate(conversations))
    with pytest.raises(ValueError):
        TransformersChatBackend(str(asset), dtype="invalid")
