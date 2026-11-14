"""API and local Transformers backends with a shared batched chat interface."""

import asyncio
import logging
import math
import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from email.utils import parsedate_to_datetime
from typing import Any, Protocol, cast

import httpx

Messages = list[dict[str, str]]
DEFAULT_MODEL = "deepseek_v4.1_flash_1017"
OPENROUTER_MODEL = "deepseek/deepseek-v4.1-flash-1017"
log = logging.getLogger(__name__)


def format_retry_messages(messages: Messages, instruction: str) -> Messages:
    """Strengthen the system instruction without echoing rejected output or mutating inputs."""
    result = [dict(message) for message in messages]
    if result and result[0]["role"] == "system":
        result[0]["content"] += "\n" + instruction
    else:
        result.insert(0, {"role": "system", "content": instruction})
    return result


class FeedbackError(RuntimeError):
    """Feedback was unavailable or malformed; no training reward is fabricated."""


class ChatBackend(Protocol):
    """Generate one response for every chat conversation, preserving order."""

    async def generate(self, conversations: list[Messages]) -> list[str]:
        """Return chat completion text."""
        ...


class APIChatBackend:
    """OpenRouter by default; secrets are read at request time and never stored.

    Only the requested snapshot's documented OpenRouter identifier is mapped.
    No fallback to a different model or provider endpoint is attempted.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = "https://openrouter.ai/api/v1",
        api_key_env: str = "CLAUDE_AUTH_KEY",
        timeout: float = 60.0,
        concurrency: int = 4,
        max_retries: int = 2,
        max_tokens: int = 256,
        temperature: float | None = 0.0,
        json_mode: bool = True,
        transport: httpx.AsyncBaseTransport | None = None,
        retry_backoff: float = 1.0,
        max_retry_delay: float = 30.0,
    ) -> None:
        if (
            any(
                isinstance(value, bool) or not isinstance(value, int) or value < minimum
                for value, minimum in (
                    (concurrency, 1),
                    (max_retries, 0),
                    (max_tokens, 1),
                )
            )
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
                for value in (timeout, retry_backoff, max_retry_delay)
            )
            or timeout == 0
        ):
            raise ValueError(
                "Invalid feedback concurrency, retries, timeout, or token limit"
            )
        url = httpx.URL(base_url)
        if url.scheme not in {"http", "https"} or url.userinfo or url.query:
            raise ValueError(
                "Feedback base URL must be HTTP(S) without credentials or query"
            )
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.timeout = timeout
        self.concurrency = concurrency
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.json_mode = json_mode
        self.transport = transport
        self.retry_backoff = retry_backoff
        self.max_retry_delay = max_retry_delay
        self.requests = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self._limiters: dict[
            asyncio.AbstractEventLoop, tuple[asyncio.Semaphore, int]
        ] = {}

    @property
    def resolved_model(self) -> str:
        """Translate the configured alias only for the known OpenRouter host."""
        if (
            httpx.URL(self.base_url).host == "openrouter.ai"
            and self.model == DEFAULT_MODEL
        ):
            return OPENROUTER_MODEL
        return self.model

    def check_ready(self) -> None:
        """Check credential presence before allocating time to a training run."""
        if not os.environ.get(self.api_key_env, "").strip():
            raise FeedbackError(f"Set {self.api_key_env} for the feedback API")

    @contextmanager
    def _request_slots(self) -> Iterator[asyncio.Semaphore]:
        """Share the cap across overlapping calls; discard closed-loop state afterwards."""
        loop = asyncio.get_running_loop()
        semaphore, users = self._limiters.get(
            loop, (asyncio.Semaphore(self.concurrency), 0)
        )
        self._limiters[loop] = (semaphore, users + 1)
        try:
            yield semaphore
        finally:
            semaphore, users = self._limiters[loop]
            if users == 1:
                del self._limiters[loop]
            else:
                self._limiters[loop] = (semaphore, users - 1)

    def _retry_delay(self, retries: int, response: httpx.Response | None) -> float:
        """Respect numeric/date Retry-After headers within the configured wait cap."""
        fallback = min(
            self.retry_backoff * 2.0 ** min(retries, 16), self.max_retry_delay
        )
        if response is None or "Retry-After" not in response.headers:
            return fallback
        value = response.headers["Retry-After"]
        try:
            delay = float(value)
        except ValueError:
            try:
                delay = parsedate_to_datetime(value).timestamp() - time.time()
            except (ValueError, TypeError, OverflowError):
                return fallback
        return (
            min(max(delay, 0.0), self.max_retry_delay)
            if math.isfinite(delay)
            else fallback
        )

    @staticmethod
    def _response_status(response: httpx.Response) -> int:
        """OpenRouter may emit a provider error inside an HTTP-200 body."""
        if response.status_code != 200:
            return response.status_code
        try:
            error = response.json().get("error", {})
        except (ValueError, AttributeError):
            return 200
        code = error.get("code") if isinstance(error, dict) else None
        return code if type(code) is int and 400 <= code <= 599 else 200

    @staticmethod
    def _in_flight_budget(response: httpx.Response) -> bool:
        """Only the explicitly transient 402 budget case is retryable."""
        if "Retry-After" not in response.headers:
            return False
        try:
            return bool(
                response.json()["error"]["metadata"]["limit_source"]
                == "openrouter_in_flight_budget"
            )
        except (ValueError, KeyError, TypeError):
            return False

    async def generate(
        self,
        conversations: list[Messages],
        *,
        validator: Callable[[str], str] | None = None,
        validation_retries: int = 0,
        retry_instruction: str = "Return only the requested output, without explanation.",
        on_result: Callable[[int, str], None] | None = None,
        metrics: dict[str, float] | None = None,
    ) -> list[str]:
        """Retry and validate each conversation independently, preserving input order.

        HTTP failures and malformed completions have separate, additive budgets.
        Backoff never occupies a request slot. Valid results are delivered to
        ``on_result`` immediately, even if another conversation later fails.
        """
        if (
            isinstance(validation_retries, bool)
            or not isinstance(validation_retries, int)
            or validation_retries < 0
        ):
            raise ValueError("validation_retries must be a nonnegative integer")
        stats = metrics if metrics is not None else {}
        stats.update(
            {
                name: 0.0
                for name in (
                    "requests",
                    "prompt_tokens",
                    "completion_tokens",
                    "transport_retries",
                    "validation_failures",
                    "validation_retries",
                    "validation_exhausted",
                    "peak_concurrency",
                    "elapsed_seconds",
                    "request_seconds",
                    "max_request_seconds",
                    "backoff_seconds",
                )
            }
        )
        if not conversations:
            return []
        self.check_ready()
        key = os.environ[self.api_key_env].strip()
        began = time.perf_counter()
        try:
            with self._request_slots() as semaphore:
                return await self._generate(
                    conversations,
                    key,
                    semaphore,
                    validator,
                    validation_retries,
                    retry_instruction,
                    on_result,
                    stats,
                )
        finally:
            stats["elapsed_seconds"] = time.perf_counter() - began

    async def _generate(
        self,
        conversations: list[Messages],
        key: str,
        semaphore: asyncio.Semaphore,
        validator: Callable[[str], str] | None,
        validation_retries: int,
        retry_instruction: str,
        on_result: Callable[[int, str], None] | None,
        stats: dict[str, float],
    ) -> list[str]:
        active = 0
        async with httpx.AsyncClient(
            timeout=self.timeout,
            transport=self.transport,
            headers={"Authorization": f"Bearer {key}"},
            follow_redirects=False,
            limits=httpx.Limits(
                max_connections=self.concurrency,
                max_keepalive_connections=self.concurrency,
            ),
        ) as client:

            async def complete(index: int, messages: Messages) -> str:
                nonlocal active
                transport_retries = parse_retries = 0
                current_messages = messages
                while True:
                    payload: dict[str, Any] = {
                        "model": self.resolved_model,
                        "messages": current_messages,
                        "max_tokens": self.max_tokens,
                        "stream": False,
                    }
                    if self.temperature is not None:
                        payload["temperature"] = self.temperature
                    if self.json_mode:
                        payload["response_format"] = {"type": "json_object"}
                    # This snapshot supports disabling hidden reasoning for concise JSON feedback.
                    if httpx.URL(self.base_url).host == "openrouter.ai":
                        payload["reasoning"] = {"effort": "none"}
                    response = None
                    try:
                        async with semaphore:
                            self.requests += 1
                            stats["requests"] += 1
                            active += 1
                            stats["peak_concurrency"] = max(
                                stats["peak_concurrency"], active
                            )
                            request_began = time.perf_counter()
                            try:
                                response = await client.post(
                                    f"{self.base_url}/chat/completions", json=payload
                                )
                            finally:
                                active -= 1
                                duration = time.perf_counter() - request_began
                                stats["request_seconds"] += duration
                                stats["max_request_seconds"] = max(
                                    stats["max_request_seconds"], duration
                                )
                    except httpx.RequestError:
                        if transport_retries >= self.max_retries:
                            raise FeedbackError(
                                "Feedback network request failed after bounded retries"
                            ) from None
                    else:
                        status = self._response_status(response)
                        if status == 200:
                            try:
                                data = response.json()
                                usage = data.get("usage") or {}
                                if isinstance(usage, dict):
                                    for name in ("prompt_tokens", "completion_tokens"):
                                        count = usage.get(name, 0)
                                        if (
                                            isinstance(count, int)
                                            and not isinstance(count, bool)
                                            and count >= 0
                                        ):
                                            stats[name] += count
                                            setattr(
                                                self, name, getattr(self, name) + count
                                            )
                                content = data["choices"][0]["message"]["content"]
                                if not isinstance(content, str) or not content.strip():
                                    raise ValueError("empty completion")
                                value = validator(content) if validator else content
                                if not isinstance(value, str) or not value.strip():
                                    raise ValueError("invalid validated completion")
                            except (
                                FeedbackError,
                                ValueError,
                                KeyError,
                                IndexError,
                                TypeError,
                                AttributeError,
                            ):
                                stats["validation_failures"] += 1
                                if parse_retries >= validation_retries:
                                    stats["validation_exhausted"] += 1
                                    raise FeedbackError(
                                        f"Feedback completion validation failed for item {index} after {parse_retries + 1} attempts"
                                    ) from None
                                parse_retries += 1
                                stats["validation_retries"] += 1
                                log.warning(
                                    "Retrying malformed feedback item %d (%d/%d)",
                                    index,
                                    parse_retries,
                                    validation_retries,
                                )
                                current_messages = format_retry_messages(
                                    messages, retry_instruction
                                )
                                continue
                            if on_result is not None:
                                on_result(index, value)
                            return value
                        retryable = (
                            status in {408, 429}
                            or status >= 500
                            or (status == 402 and self._in_flight_budget(response))
                        )
                        if not retryable:
                            raise FeedbackError(
                                f"Feedback API HTTP {status}; check endpoint, credentials, and model ID {self.resolved_model}"
                            )
                        if transport_retries >= self.max_retries:
                            raise FeedbackError(
                                f"Feedback API HTTP {status} after bounded retries"
                            )
                    delay = self._retry_delay(transport_retries, response)
                    transport_retries += 1
                    stats["transport_retries"] += 1
                    backoff_began = time.perf_counter()
                    try:
                        await asyncio.sleep(delay)
                    finally:
                        stats["backoff_seconds"] += time.perf_counter() - backoff_began

            tasks = [
                asyncio.create_task(complete(i, messages))
                for i, messages in enumerate(conversations)
            ]
            try:
                return list(await asyncio.gather(*tasks))
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)


class TransformersChatBackend:
    """Self-host a frozen causal feedback model, loaded lazily on an explicit device."""

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cpu",
        dtype: str = "float32",
        revision: str = "main",
        max_new_tokens: int = 256,
        max_input_tokens: int = 2048,
        batch_size: int = 1,
    ) -> None:
        if (
            dtype not in {"float32", "float16", "bfloat16"}
            or min(max_new_tokens, max_input_tokens, batch_size) < 1
        ):
            raise ValueError("Invalid feedback dtype, token limit, or batch size")
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.dtype = dtype
        self.revision = revision
        self.max_new_tokens = max_new_tokens
        self.max_input_tokens = max_input_tokens
        self.batch_size = batch_size
        self.model: Any = None
        self.tokenizer: Any = None

    def _generate_sync(self, conversations: list[Messages]) -> list[str]:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path, revision=self.revision
            )  # type: ignore[no-untyped-call]
            if self.tokenizer.pad_token_id is None:
                if self.tokenizer.eos_token_id is None:
                    raise FeedbackError("Local feedback tokenizer requires PAD or EOS")
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"
            self.model = (
                cast(Any, AutoModelForCausalLM)
                .from_pretrained(
                    self.model_name_or_path,
                    revision=self.revision,
                    torch_dtype=getattr(torch, self.dtype),
                )
                .to(self.device)
                .requires_grad_(False)
                .eval()
            )
        output: list[str] = []
        for offset in range(0, len(conversations), self.batch_size):
            batch = conversations[offset : offset + self.batch_size]
            if not self.tokenizer.chat_template:
                raise FeedbackError(
                    "Local feedback tokenizer needs an instruction chat template"
                )
            prompts = [
                self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                for messages in batch
            ]
            encoded = self.tokenizer(
                prompts, padding=True, return_tensors="pt", return_token_type_ids=False
            ).to(self.device)
            if encoded["input_ids"].shape[1] > self.max_input_tokens:
                raise FeedbackError(
                    "Feedback input exceeds max_input_tokens; refusing to truncate the candidate"
                )
            with torch.inference_mode():
                sequences = self.model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            output.extend(
                self.tokenizer.batch_decode(
                    sequences[:, encoded["input_ids"].shape[1] :],
                    skip_special_tokens=True,
                )
            )
        return output

    async def generate(self, conversations: list[Messages]) -> list[str]:
        """Run local inference outside the caller's event loop."""
        if not conversations:
            return []
        return await asyncio.to_thread(self._generate_sync, conversations)
