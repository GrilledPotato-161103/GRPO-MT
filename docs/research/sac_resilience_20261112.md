# SAC recovery and throughput verification ? 12 November 2026

The previous A100 run stopped at RL step 1,416 after an unparseable SAC response.
This change retries only the malformed conversation and preserves already validated results.

## Behavior

- `sac_reward.max_parse_retries=2`: at most two additional format requests per candidate.
- `sac_reward.backend.concurrency=8`: shared HTTP cap across overlapping calls on one event loop.
- Network and format retry budgets are additive; default maximum is five HTTP attempts per candidate.
- Network backoff releases the slot. Retry-After values are parsed as seconds or HTTP dates, capped by `max_retry_delay`.
- Invalid credentials and ordinary credit failures fail immediately. Explicit transient in-flight-budget 402 and provider errors embedded in HTTP-200 bodies are classified separately.
- Valid responses are atomically cached as they arrive. Malformed responses are never cached or replaced with an invented reward.
- Existing compatible v1 cache entries are read and migrated; scheduling changes no longer invalidate new semantic cache entries.
- Self-hosted judges retain batched inference and retry only failed rows.

## Live OpenRouter timing

Model: `openai/gpt-6-luna`. Eight distinct synthetic English?Vietnamese comparisons per setting; one repeat. All 24 successful responses parsed without format retries. The current Windows user key was selected without exposing it. Timing covers SAC requests, not model generation or backpropagation.

| Concurrency | Eight scores, seconds | Scores/second | Speedup versus serial |
|---|---:|---:|---:|
| 1 | 11.9408 | 0.67 | 1.00? |
| 4 | 4.0468 | 1.98 | 2.95? |
| 8 | 1.6001 | 5.00 | 7.46? |

This is a small live sample; provider latency and rate limits can change. The earlier production profile already used concurrency 8, so the serial speedup is not a claim of an equivalent improvement over that profile.

## Controlled fault-injection timing

64 candidates, 20 ms simulated request latency, one malformed initial reply per eight candidates, median of three repeats. Each repeat made 72 HTTP requests: 64 original requests plus exactly eight isolated retries.

| Concurrency | Median seconds | Scores/second | Speedup versus serial |
|---|---:|---:|---:|
| 1 | 1.8608 | 34.39 | 1.00? |
| 4 | 0.5584 | 114.61 | 3.33? |
| 8 | 0.2842 | 225.16 | 6.55? |
| 16 | 0.1688 | 379.21 | 11.02? |

## Error coverage and checks

- Full suite: 316 passing cases. Focused final feedback run: 102 passing cases. Together they cover 317 distinct passing cases.
- API/SAC coverage: 98.18% statements, 92.54% branches, 96.72% combined.
- The SAC module has 100% statement coverage; the focused run also exercises tiny-model LoRA optimizer updates with SAC rewards.
- Cases cover malformed/empty responses, schema errors, bounded exhaustion, transient HTTP/network errors, permanent authentication/credit errors, Retry-After, independent progress with slow peers, concurrency caps across simultaneous calls, cancellation, corrupt and legacy caches, and preservation of valid results after peer failures.
- Mypy passed for 54 source files; project Ruff and Black checks passed.
- Two CLI integration fixtures were corrected to use temporary SFT-export/cache paths instead of resolved production directories. Existing production artifacts were preserved.

## Reproduce

```bash
python -m scripts.benchmarks.sac --output logs/benchmarks/sac-offline.json
python -m scripts.benchmarks.sac --live --output logs/benchmarks/sac-live.json
python -m pytest tests/grpo/test_sac_resilience.py tests/grpo/test_feedback.py tests/grpo/test_sac_reward.py --cov=src.feedback.backends --cov=src.feedback.sac --cov-branch
```

Live mode performs paid synthetic comparisons and reads `ANTHROPIC_AUTH_TOKEN`. Offline mode sends no API traffic. Logs/JSON/XML/HTML evidence is in `logs/benchmarks/sac_20261112/`. No GPU rental or training restart was performed.

Reference behavior: [OpenRouter errors](https://openrouter.ai/docs/api_reference/errors-and-debugging) and [HTTPX resource limits](https://www.python-httpx.org/advanced/resource-limits/).
