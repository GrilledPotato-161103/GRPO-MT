# Dynamic COMET/SAC fusion with parallel bilingual endpoints

Use `configs/runs/a100.yaml` for a 40 GiB A100 or `configs/runs/rtx4090.yaml` for 24 GiB.
Both decode EN -> VI and VI -> EN together. The first 2,000 SFT updates,
subsequent GRPO, frozen-backbone LoRA, plain AdamW, and disabled KL remain.

For an original parallel pair `(e, v)` and candidate index `k`:

```text
v_hat[k] = policy(e, English -> Vietnamese)
e_hat[k] = policy(v, Vietnamese -> English)  # same generation batch
symmetric_sac[k] = (SAC(e, e_hat[k]) + SAC(v, v_hat[k])) / 2
R_Comet[k] = clip(COMET(source, prediction[k]), 0, 1)
R_SAC[k] = 0.5 * direct_SAC(source, prediction[k]) + 0.5 * symmetric_sac[k]
alpha_t = 0.1 + 0.8 * clip(completed_RL_updates / total_RL_updates, 0, 1)
r_fused[k] = sigmoid((1 - alpha_t) * R_Comet[k] + alpha_t * R_SAC[k])
```

Both streams are higher-is-better rewards. No negative mismatch penalty is
subtracted after fusion. The default training configuration standardizes each
stream separately within its source's candidate group before the final formula;
this balances variance without changing either stream's preference direction.
Set `sac_reward.fusion.normalize_groups=false` for the literal raw-score mix.
Raw scores remain bounded in [0,1]; group-standardized scores can be negative.
Empty outputs receive zero component rewards before fusion.

Greedy evaluation uses raw component scores because a one-candidate z-score
would be constant and contain no quality information. Metrics identify this
with `fusion_normalized=0`, and report the alpha used. Compare validation
COMET, SAC, BLEU and chrF directly: changing alpha can change fused reward
without a translation-quality improvement.

Alpha advances over RL optimizer updates, not SFT steps. SFT validation uses
the starting alpha. A rollout's scores and alpha stay fixed across its reused
GRPO updates; the next rollout uses the new progress value. Checkpoint loading
and standalone evaluation recover the clock from saved optimizer progress.
The horizon follows `max_optimizer_steps.rl` in production; the five-step
synthetic probe uses horizon 5 to exercise the schedule.

Each English/Vietnamese endpoint is compared with its matching original in
the same language. The two endpoint scores are averaged symmetrically, while
direct SAC remains directional. Candidate `k` is paired with the opposite
direction's candidate `k`; averaging the whole group into a constant would
erase the endpoint signal during GRPO advantage centering. The direct-versus-
symmetric SAC weight is configurable as `sac_reward.fusion.sac_direct_weight`.

There is no `policy(policy(source))` call. Both directions share one generation
batch, with up to two judge comparisons per generated candidate before caching.
This removes dependent reverse decoding. It requires aligned parallel training
pairs; held-out examples are not introduced into training or augmentation.

This implements your COMET/SAC variant of the dynamic fusion in
[M2PO](https://aclanthology.org/2026.acl-long.469.pdf). The paper's original
curriculum combines a static faithfulness score with model confidence; this
configuration combines COMET with external SAC rewards. SAC retains Full
match=1.0, High Partial match=0.7, Low Partial match=0.3, No match=0.1.
The paper's multi-pair CPO objective is not implemented here.

## Pairing and augmentation

`paired_batches: true` keeps both views together through sampling, length
bucketing, and collation. Batch sizes count directional examples and must be
even. Batch size 4 contains two parallel pairs; group size 4 produces 16
candidates. Pair IDs and candidate indices survive row sorting. Missing halves
or incorrect original/language alignment fail before generation.

Concatenation is drawn once per pair from the current training partition and
shared by both directions. Each input gets independent whitespace/Unicode
noise, with clean originals retained for scoring. Validation/test remain clean.
Length budgets account for both directional views and all candidates.

## Run and evaluate

For fresh Qwen3-1.7B LoRA training on a 24 GiB RTX 4090, use
`configs/runs/qwen3_1p7b_4090.yaml`. It disables Qwen3 thinking in the native
chat prompt, uses 12 data workers and 16 concurrent SAC requests, and caps
validation and test at **200 batches per evaluation**. A split containing fewer
batches uses its full available size. SFT uses batch size 24; RL and standalone
evaluation use 16, so the cap is not a fixed example count across phases.

```bash
python -m scripts.main.train --config configs/runs/qwen3_1p7b_4090.yaml --stage pipeline
```

From `GRPO-MT`, activate the existing environment (`conda activate ../venv`
locally, or `source /venv/main/bin/activate` on the prepared rental). Keep
`ANTHROPIC_AUTH_TOKEN`, `WANDB_API_KEY`, and `HUGGING_FACE_API_KEY` in environment
variables. The first variable is used as the OpenRouter key despite its name.
The HF token needs COMETKiwi model access. On Windows, refresh the shell after
changing user environment variables so an inherited key does not shadow them.

```bash
python -m scripts.main.train --config configs/runs/rtx4090.yaml --dry-run
python -m scripts.main.train --config configs/runs/rtx4090.yaml --stage pipeline
```

Production data must exist under `data/full_prepared` and `data/full` as
configured. The rental's synthetic smoke data are not substitutes. To reuse
existing rental caches, add `--set hf_home=/workspace/hf-cache` and
`--set comet_checkpoint=/workspace/cometkiwi/checkpoints/model.ckpt`.

SFT exports its final checkpoint at exactly 2,000 updates; RL then trains over
that tuned foundation for 10,000 updates by default. SFT does not recur during
RL. Evaluate every 1,000 updates and after each stage; use validation for
selection and the official test explicitly afterward. These scripts never
rent, stop or destroy instances and the profile has no budget watchdog.

The 4090 profile uses SFT/RL batch sizes 4, log-probability microbatch size 2,
a 768-token scoring graph budget, source cap 256, output cap 128, evaluation
batch size 4, and 16 persistent workers. No extra reverse-decoding cap is used.
A smaller batch changes data exposure in 2,000 updates; compare GPUs using
matched examples/tokens as well as optimizer-step times.

```bash
python -m scripts.main.eval --config configs/runs/rtx4090.yaml --stage rl \
  --checkpoint logs/sac-phomt-4090/rl/checkpoints/handoff.ckpt \
  --trust-checkpoint --split val --save-predictions \
  --output logs/sac-phomt-4090/manual-validation
```

For persistent progress use `tmux new-session -s mt-sac`, then run the training
command. Detach with Ctrl-b then d; reattach with `tmux attach -t mt-sac`.
W&B project remains `Machine Translation`. BLEU and chrF++ remain evaluation
metrics and have zero training-reward weight.

## Metrics and controls

Reports, prediction exports and W&B curves include `endpoint_sac`,
`symmetric_sac`, raw `comet_reward`/`sac_reward`, normalized fusion inputs,
`fusion_alpha`, direct SAC, COMET and fused reward, by direction.
Legacy penalty fields are only produced when fusion is disabled.
Conflict diagnostics compare the negative effect of penalties, while raw
penalty means stay positive. API/cache/token counters describe a rollout and
repeat on reused optimizer updates; do not sum repeated snapshots as new calls.
Symmetric mode must have no `reward/round_trip_generation` timing event.
With beta=0, KL metrics are zero by definition, not measured divergence.

| Comparison | Override |
|---|---|
| Direct SAC only | `--set sac_reward.fusion.sac_direct_weight=1` |
| Raw fusion scores | `--set sac_reward.fusion.normalize_groups=false` |
| Legacy subtraction objective | `--set sac_reward.fusion.enabled=false` |
| Transfer projection/L2-SP | `--set optimizer=transfer` |
| Fixed-reference KL | `--set beta=0.01` |
| Cache-only judge | `--set sac_reward.cache_only=true` (misses fail) |
| Sequential cycle comparison | `--set sac_reward.consistency_mode=round_trip` |

Use a fresh output directory per ablation. Legacy round-trip batch and token
options remain supported but are unused in symmetric mode. For self-hosting,
replace the entire `sac_reward.backend` mapping with `{kind: self_hosted,
model_name_or_path: /path/to/model, device: cuda:1, dtype: bfloat16, revision: main,
max_new_tokens: 32, max_input_tokens: 2048, batch_size: 4}` and remove API-only
fields. API defaults remain OpenRouter `openai/gpt-6-luna`, strict labels,
bounded concurrency, caching, and explicit failure rather than fabricated reward.

### SAC response recovery and concurrency

The scorer validates each OpenRouter response as it arrives. A malformed label,
empty response or invalid completion envelope triggers a new request for that
candidate only, with a stronger format instruction. Correct responses are cached
immediately and remain reusable even if another candidate exhausts its retries.
The strict four-label interpretation and reward values are unchanged.

```yaml
sac_reward:
  max_parse_retries: 2       # two additional format attempts per candidate
  backend:
    concurrency: 8          # simultaneous HTTP requests, shared by overlapping calls
    max_retries: 2          # additional network/408/429/5xx attempts per candidate
    retry_backoff: 1.0      # exponential network backoff, seconds
    max_retry_delay: 30.0   # cap for backoff and Retry-After delays
```

For the manual launcher, override with `--set sac_reward.backend.concurrency=16`
and `--set sac_reward.max_parse_retries=3`. Backoff releases the request slot so
healthy candidates can proceed. HTTP and format retry budgets are additive:
the defaults allow at most five HTTP attempts per candidate. Permanent HTTP
errors such as invalid credentials and insufficient credit fail immediately.
OpenRouter's explicit in-flight-budget 402 with `Retry-After` is retried;
ordinary insufficient-credit 402 responses are not. Provider errors embedded in
HTTP-200 bodies use the same status-based policy.
If format retries are exhausted, scoring raises an explicit error; it does not
assign a synthetic passing or failing reward. GRPO still requires the complete
candidate group's valid rewards before applying its optimizer update.

`reward/sac_parse_failures`, `reward/sac_parse_retries`,
`reward/sac_parse_exhausted`, `reward/sac_api_requests`,
`reward/sac_api_transport_retries`, `reward/sac_api_peak_concurrency` and
`reward/sac_api_elapsed_seconds` expose recovery and throughput in training logs.
Request-duration and backoff counters are also available. Cache keys exclude
concurrency and retry settings; compatible old cache entries migrate on read.

```bash
# Offline: 64 candidates, simulated 20 ms HTTP latency, one malformed response per 8 candidates.
python -m scripts.benchmarks.sac --output logs/benchmarks/sac-offline.json
# Live: 24 synthetic comparisons before retries; reads ANTHROPIC_AUTH_TOKEN.
python -m scripts.benchmarks.sac --live --output logs/benchmarks/sac-live.json
python -m pytest tests/grpo/test_sac_resilience.py -q
```

The live benchmark does not send training or held-out data. Offline timing
isolates scheduling overhead; it is not an estimate of provider latency.
Retry handling follows [OpenRouter's error guidance](https://openrouter.ai/docs/api_reference/errors-and-debugging),
with explicitly bounded [HTTPX connection pools](https://www.python-httpx.org/advanced/resource-limits/).
Measured timings and error coverage are recorded in
[the SAC recovery report](research/sac_resilience_20261112.md).

## Verification status

Local tests cover symmetry, paired augmentation with workers and length
buckets, nonconstant GRPO advantages, tiny-model optimizer updates, checkpoint
resume, one generation call per rollout, and bilingual evaluation/export.
The November 2026 A100 run completed 2,000 SFT updates and 1,416 GRPO updates
before malformed SAC feedback stopped training. Its held-out test covered all
38,302 directional examples with local COMETKiwi/BLEU/chrF++ scoring. Checkpoints
and predictions are saved under `../runs/a100-60228333-20261112/`; that rental was
destroyed at the user's request. The recovery changes above were subsequently
tested offline and against live OpenRouter, without restarting GPU training.
