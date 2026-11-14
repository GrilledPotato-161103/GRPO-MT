# LLM self-play GRPO for machine translation

This optional/reference guide retains earlier experiments and measured results.
Use [the execution manual](execution.md) and `configs/runs/a100.yaml` for the
current COMET/SAC pipeline. Archived shell launchers describe historical runs.

This optional/reference guide retains earlier experiments and measured results.
Use [the execution manual](execution.md) and `configs/runs/a100.yaml` for the
current COMET/SAC pipeline. Archived shell launchers describe historical runs.

## Relationship to the paper

This is an MT adaptation of [Absolute Zero, arXiv:2505.03335v3](https://arxiv.org/abs/2505.03335). It shares a proposer/solver policy, validates tasks, rewards learnable tasks, and maintains a task buffer. The paper uses executable code verification and task/role-relative REINFORCE++; this implementation uses a fixed translation judge and per-context GRPO. It does not reproduce the paper's verifier guarantees.

The proposer reward follows equation 4: `0` when every sampled solution fails, otherwise `1 - success_rate`. Translation success is a valid judge score at or above `solver_success_threshold`. Solver rewards default to continuous quality; `model.solver_reward_mode=binary` uses thresholded success. MT formatting penalties are configurable adaptations rather than a reproduction of the paper's task formats. No translation-quality improvement is established by the software tests.

## Training loop

1. Sample P source proposals per topic with the trainable proposer. Require JSON `{"source":"..."}`; reject malformed, oversized and duplicate proposals.
2. Have the fixed judge validate well-formed sources.
3. Use the **same trainable policy** as solver to sample G translations per valid source. If no new source is valid, replay one task from memory when available.
4. Judge adequacy, fluency, target language, omissions/additions, entities and numbers. Require JSON `{"valid":true,"score":0.9,"feedback":"brief errors"}`. Invalid candidates get zero quality; malformed feedback or transport failures abort the step rather than fabricate rewards.
5. Collect **both roles' rollouts before any parameter update**, then optimize weighted GRPO losses with fixed old-policy scores and the original frozen reference. Normalize each proposer context and solver source separately.
6. Keep valid tasks in a bounded, deduplicated checkpointed memory. The feedback model stays fixed and outside the optimizer/checkpoint weights. Credentials never enter Hydra config.

With topic batch B, P proposals and G translations, an all-valid batch generates `B*P` proposal and `B*P*G` solver completions and uses `B*P*(1+G)` feedback requests. Defaults B=1, P=2, G=4 mean ten judge requests per rollout batch. Two optimizer iterations reuse those same scores.

## Option 1: OpenRouter (default)

Use the existing environment from GRPO-MT:

```powershell
conda activate E:/IAI/Lab/Research/venv
python -m pip install -e .
```

Set `CLAUDE_AUTH_KEY` securely in the environment or ignored project `.env`. Its value must be an OpenRouter credential despite the user-selected variable name. Do not put the key on a command line or in YAML.

The default endpoint is `https://openrouter.ai/api/v1`. The configured alias `deepseek_v4.1_flash_1017` maps, for OpenRouter only, to `deepseek/deepseek-v4.1-flash-1017`, verified in the [model catalog](https://openrouter.ai/api/v1/models) on 7 November 2026. Other model strings/endpoints pass through literally, with no alternate-model fallback. See [authentication](https://openrouter.ai/docs/api_reference/authentication) and [API reference](https://openrouter.ai/docs/api_reference/overview).

Start with a five-batch pilot:

```powershell
python -m src.train --config-name self_play data.steps_per_epoch=5 model.group_size=2 model.proposal_group_size=2 model.max_source_length=512 model.max_new_tokens=64 model.proposal_max_new_tokens=64 +trainer.precision=bf16-mixed +trainer.limit_val_batches=2 +trainer.limit_test_batches=2
```

The default trainable policy is `Qwen/Qwen2.5-0.5B-Instruct`. Override it with a translation-capable instruction checkpoint for meaningful research. A 0.5B policy is the intended small-model 24GB pilot, not a measured memory guarantee. The full reference and optimizer need additional memory; begin with short sequences and B=1. Increasing P and G multiplies solver work and judge costs.

For a 7B policy, use the [LoRA / QLoRA presets and capacity guide](lora_qwen7b.md). Adapter mode uses a shared frozen base for reference scoring. The proposer output cap follows `model.max_new_tokens` by default and can be overridden separately.

API overrides:

```text
feedback.backend.model=provider/model-id
feedback.backend.base_url=https://openrouter.ai/api/v1
feedback.backend.api_key_env=CLAUDE_AUTH_KEY
feedback.backend.concurrency=4
feedback.backend.max_retries=2
feedback.backend.max_tokens=256
```

Async requests use bounded concurrency, finite timeouts, no redirects, and retries for network failures, HTTP 429 and server errors. Hidden reasoning is disabled for OpenRouter feedback to keep JSON concise. Training sends generated sources and translations to the API. Held-out references stay in local corpus evaluation and are not sent to the judge by default.

## Option 2: self-hosted feedback

```powershell
python -m src.train --config-name self_play feedback=self_hosted feedback.backend.model_name_or_path=path/to/local-instruction-judge feedback.backend.device=cpu data.steps_per_epoch=5
```

This option loads a frozen causal Transformers model and uses its tokenizer's chat template. Supply the model path explicitly; no enormous DeepSeek checkpoint is automatically downloaded. Choose a model with translation-evaluation and structured-JSON capability. A tiny random judge can test inference mechanics but cannot provide useful translation rewards.

The default judge device is CPU. A separate GPU can use `feedback.backend.device=cuda:1 feedback.backend.dtype=bfloat16`; sharing `cuda:0` needs extra memory. Local inference is lazy, deterministic, in eval mode, and gradient-free. Inputs over `max_input_tokens` fail instead of silently truncating the candidate. A chat template is required.

## Validation, resume and export

The default validation/test JSONL files are synthetic smoke data. For research, provide established disjoint parallel splits:

```powershell
python -m src.train --config-name self_play data.val_file=data/phomt_dev.csv data.test_file=data/phomt_test_selected.csv data.steps_per_epoch=1000
```

`phomt_dev.csv` is a placeholder for your own development split; this feature does not create it or repurpose the test CSV. Training topics/memory do not draw from evaluation text. Default checkpoint selection uses `val/chrf`. Inspect independent corpus BLEU/chrF++ and translations before interpreting proxy-reward gains.

```powershell
python -m src.eval --config-name self_play_eval ckpt_path=logs/RUN/checkpoints/last.ckpt data.test_file=data/phomt_test_selected.csv
python -m src.train --config-name self_play ckpt_path=logs/RUN/checkpoints/last.ckpt trainer.max_epochs=2
```

Use the same policy/tokenizer, language direction, feedback configuration, prompts and generation settings on resume/evaluation. Evaluation uses the exact solver prompt and makes no feedback call; training checks for the API credential before starting. Python checkpoint loading requires the judge again: `SelfPlayGRPOModule.load_from_checkpoint(path, feedback=judge, weights_only=False)`. Load only trusted checkpoints. Inherited `export_pretrained` saves a full policy/tokenizer for full tuning, or PEFT adapters/tokenizer for LoRA.

## Tests and timing

```powershell
python -m pytest tests/grpo -q -o addopts='' --cov --cov-report=term-missing --cov-fail-under=90
```

Tests cover API payload/authentication, retry behavior, schemas, secret handling, frozen local feedback inference, both role rewards, real policy gradients, task replay, checkpoint resume, and held-out metrics. API responses and curriculum text are scripted for deterministic regressions; this does not measure learned proposal/translation quality. The integration test runs on CUDA when available. No paid API calls are required.

Self-play callbacks write `timing.json`: batch latency, optimizer steps, peak GPU allocation and feedback request/token counters. Batch timing includes proposal/solver generation, feedback wait and every optimizer iteration. It excludes model download/load; the training-loop total includes intervening validation. Use a warm pilot with the actual model and real feedback to estimate training cost.

## Time and cost scenarios

A subsequent authorized RTX 4090 rental completed successfully and was destroyed. The short-output Qwen2.5-0.5B pilot averaged 2.493 seconds per solver-active batch with 11.364 GiB peak allocation; feedback was mocked. See [measured results, prepared PhoMT split and cost estimates](../archive/reports/vast_results_2026-11-07.md). Live OpenRouter latency/authentication has not been tested.

The following are **planning assumptions**, not quotes or measured runtimes: B=1, P=2, G=4; 1,000 rollout batches; $0.30/GPU-hour; ten minutes setup and five minutes total validation. At ten feedback calls/batch averaging 350 input and 100 output tokens, there are 10,000 requests. Illustrative rates of $0.15/million input and $0.60/million output tokens produce $1.125 feedback cost. These rates match the requested snapshot's [catalog entry](https://openrouter.ai/api/v1/models), but provider/time-based overrides can change them.

| Assumed batch time | Wall time including overhead | GPU | Feedback | Combined |
|---|---:|---:|---:|---:|
| 5 seconds | 1.64 hours | $0.49 | $1.13 | $1.62 |
| 15 seconds | 4.42 hours | $1.33 | $1.13 | $2.45 |
| 30 seconds | 8.58 hours | $2.58 | $1.13 | $3.70 |

Replace assumptions with the measured pilot and current offer:

```powershell
python -m scripts.benchmarks.estimate --steps 1000 --seconds-per-step 15 --gpu-hourly-usd 0.30
```

Estimates exclude storage, transfers, tax, retries, unexpected reasoning tokens and extra memory-replay calls. Longer texts change both compute and judge costs. A twenty-minute GPU-only verification at the assumed $0.30/hour costs $0.10 plus storage/transfer. Image setup can dominate a short test; use the actual offer and a firm cleanup deadline when authentication is restored.
