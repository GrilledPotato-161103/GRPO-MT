# RTX 4090 verification and training estimate

Completed on 7 November 2026. Instance **59558218** was destroyed after artifacts were downloaded. Vast.ai then reported zero active instances. The task-specific SSH registration and local private key were removed; the cleanup watchdog was marked finished.

## Results

| Check | Result |
|---|---|
| Core GRPO/self-play suite | **50 passed**, 95.16% configured framework coverage |
| Encoder–decoder BF16 CLI | Train, evaluation and checkpoint resume passed |
| Causal BF16 CLI | Train, evaluation and checkpoint resume passed |
| Pretrained policy | Qwen/Qwen2.5-0.5B-Instruct |
| Warm self-play pilot | Eight rollout batches, 16 optimizer updates; passed |
| Solver-active timing | Seven batches; mean **2.493 seconds**, median 2.428 seconds |
| Peak allocated GPU memory | **11.364 GiB** |
| Fixed feedback | In-process mock HTTP responses; no paid OpenRouter calls |
| Final local source verification | 50 tests passed, 95.16% coverage, using the existing conda environment |

The rental ran Python 3.12.15, PyTorch 2.14.1+cu130, CUDA 13.0, Transformers 4.57.6 and Lightning 2.6.6 on an NVIDIA GeForce RTX 4090. Local verification used the existing Python 3.11 conda environment and RTX 4060 Laptop GPU.

The full test suite exercised feedback payload/authentication/retry handling, strict judge JSON parsing, frozen local Transformers inference, gradients, frozen references, task memory, checkpoint/optimizer resume, and held-out metrics. A small follow-up timing refinement recorded solver work per batch; the final source was rechecked locally and exercised by the warm remote benchmark.

The pretrained pilot used B=1, P=2 proposer samples, G=2 solver samples, two optimizer iterations, source cap 512, proposer cap 96 tokens and solver cap 32 tokens, BF16 mixed precision. Solver tasks per batch were `[0,1,1,1,1,1,1,1]`; the first batch bootstrapped the curriculum and was excluded from warm timing. There were five validated new tasks, 14 scored solver candidates and 19 mock feedback requests. This reflects a mixture of new-task and memory-replay work, not an all-valid maximum-load curriculum.

The pilot verifies real model loading and policy optimization, not the accuracy of the feedback model or an improvement in translation quality. Evaluation used tiny synthetic examples; its BLEU/chrF++ values are not PhoMT benchmark results. Live OpenRouter latency and authentication remain untested because no real feedback credential was used.

## Rental cost evidence

- Accepted offer: GPU $0.40/hour plus 20 GB storage $0.007407/hour, **$0.407407/hour total**; transfer quote $0.004/GiB each way.
- Created around **18:01:21 Asia/Saigon**; cleanup requested around **18:22:07**. Total instance lifetime was 20 minutes 46 seconds, including image loading.
- Observed account credit: $4.67086163206001 before rental, $4.5892489394199885 after cleanup: **$0.0816126926400215 decrease**.
- Applying the full hourly quote to the complete lifetime gives $0.1410 before transfers; this is a conservative arithmetic estimate because GPU billing did not cover the entire image-loading interval.
- The optional itemized charge query failed in the installed Vast.ai CLI with exit code 1 and `TypeError: can only concatenate str (not "int") to str`. It was not retried through another CLI or direct API. The balance delta is an observation rather than an itemized invoice.

## Training and validation readiness

The local preparation script created `data/phomt_prepared/train.csv` with **4,750 pairs**, `data/phomt_prepared/validation.csv` with **250 pairs**, and a seed/input-hash manifest. The original **3,500-pair test CSV** is unchanged. Normalized English source overlap across the original train/test files was zero; the development split is deterministic and disjoint. No PhoMT CSV or real credential was uploaded to the rental.

Reproduce preparation on a fresh checkout with `python -m scripts.prepare_phomt`. It refuses to overwrite existing prepared files or silently discard duplicates/overlap. The `experiment=self_play_phomt` configuration uses the prepared development split and original held-out test set. Self-play training still uses policy-generated sources; the prepared training subset is available for baseline/SFT comparisons.

After configuring an OpenRouter credential in `CLAUDE_AUTH_KEY`, start a pilot matching the short-output benchmark:

```powershell
python -m src.train --config-name self_play experiment=self_play_phomt data.steps_per_epoch=5 model.group_size=2 model.proposal_group_size=2 model.max_source_length=512 model.max_new_tokens=32 model.proposal_max_new_tokens=96 model.memory_examples=1 trainer.accelerator=gpu +trainer.precision=bf16-mixed
```

Use `data.steps_per_epoch=1000` only after inspecting a pilot's proposal-validity rate, truncation and translations. Increase the completion cap for realistic translation lengths as needed, then remeasure; timings below do not establish throughput for the default G=4/128-token configuration. Evaluation is:

```powershell
python -m src.eval --config-name self_play_eval experiment=self_play_phomt ckpt_path=logs/RUN/checkpoints/last.ckpt
```

The `self_play_eval` entry point supports the same experiment override and solver settings as training. Keep the full model/tokenizer/language/prompt/generation overrides consistent with the training run; the shortened command above illustrates checkpoint and data selection only.

## Time and cost extrapolation

For **1,000 rollout batches / 2,000 optimizer updates** at the measured short-output workload, compute-only extrapolation is **41.56 minutes**, or **$0.282 GPU/storage cost** at the accepted hourly rate. This excludes model setup, API wait, real feedback charges and full validation/test. Eight pilot batches are a small sample; curriculum and output lengths will evolve.

For planning, assume two sequential feedback waves per batch, concurrency four, up to six feedback requests per batch, and 350 input/100 output tokens per request. At the earlier catalog snapshot's illustrative $0.15/million input and $0.60/million output rates, 6,000 requests cost **$0.675**. These are token and latency assumptions, not observed API usage or a current price guarantee.

| Assumed live feedback latency per wave | Estimated compute + feedback-wait time for 1,000 batches | GPU/storage estimate | Feedback estimate | Subtotal |
|---|---:|---:|---:|---:|
| 1 second | 74.9 minutes | $0.508 | $0.675 | $1.183 |
| 3 seconds | 141.6 minutes | $0.961 | $0.675 | $1.636 |

Add setup and evaluation separately. For example, an assumed batch-one evaluation time of 0.5–2 seconds/pair gives **2.1–8.3 minutes** for the 250-pair development split and **29.2–116.7 minutes** for the 3,500-pair test set. Together those would add roughly **$0.21–$0.85 GPU/storage cost**, plus any setup time. Full PhoMT evaluation throughput was not measured. Evaluation uses local corpus metrics and does not call the feedback API.

Larger G, longer completions, higher task acceptance, repeated validation epochs, retries and provider latency can increase runtime. The existing `scripts.estimate_self_play` accepts an observed end-to-end seconds-per-batch and an hourly price; supply the matching rollout configuration through its Python function when using G=2 instead of its default G=4. Save each pilot's `timing.json` with its Hydra config before extrapolating.

## Saved evidence and reproducibility

Remote evidence is saved locally under `logs/vast_59558218/grpo-results/`, including `pytest.log`, `coverage.json`, hardware metadata, six GPU smoke stages, and both pretrained pilot reports. The downloaded archive is `logs/vast_59558218_results.tar.gz`. Files under `logs` are intentionally excluded from Git.

`scripts/run_vast_verification.sh` executes the original full verification flow inside the rented CUDA environment. `scripts/benchmark_self_play.py --steps 8 --mock-feedback` reproduces the explicit mock-feedback pilot. Without `--mock-feedback`, it uses the configured real OpenRouter credential and can incur API charges. Full benchmark checkpoint writing is opt-in through `--save-checkpoint` to avoid large unnecessary test artifacts.
