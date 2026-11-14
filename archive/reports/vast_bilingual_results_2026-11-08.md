# Bilingual SFT, GRPO, and COMETKiwi: RTX 4090 verification

The authorized rental completed on **8 November 2026**. All **120 framework tests passed**, with **95.21% coverage** over the configured framework scope. Full tuning, LoRA, and NF4 QLoRA each completed supervised training, dense SFT export, paired-data GRPO, shared-policy proposer/solver training, validation, and bidirectional testing. The real COMETKiwi evaluator ran on the GPU. Self-play judge responses were explicitly mocked.

## Workload and reproducibility

The policy was `Qwen/Qwen2.5-0.5B-Instruct`, with batch size 1, group size 2, three training batches per stage, and two optimizer iterations per RL rollout. Thus SFT performed three updates and each RL stage performed six. Source prompts were capped at 128 tokens, translations at 16, and proposals at 24. LoRA rank was 8, attention projections only. TransferAdamW, gradient checkpointing, row-wise log-probability microbatching, and BF16 mixed precision were enabled. Full tuning stored policy weights in FP32; LoRA used BF16 base weights; QLoRA used NF4 and FP32 non-quantized parameters. These are memory/runtime checks of the supplied modes, rather than a controlled comparison at identical weight precision.

Only synthetic examples and public model artifacts were uploaded. Synthetic data comprised 20 training pairs split into 16 SFT / 4 RL, plus two development and two test pairs. Each test pair was evaluated in both directions: two EN→VI and two VI→EN examples per stage. Both shared-policy roles executed real generation and gradient calculations; seeded task memory guaranteed solver activity despite short, potentially malformed proposals. Mock judge latency is labeled separately in the CSV and is not live OpenRouter latency.

Hardware/software: RTX 4090, 23.65 GiB reported GPU memory; Python 3.12.15; PyTorch 2.14.1+cu126; Lightning 2.6.6; Transformers 4.57.6; PEFT 0.18.1; bitsandbytes 0.50.2; unbabel-comet 2.2.7; NumPy 1.26.4; TorchMetrics 0.10.3. Dependency validation passed. Host driver discovery required preloading its native `libcuda.so.1`; the runner includes that preflight.

The local conda environment separately passed 120 tests at 95.20% coverage and strict mypy for 33 source/script files. A constructor/forward compatibility smoke check passed for the inherited MNIST module after making its default `top_k=1` explicit for COMET's older TorchMetrics dependency.

## Measured training time and GPU memory

Times below are **mean seconds per training batch**, including cold calls. Peak allocation includes the GPU-resident COMETKiwi evaluator where loaded. Total mode runtime includes loading, export, training, validation, testing, and reporting; the first mode also encounters cold model assets. The short runs do not support claims about statistically significant differences.

| Mode | SFT batch | Paired GRPO batch | Self-play batch | Peak allocated GiB, SFT / GRPO / self-play | Total mode seconds |
|---|---:|---:|---:|---|---:|
| Full | 1.005 | 2.172 | 4.790 | 12.911 / 18.702 / 18.703 | 92.505 |
| LoRA | 1.042 | 2.697 | 5.772 | 3.070 / 3.110 / 3.151 | 75.516 |
| QLoRA | 1.104 | 3.454 | 7.294 | 3.074 / 3.373 / 3.936 | 89.536 |

SFT batch medians were 0.348 / 0.382 / 0.461 seconds, showing substantial first-call overhead. Paired GRPO medians were 2.042 / 2.678 / 3.405 seconds. Profiling synchronizes CUDA and adds logging overhead. Reserved memory can exceed allocated memory; full GRPO reserved 20.783 GiB.

The following are **mean milliseconds per invocation**. Paired GRPO and self-play invoke current forward, backward, and optimizer twice per training batch. Shared-policy backward differentiates the combined proposer/solver loss once per optimizer iteration, so it is reported jointly.

| Mode | Stage / role | Current forward ms | Backward ms | Optimizer ms |
|---|---|---:|---:|---:|
| Full | SFT | 650.59 | 166.43 | 141.70 |
| Full | Paired GRPO | 131.46 | 324.51 | 125.65 |
| Full | Proposer | 140.26 | joint below | joint below |
| Full | Solver | 242.81 | joint below | joint below |
| Full | Self-play joint | — | 768.77 | 123.46 |
| LoRA | SFT | 672.83 | 225.31 | 87.78 |
| LoRA | Paired GRPO | 190.79 | 430.86 | 67.48 |
| LoRA | Proposer | 206.41 | joint below | joint below |
| LoRA | Solver | 197.34 | joint below | joint below |
| LoRA | Self-play joint | — | 886.61 | 65.92 |
| QLoRA | SFT | 690.20 | 274.30 | 79.10 |
| QLoRA | Paired GRPO | 261.10 | 546.10 | 61.31 |
| QLoRA | Proposer | 267.53 | joint below | joint below |
| QLoRA | Solver | 260.92 | joint below | joint below |
| QLoRA | Self-play joint | — | 1060.45 | 59.52 |

Backward includes gradient-checkpoint recomputation. Frozen reference/old-policy likelihoods and COMETKiwi have no backward pass. Parent batch totals overlap child phases and must not be added to them.

| Mode | Paired generation output tokens/s | Proposer generation output tokens/s | Solver generation output tokens/s |
|---|---:|---:|---:|
| Full | 41.17 | 40.71 | 50.19 |
| LoRA | 30.84 | 26.85 | 28.34 |
| QLoRA | 24.45 | 22.07 | 23.07 |

All **138 mode/stage/phase rows**, including old/reference forward, augmentation, BLEU, COMET load/tokenization/forward, evaluation generation, and mock feedback, are in `logs/vast_bilingual_59643586/translation-results/policy_speed.csv`. Rows identify the policy model and measured component separately. The CSV includes call counts, total/mean/median/p95/min/max seconds, token/sample counts, and throughput. Original per-event JSONL and summary JSON files remain under `benchmarks/MODE/STAGE/phase_timing/`. The downloaded archive preserves the runner's original CSV; the extracted CSV was regenerated with the improved component labels and count columns.

## Real COMETKiwi verification

The evaluator was [Unbabel/wmt22-cometkiwi-da](https://huggingface.co/Unbabel/wmt22-cometkiwi-da), snapshot `1ad785194e391eebc6c53e2d0776cada8f83179a`. It scores source/candidate pairs without references. All evaluator parameters were frozen and inference explicitly disabled policy AMP, retaining FP32 evaluator precision.

| Direction | Correct synthetic candidate | Unrelated synthetic candidate |
|---|---:|---:|
| EN→VI | 0.857143 | 0.363864 |
| VI→EN | 0.845930 | 0.373527 |

Warm scoring of all four short pairs took **20.382 ms**, or **196.25 pairs/s**, including tokenization and device transfer. Warm model forward took 18.918 ms. Cold scoring took 15.666 seconds, including 15.183 seconds of initialization. Training reward calls often contain one or two candidates, so their throughput differs from this four-pair probe. These scores test wiring and a simple quality ordering, not evaluator calibration on PhoMT.

Access succeeded using the named Windows user-level `HUGGING_FACE_API_KEY`. Automatic approval review rejected exporting the credential to the rented host. The safe completed path downloaded the official checkpoint and hyperparameters locally, then uploaded only those model files. No Hugging Face or OpenRouter credential was copied to the rental.

## Training readiness and time/cost scenarios

The real local dataset is prepared independently: 3,800 SFT pairs, 950 RL pairs, 250 development pairs, and 3,500 original test pairs. Direction expansion yields 7,600 / 1,900 / 500 / 7,000 directional examples. Normalized source and target overlap checks passed across stage/development/test partitions. Stage generation occurred before direction expansion or training-only augmentation. The [workflow guide](bidirectional_training.md) contains training, export, and held-out evaluation commands.

The following is an **illustrative count extrapolation for the tested 0.5B policy and short 128/16-token shapes**, using observed batch medians and warm evaluation/reward medians. It assumes one SFT epoch, one paired GRPO epoch, one 500-example validation pass after each stage, and one final 7,000-example test. It excludes setup, downloads, exports, checkpoints, extra validation passes, and live feedback. The rate is the observed rental offer's $0.411111/hour, including its allocated disk.

| Mode | SFT training minutes | RL training minutes | Two validation passes minutes | Final test minutes | Total hours | Scenario rental cost |
|---|---:|---:|---:|---:|---:|---:|
| Full | 44.13 | 64.68 | 10.82 | 76.35 | 3.27 | $1.34 |
| LoRA | 48.39 | 84.81 | 16.05 | 113.67 | 4.38 | $1.80 |
| QLoRA | 58.40 | 107.81 | 20.37 | 143.52 | 5.50 | $2.26 |

Training time is `directional_examples × median_batch_seconds` at batch size 1. Evaluation time is `directional_examples × (median_generation + median_COMET + median_BLEU)`. Only three training batches informed each median. These numbers are shape-limited planning scenarios; longer real translations and the supplied **7B, 512/64-token presets require a separate pilot**. They are not estimates of full 7B training. The 7B configuration defaults COMETKiwi to CPU to conserve GPU memory; prior 7B capacity measurements did not include a GPU-resident COMETKiwi evaluator.

## Rental cost and cleanup

Instance **59643586** used one verified RTX 4090 in South Korea, with 40 GB disk at **$0.411111111/hour**. It was created at **00:46:41 UTC** and destruction was verified at **01:15:18 UTC**, approximately 28 minutes 38 seconds including SSH setup, installation, model transfer, testing, and report download. This is **07:46:41–08:15:18 in Asia/Saigon**.

Observed account credit changed from **$4.3280559007 to $4.1236450449**, a decrease of **$0.2044108558**. This is an account-credit observation, not an itemized invoice; billing can settle later. The user-requested deferred itemized-billing investigation was not retried.

The official Vast.ai listing confirmed **zero remaining instances** after destruction. The temporary SSH registration and local test key files were removed, while the preexisting user key was preserved. The local watchdog was marked finished. Cleanup/account records are in `logs/bilingual_vast_cleanup.json` and `logs/bilingual_vast_account_{before,after}.json`.

No full PhoMT training, 7,000-translation quality benchmark, live API judge call, or new Qwen 7B capacity test was performed during this verification.
