# Framework validation

## Bilingual SFT/RL and COMETKiwi extension

The local conda environment passes **120 tests with 95.20% configured framework coverage**. This includes deterministic SFT/RL stage isolation, overlap checks in both languages, online alignment-preserving augmentation, bilingual prompts, exact per-direction metric accounting, supervised prompt/padding/EOS masks, full/adapter SFT checkpoint and dense export into GRPO, gated-model error handling without credential leakage, and synchronized timing semantics. Static checking covers 33 source/script files.

Real `Unbabel/wmt22-cometkiwi-da` inference was verified using the named Windows user-level `HUGGING_FACE_API_KEY`. Correct/unrelated synthetic candidate scores were approximately 0.857/0.364 for EN→VI and 0.846/0.374 for VI→EN. Warm CPU scoring took 0.294 seconds for four pairs, with four CPU threads. The evaluator remained frozen. The SFT → paired GRPO → shared-policy benchmark driver also passed locally on a tiny Qwen policy with the real evaluator; its judge responses were explicitly mocked for the self-play role pilot.

The original development/test CSVs are unchanged. Prepared stage counts are 3,800 SFT pairs and 950 RL pairs, with 250 development and 3,500 test pairs. The new loader expands clean test data to 7,000 directional examples. Full PhoMT model training and translation-quality evaluation have not been performed by these infrastructure checks. See [the workflow and timing guide](bidirectional_training.md).

The authorized RTX 4090 rental subsequently passed **120 tests at 95.21% coverage** and all full/LoRA/QLoRA SFT → paired GRPO → self-play pilots. Real COMETKiwi GPU scoring passed, and 138 mode/stage/phase rows include forward, backward, optimizer, generation, and reward timing. All nine stage/mode test reports contain both directions. The judge was mocked. The rental was destroyed, zero remaining instances verified, and the temporary SSH key removed. Observed account credit decreased by $0.2044; see [the complete report and time/cost scenarios](vast_bilingual_results_2026-11-08.md).

The inherited MNIST module also passed a constructor/forward smoke check with the COMET-compatible TorchMetrics version, using an explicit `top_k=1`. The dataset-downloading MNIST suite was not run.

## Transfer optimizer extension

Validated on **8 November 2026** using the existing conda environment at `E:/IAI/Lab/Research/venv`: **101 tests passed**, **95.67% configured framework coverage**. The transfer optimizer itself has 98.56% line coverage; the factory configuration helper has 97.06%. Strict mypy passes for 22 source/script files.

Tests compare disabled protection against native AdamW, compare L2-SP against an explicit regularized objective, verify projection with zero/nonzero anchors, exercise per-group rates and warmup, preserve frozen tensors even with stale gradients, and check exact optimizer continuation and original-anchor restoration. New full/LoRA/NF4 Lightning cases include validation and optimizer resume; self-play runs with both AdamW and transfer optimization on CPU and the local RTX 4060 Laptop GPU. New optimizer-factory checkpoints load with `torch.load(..., weights_only=True)`.

Logs and configured coverage are in `logs/transfer_full_tests.log` and `logs/transfer_coverage.json`. No GPU was rented for this extension. These infrastructure checks establish parameter-preservation behavior, not foundation-capability retention or translation-quality gains. See [the transfer guide](transfer_learning.md) for configuration, memory estimates, and retention evaluation.

## LoRA / QLoRA extension

The local conda environment and the rented RTX 4090 both pass **63 tests with 95.14% coverage** across the configured framework scope. New regressions cover causal and encoder-decoder adapters, an unchanged frozen base, zero-initialized reference equivalence, checkpointed likelihoods, log-probability microbatches, warmup, adapter-only checkpoint/export, and optimizer resume. A real NF4 tiny-Qwen training/resume test passes on both GPUs. Strict mypy passes locally for 19 source/script files; Ruff and dependency checks pass.

Tests use tiny random policies and synthetic/scripted rewards. See [Qwen 7B configuration and capacity measurements](lora_qwen7b.md) for the separate real-model memory probe. Live feedback latency and translation quality are outside these tests.

## Self-play extension verification

The self-play extension passes **50 tests with 95.16% coverage** across the configured framework scope, both locally and on the rented RTX 4090. This includes API transport/auth/retry/schema tests using in-process HTTP responses, actual frozen tiny Transformers feedback inference, shared-policy proposer/solver gradients, task memory replay, held-out evaluation, checkpoint/optimizer resume, and deterministic PhoMT data preparation. Strict mypy covers 17 source/script files. The additional evaluation experiment override is separately configuration-tested.

The local CUDA device remains the RTX 4060 Laptop GPU. Tests use tiny random policies and scripted task text/judgments for deterministic regressions; they do not measure translation quality or real feedback accuracy. The authenticated OpenRouter endpoint was not called because `CLAUDE_AUTH_KEY` is absent in the executing shell.

After the Vast.ai credential was refreshed, instance 59558218 completed the suite, six BF16 CLI smoke stages and a Qwen2.5-0.5B warm self-play benchmark. Results were downloaded and destruction verified. The [4090 report](vast_results_2026-11-07.md) records 2.493 seconds per solver-active short-output batch, 11.364 GiB peak GPU allocation, and an observed $0.0816 account credit decrease. Feedback was mocked, so live API latency and feedback accuracy were not measured.

See [self-play training, validation and cost scenarios](self_play.md). Actual batch throughput and API usage will be written to `timing.json` during a pilot; the scenario calculator explicitly distinguishes assumptions from measurements.

## Initial supervised-data GRPO baseline

Validated on 7 November 2026 with the existing conda environment at `E:/IAI/Lab/Research/venv`.

| Component | Version / result |
|---|---|
| Python | 3.11.17, conda |
| PyTorch | 2.10.0+cu126 |
| Lightning | 2.6.6 |
| Transformers | 4.57.6 |
| Hydra | 1.3.2 |
| Datasets | 4.8.5 |
| SacreBLEU | 2.6.0 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| CUDA runtime | 12.6 |
| GRPO test suite | 24 passed; two upstream Lightning deprecation warnings |
| Coverage | 94% overall across four new framework modules; objective and reward helpers 100% |
| Strict mypy | Passed, seven new source/script files |
| Ruff / Black | Passed, new framework code, scripts, and tests |
| pip check | No broken requirements |
| GPU FP32 | Both architectures: train, eval, and resume passed |
| GPU bf16 mixed | Both architectures: train, eval, and resume passed |
| Vast.ai | Prepared only; no rental executed |

CPU regression tests exercise real locally constructed T5 and GPT-2 policies, likelihood agreement with Transformers sampling, policy parameter changes, frozen reference parameters, corpus metrics, checkpoint and optimizer-state resume, HF export/reload, data splitting, EOS/PAD masking, language-token conditioning, and invalid hyperparameters.

GPU verification uses the Hydra CLI and tiny random local policies. Each precision mode runs six subprocesses. Resumed checkpoints are checked for `global_step=4`, reflecting two optimizer updates per rollout batch. Local run reports are in `logs/gpu_smoke/report.json` and `logs/gpu_smoke_bf16/report.json`, with detailed stage logs alongside them. Generated logs and checkpoints are intentionally ignored by Git.

These results verify software behavior on the local GPU. They do not establish performance on Vast.ai, compatibility with every Transformers architecture, real-model memory capacity, or translation-quality gains. The inherited MNIST suite was not run as part of this GRPO validation. CI workflow definitions were updated but remote CI has not been executed in this session.

Portable test bundles are checked for required configs/scripts, safe member paths, and exclusion of PhoMT files, credentials, and Git history. The subsequent authorized rental was completed and cleaned up; see the self-play verification above.
