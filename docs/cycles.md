# Sampled SFT / GRPO cycles

This optional/reference guide retains earlier experiments and measured results.
Use [the execution manual](execution.md) and `configs/runs/a100.yaml` for the
current COMET/SAC pipeline. Archived shell launchers describe historical runs.

**Latest status, November 11:** all 10 cycles and final official-test evaluation completed. Instance 60031289 was destroyed after verified local downloads. See [the final report](../archive/reports/final_test_and_cleanup_20261111.md) for checkpoints, predictions and all metric logs.

Live run: A100 **60031289**, tmux **`mt-cycles`**, continuing from step 24,393.
Handoff verified after **93 successful SFT updates** in the new schedule;
the backbone remained frozen. Both real-GPU probe phases and their evaluations
passed. Local checks covered two cycles, optimizer continuity, reference
immutability, interrupted-phase resume and completed-phase skipping.

[W&B cycle overview](https://wandb.ai/25025069-vietnam-national-university-hanoi/Machine%20Translation/runs/cyclic-phomt-20261110-overview).

Each cycle runs **1,000 SFT batches**, evaluates, runs **200 GRPO optimizer
updates**, then evaluates again. At batch sizes 24 and 16, with two GRPO
updates per rollout, this means 12,000 SFT pairs and 800 RL pairs per cycle.
Each pair contributes EN→VI and VI→EN. “200 steps” counts optimizer updates,
not rollouts: the RL phase has 100 rollout batches.

The starting policy and immutable GRPO reference are the saved SFT checkpoint
at update **24,393**. A separate frozen LoRA adapter stores that reference on
the same Hugging Face foundation; a second 7B backbone and repeated dense
model exports are unnecessary. The backbone and reference adapter stay frozen.
The next SFT phase continues the preceding RL phase's policy weights.
SFT and RL retain separate optimizer moments, transfer anchors and cumulative
warmup counters across cycles. The first SFT phase restores the old optimizer;
the first RL phase initializes its own optimizer. KL is fixed at the configured
0.01 throughout this new ablation and never re-anchored between cycles.

Training draws from the existing, disjoint 80% SFT and 20% RL partitions of the
full prepared corpus. Each partition has its own deterministic shuffled stream:
successive cycles consume different pairs until that partition is exhausted,
then a newly shuffled sweep begins. Each sample records its original row indices
and hash. Whitespace, Unicode and aligned concatenation remain training-only.

A **fixed, seeded subset of 1,000 validation pairs** is selected from the
2,000-pair validation split: **2,000 translations per evaluation**, balanced
across both directions. Both phases report corpus BLEU, chrF++ and COMETKiwi
using identical prompts, token caps and evaluation batch size. Compare cycle evaluations with one another; their 1,000-pair subset differs
from the earlier 2,000-pair evaluation. The official 19,151-pair test is untouched. Source/target caps remain 256/128, with rank-8
LoRA, BF16, and 16 persistent data-loader workers on the A100.

Configuration: [`configs/runs/cycles.yaml`](../configs/runs/cycles.yaml).
The default trial has **10 cycles**, configurable with `--set cycles=N`.
This is a sampled-cycle count, not ten complete passes over the corpus.
The runner performs no rental shutdown, destruction or billing control.

From the prepared rental:

```bash
cd /workspace/grpo-mt
source /venv/main/bin/activate
python -m scripts.main.cycles --config /workspace/cycles-run.yaml --dry-run
python -u -m scripts.main.cycles --config /workspace/cycles-run.yaml --trust-checkpoint
```

The rental YAML sets `base_config: /workspace/full-run.yaml`, retains existing
cached assets and dataset paths, and points `initial_checkpoint` to
`logs/full-phomt-20261110/sft/checkpoints/interrupted.ckpt`. Keep keys in environment
variables. No model or dataset transfer is required on this host.

On Windows, use `../venv/python.exe -m scripts.main.cycles --dry-run`.

W&B project **Machine Translation** has an overview run plotting `evaluation/*`
against `phase_index`, plus `sft_eval/*` and `rl_eval/*` against `cycle`.
Each phase has a separate run with on-step metrics and CUDA-synchronized timings.
Overview cumulative updates include the original 24,393 SFT updates; phase
progress counts are local.

Output: `logs/cyclic-phomt-20261110/`:

- `cycle_state.json`: completed phases and current status; `latest.json`: latest
  completed phase, checkpoint, metrics and cumulative update counts.
- `validation.csv` and `validation.sampling.json`: fixed evaluation subset.
- `cycle_001/sft/` and `cycle_001/rl/`: sample CSVs, row indices, immutable plans,
  diagnostics, timings, metrics and checkpoints.
- `checkpoints/trained.ckpt` in a phase: weights and optimizer saved before
  evaluation; a failed evaluation can be retried without more training.

Request a checkpointed stop:

```bash
touch /workspace/grpo-mt/logs/cyclic-phomt-20261110/STOP
```

After exit, explicitly remove that STOP file, then resume:

```bash
python -u -m scripts.main.cycles --config /workspace/cycles-run.yaml \
  --resume --trust-checkpoint
```

Completed phases are skipped. Partial phases resume a Lightning checkpoint.
Worker prefetch and stochastic augmentation do not guarantee a byte-identical
sample stream after interruption; optimizer-update budgets remain exact.
Only remove a stale `.cycle.lock` after confirming its launcher has exited.

The Rich display is in tmux session `mt-cycles`; detach with **Ctrl-b, then d**.
From Windows, `scripts/rental/monitor.ps1 train` uses the active session recorded
in `logs/training_active.json`.
