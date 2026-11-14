# A100 training and monitoring

**This run completed and its rental was destroyed.** The requested fresh run is
documented in [fresh_a100_handoff.md](fresh_a100_handoff.md).

Instance **59675544**: **A100 SXM4 40 GB**, 128 GB host RAM, 100 GB disk,
**$0.401111/hour**. The 4090 (59667972) was destroyed before this rental.
The combined cap remains **$10**. Deadlines and endpoint are recorded in
`logs/training_a100/access/state.json` and `logs/training_active.json`.

## Monitor from Windows

```powershell
cd E:\IAI\Lab\Research\GRPO-MT
.\scripts\monitor_rental.ps1 train
.\scripts\monitor_rental.ps1 gpu
.\scripts\monitor_rental.ps1 status
```

Detach with **Ctrl-b, then d**. Inspect setup separately:

```powershell
ssh -t -i logs/training_a100/access/training_key -p 21702 root@38.65.91.243 "tmux attach -t mt-setup"
```

## Verify and resume on the rental

```bash
cd /workspace/grpo-mt
bash scripts/start_a100_session.sh verify
cat logs/a100_verify10/report.json
# After a successful report, with verification finished:
bash scripts/start_a100_session.sh train
tmux attach -t mt-train
```

Verification performs **exactly 10 optimizer updates**: two SFT updates and eight
GRPO updates over four grouped rollouts. It checks GPU placement of both models,
finite parameters, adapter updates, and held-out EN→VI / VI→EN translations.
The report records peak VRAM and forward, reward, backward and optimizer timings.
Verification updates are discarded; the source checkpoint is hash-checked.
Production requires a successful ten-update report.

**Verified on the rental:** all 10 updates passed in 324.9 seconds, with both
models on GPU, finite adapter parameters, changed adapters, unchanged source
checkpoint, and successful held-out translations in both directions. Peak
allocated memory was **32.89 GiB**; peak reserved memory was **37.72 GiB**.
Local evidence: `logs/training_a100/access/verification/report.json` and
`phase_timing/{events.jsonl,summary.json}`. Settled batches remain **24 / 48**.

Measured long-input GRPO rollouts average **69.71 seconds per batch**, including
two optimizer updates. GPU COMET averages **1.19 seconds per 96 candidates**.
Remaining RL training is estimated at **43 minutes / $0.29**, excluding
validation, final testing, and backup. Actual shorter inputs can finish sooner.

## Configuration

Preset: `configs/rental_a100.json`. Deployment settings:
`logs/production/rental_config.json`. Prompt/source limit **256 tokens**, target
limit **128 tokens**, BF16 rank-8 attention LoRA, gradient checkpointing,
COMET on `cuda:0`, scorer batch 4. Initial verification batches: **SFT 24 / GRPO 48**
(96 candidates). Final measured settings are recorded in the verification report.

SFT completed at step **2320**. GRPO resumes from step **18**, preserving Adam,
adapters, transfer anchors and remaining shuffled examples after 144/1900 views.
The dense SFT reference base is recreated from the saved checkpoint and checked
against every original export SHA-256 before resuming.

Reward: **70% Unbabel/wmt22-cometkiwi-da + 30% BLEU**. Disjoint 80% SFT / 20% RL
partitions, training-only online augmentation, 500 directional development
examples and 7000 final test examples covering both directions.

```bash
cat logs/production/rl/progress.json
tail -n 30 /workspace/train_driver.log
cat logs/production/rl/phase_timing/summary.json
# Request a checkpointed stop:
touch /workspace/grpo-mt/logs/production/STOP
```

## W&B, backup and budget

Project: [Machine Translation](https://wandb.ai/25025069-vietnam-national-university-hanoi/Machine%20Translation).
The Windows metrics relay keeps the W&B key local. Backups are under
`logs/training_a100/access/checkpoints_and_logs` locally.
Additional dense-model downloads are disabled (`cache_sft_base=false`);
lightweight adapter/optimizer checkpoints and logs remain backed up.
**Keep this workstation online for metrics, backups and automatic destruction.**

Restart the local supervisor only if it is not running:

```powershell
conda activate E:\IAI\Lab\Research\venv
cd E:\IAI\Lab\Research\GRPO-MT
python -m scripts.watch_vast_training --state logs/training_a100/access/state.json
```

`access/TUNING_HOLD` suppresses ordinary cleanup during setup/verification;
the independent hard deadline remains active. Remove it after production is
stable. Completion or quota stop triggers backup and rental destruction.
