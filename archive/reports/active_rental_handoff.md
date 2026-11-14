# Current rental: run and monitor

**Superseded:** the 4090 below was destroyed before the A100 rental. Use
[the A100 hand-off](a100_handoff.md) and `logs/training_active.json`.

Replacement instance **59667972**, `root@38.117.87.43:43933`, costs **$0.49037037/hour**. The combined budget across both rentals remains **$10**. `logs/training_active.json` contains the current endpoint, key path, run group, and deadlines. The original instance 59656607 no longer exists; recovery uses its locally preserved step-1,000 checkpoint.

## Monitor from Windows

```powershell
cd E:\IAI\Lab\Research\GRPO-MT
.\scripts\monitor_rental.ps1 train
.\scripts\monitor_rental.ps1 gpu
.\scripts\monitor_rental.ps1 status
```

The first two commands attach to named `tmux` sessions. **Detach with Ctrl-b, then d**; training keeps running. If local PowerShell scripts are blocked, use `powershell -ExecutionPolicy Bypass -File .\scripts\monitor_rental.ps1 train` for that invocation.

Equivalent direct command:

```powershell
ssh -t -i logs/training_20261108_recovery/access/training_key -p 43933 root@38.117.87.43 "tmux attach -t mt-train"
```

## Start or inspect on the rental

```bash
cd /workspace/grpo-mt
bash scripts/start_rental_training.sh
tmux attach -t mt-train
# GPU view:
tmux attach -t mt-monitor
# Deployment, while still active:
tmux attach -t mt-setup
```

The startup script refuses duplicate training/tuning processes. `run_training_session.sh` installs/checks dependencies, validates CUDA, caches the pinned models, measures batch sizes, and resumes SFT followed by paired GRPO. Exited panes remain visible for diagnosis. Remove a **dead** `mt-train` session only after correcting its error, then rerun the startup script.

```bash
cat logs/production/state.json
cat logs/production/batch_sizes.json
cat logs/production/sft/progress.json
cat logs/production/rl/progress.json
tail -n 30 /workspace/train_driver.log
```

## Configuration

**Settled configuration:** SFT batch **5**, paired GRPO batch **16** (32 candidates). Full-cap measured allocation peaks are 20.40 / 20.90 GiB, about 87% / 89% of usable VRAM. Variable token-budget batching is disabled after a real-data OOM; `use_token_budget` defaults to false. The expandable CUDA allocator is enabled. COMET stays on CPU and consumes no training-GPU VRAM. A second feedback rental/A100 has not been rented.

| Setting | Location |
|---|---|
| Group, deadlines, pinned revision | `logs/production/rental_config.json` |
| Measured SFT / GRPO batch sizes | `logs/production/batch_sizes.json` |
| Already consumed SFT example count | `logs/production/rebatch_resume.json` |
| Exact resolved configurations | `logs/production/{sft,rl}/resolved_config.yaml` |
| Base presets | `configs/experiment/phomt_sft.yaml`, `phomt_rl.yaml` |
| Runtime overrides | `scripts/train_bilingual_rental.py` |
| Recovery checkpoint | `logs/production/sft/checkpoints/interrupted.ckpt` |
| Periodic checkpoints | `logs/production/{sft,rl}/checkpoints/periodic/last.ckpt` |

The tuner targets **90% peak allocated VRAM** at **512 source / 128 target tokens**, including backward and Adam state. GRPO also exercises grouped generation and two optimizer updates. Shorter sentences can use less memory. Probe updates are discarded. Resume preserves adapter weights, Adam state, transfer anchors, global step, warmup, and the remaining original shuffled examples. Single-worker augmentation continuation and checkpoint migration are tested.

Policy: **Qwen2.5-7B-Instruct**, revision `a09a35458c702b33eeacc393d103063234e8bc28`; BF16 rank-8 attention LoRA, gradient checkpointing, TransferAdamW. SFT learning rate 1e-4; RL 5e-6; 100 warmup steps. One SFT epoch then one paired GRPO epoch. The 3,800 SFT / 950 RL pairs remain disjoint; development has 500 directional examples and final testing has 7,000. Reward: 70% frozen CPU COMETKiwi / 30% BLEU.

## W&B, backup, and quota control

Project: [Machine Translation](https://wandb.ai/25025069-vietnam-national-university-hanoi/Machine%20Translation). [Recovery SFT run](https://wandb.ai/25025069-vietnam-national-university-hanoi/Machine%20Translation/runs/phomt-7b-recovery-20261108-050610-sft); its RL run is created when that stage starts.

Metrics are relayed through the Windows workstation. The W&B key stays local; checkpoints stay in rental/local storage. **Keep this workstation online for the relay and automatic cleanup.** The hard watchdog remains independent of the main supervisor. Local checkpoints/reports are under `logs/training_20261108_recovery/access/checkpoints_and_logs`; large model caches run in the background.

To restart the local supervisor if it is not already running:

```powershell
conda activate E:\IAI\Lab\Research\venv
cd E:\IAI\Lab\Research\GRPO-MT
python -m scripts.watch_vast_training --state logs/training_20261108_recovery/access/state.json
```

During intentional setup/tuning, `access/TUNING_HOLD` suppresses ordinary cleanup; the hard budget limit remains active. Remove that marker only after the remote state says `resuming` or `training`.

Graceful stop on the rental:

```bash
touch /workspace/grpo-mt/logs/production/STOP
```

Wait for checkpoint and backup completion. The supervisor handles normal completion/quota cleanup. Manual destruction, **after preserving needed files**:

```powershell
vastai destroy instance 59667972 -y --raw
vastai show instances-v1 --limit 25 --raw
```
