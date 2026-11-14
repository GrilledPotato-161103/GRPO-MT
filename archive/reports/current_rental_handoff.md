# Current RTX 4090 run: commands and configuration

**Historical first-instance notes.** Instance 59656607 no longer exists. Use [the active replacement handoff](active_rental_handoff.md) for current commands and configuration.

Instance **59656607**, one RTX 4090, 100 GB disk, **$0.347777778/hour**. The authorized total budget is **$10**, including a transfer reserve. W&B project: [Machine Translation](https://wandb.ai/25025069-vietnam-national-university-hanoi/Machine%20Translation). [SFT run](https://wandb.ai/25025069-vietnam-national-university-hanoi/Machine%20Translation/runs/phomt-7b-20261108-030626-sft).

The current batch tuner preserves `logs/production/sft/checkpoints/interrupted.ckpt`. It probes full 512-source/128-target-token shapes with actual backward/Adam work, then writes separate SFT and GRPO choices to `logs/production/batch_sizes.json`. Probe updates are discarded; production resumes from the unchanged interrupted checkpoint. This targets 90% **peak allocated VRAM**; actual shorter batches use less memory. Do not run a tuner and training concurrently.

## Connect from Windows

```powershell
cd E:\IAI\Lab\Research\GRPO-MT
conda activate E:\IAI\Lab\Research\venv
$state = Get-Content logs/training_20261108/access/state.json -Raw | ConvertFrom-Json
ssh -i $state.key -p $state.port -o "UserKnownHostsFile=$($state.known_hosts)" $state.host
```

The SSH endpoint is `root@ssh2.vast.ai:15854`. The temporary private key remains in the ignored local task folder; do not publish it.

## On the rental: inspect or resume

```bash
cd /workspace/grpo-mt
source /venv/main/bin/activate
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so.1
export HF_HOME=/workspace/hf-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
pgrep -af 'tune_batch_size|train_bilingual_rental'
tail -n 15 /workspace/batch_tuning.log
cat logs/production/batch_sizes.json
cat logs/production/state.json
cat logs/production/sft/progress.json
```

If the tuner has completed and no training process is running, resume with:

```bash
nohup python -m scripts.resume_bilingual_rental \
  --output logs/production \
  --deadline 1794200345.9392972 \
  --group phomt-7b-20261108-030626 \
  >> /workspace/train_driver.log 2>&1 < /dev/null &
echo $! > /workspace/training_driver.pid
```

The helper keeps adapter weights, Adam state, transfer anchors, global optimizer step, and warmup progress. It resets batch counters and skips the already consumed examples in the original seeded permutation, so changing batch size does not skip the rest of the epoch or repeat the consumed data. The current implementation requires the original first-epoch SFT run with batch size one and zero data workers. It refuses an incomplete batch-probe result or an expired budget deadline.

The supplied pipeline completes the remaining SFT data, validates both directions, exports `exports/phomt_sft`, trains paired GRPO on the disjoint RL partition, and tests both directions when budget permits.

## Configuration

- Base presets: `configs/experiment/phomt_sft.yaml` and `phomt_rl.yaml`.
- Runtime settings and measured batch choices: `scripts/train_bilingual_rental.py`, `logs/production/batch_sizes.json`, and `logs/production/rebatch_resume.json`.
- Exact resolved settings: `logs/production/sft/resolved_config.yaml`, then `logs/production/rl/resolved_config.yaml`.
- Qwen2.5-7B-Instruct revision: `a09a35458c702b33eeacc393d103063234e8bc28`.
- BF16 rank-8 attention LoRA; gradient checkpointing; TransferAdamW; SFT learning rate 1e-4; RL 5e-6; 100 warmup steps; source/target caps 512/128; two candidates and two updates per GRPO rollout.
- Reward: 70% frozen CPU COMETKiwi and 30% BLEU; online augmentation and bidirectional EN↔VI views.

## W&B relay and automatic cleanup on Windows

W&B receives metrics only through this workstation. `WANDB_API_KEY` stays local; checkpoint weights stay in the rental/local backups. Keep the workstation online for the local relay and cleanup guards.

The tuning hold file prevents accidental cleanup of the deliberately stopped run. After the remote job reports `resuming` or `training`, remove that hold and restart the local supervisor if it is not already running:

```powershell
cd E:\IAI\Lab\Research\GRPO-MT
conda activate E:\IAI\Lab\Research\venv
Remove-Item -LiteralPath logs/training_20261108/access/TUNING_HOLD -ErrorAction SilentlyContinue
python -m scripts.watch_vast_training --state logs/training_20261108/access/state.json
```

That foreground command must remain running; alternatively use `Start-Process -WindowStyle Hidden` with stdout/stderr redirected to the task logs. The independent hard watchdog was already launched; its PID is in `access/hard_watchdog_pid.txt`. Local state contains the graceful and hard UTC deadlines. Changing batch size does not extend the $10 budget.

Backups are under `logs/training_20261108/access/checkpoints_and_logs`. Public policy/encoder assets and all data are already cached on the rental. Full local model snapshot copies are additional backup work and may still be pending; the original source/data/COMET files remain local.

To request graceful stop on the rental:

```bash
touch /workspace/grpo-mt/logs/production/STOP
```

Wait for `state.json` to say `quota_stopped` and verify `sft/checkpoints/interrupted.ckpt` or the RL checkpoint. Download required backups before destroying the instance. Official manual destruction command:

```powershell
vastai destroy instance 59656607 -y --raw
vastai show instances-v1 --limit 25 --raw
```

To rerun the capacity probe after changing token caps, first checkpoint/stop training, then use `python -m scripts.tune_batch_size --checkpoint logs/production/sft/checkpoints/interrupted.ckpt --output logs/production/batch_sizes.json`. The current probe itself fixes 512/128 token caps; edit and review it if those production caps change.
