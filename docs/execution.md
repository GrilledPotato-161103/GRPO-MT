# Execution

Run commands from the GRPO-MT root. No rental is currently managed by these
scripts. Rent the GPU yourself using your existing SSH key, then upload the
source and data. Use a PyTorch image; 40 GiB A100 is the prepared target.

## Environment

Locally, use the existing Conda environment:

```bash
conda activate ../venv
python -m pip install -r requirements.txt
```

On a PyTorch Linux rental, activate `/venv/main/bin/activate` if available and run:

```bash
bash scripts/rental/setup.sh
```

Install tmux with `apt-get update && apt-get install -y tmux` when absent.
Export `ANTHROPIC_AUTH_TOKEN` (an OpenRouter key), `HUGGING_FACE_API_KEY`
(COMETKiwi access), and `WANDB_API_KEY` securely in the shell. Export the keys before
starting the launcher. Never put keys in YAML or commands that will be saved
to logs. `.env.example` lists the variable names.

## Prepare and preview

The [data guide](data.md) describes the full corpus and existing split reuse.
Training needs four CSVs and a manifest, at the paths in `data_files` and
`dataset_manifest`. Model weights are downloaded through Hugging Face; no
previous checkpoint is needed for a fresh run.

```bash
python -m scripts.main.prepare --index-only
python -m scripts.main.train --config configs/runs/a100.yaml --dry-run
```

Dry-run resolves both Hydra stages without downloading models or opening
checkpoints. It needs no service keys. The A100 configuration uses SFT batch
24, RL batch 16, four candidates per source, two updates per rollout,
log-probability microbatch 8 with a 3,072-token graph budget, caps 256/128,
12 data workers, and evaluation batch 16. For 24 GiB, use `rtx4090.yaml`.
These settings bound sequence shapes; verify memory on the actual GPU.

## Start and monitor

```bash
bash scripts/rental/start.sh --config configs/runs/a100.yaml
tmux attach -t mt-train
```

Detach with Ctrl-b then d. The launcher forwards the calling shell environment
through a restricted RAM pipe, including keys set after an earlier tmux server
started. It leaves the process running and records console output under
`logs/sessions/`. Starting twice refuses to reuse an existing `mt-train` session.
Inspect a completed pane before removing that tmux session yourself.

For a foreground run inside your own tmux session:

```bash
python -m scripts.main.train --config configs/runs/a100.yaml
```

Rich progress is enabled. W&B project is `Machine Translation`; set
`--set wandb.entity=YOUR_ENTITY` when needed. Local progress is in
`logs/sac-a100-production/{sft,rl}/progress.json`; phase timings and metrics are
stored beside it. `nvidia-smi` shows GPU utilization. Normal launch scripts
never start a rental watchdog or change instance state.

## Stages, overrides and resume

```bash
python -m scripts.main.train --stage sft
python -m scripts.main.train --stage rl
python -m scripts.main.train --set output_dir=logs/my-run \
  --set batch_sizes.rl=8 --set logprob_batch_size=4
python -m scripts.main.train --stage rl \
  --resume logs/my-run/rl/checkpoints/interrupted.ckpt --trust-checkpoint
```

Fresh runs refuse to overwrite checkpoints/results or an existing SFT export.
Keep the tuned SFT foundation in `sft_export_dir` for RL checkpoint restoration.
Resume requires compatible model, adapter, optimizer and data settings.
The exact online-augmentation stream is not restored across worker restarts.
Only load checkpoints you trust. After a killed process, confirm it exited
before removing its `.launcher.lock`. For a graceful checkpointed stop, create `logs/sac-a100-production/STOP`
(or the STOP file under your configured output directory). The callback saves
`interrupted.ckpt` between batches. Remove that STOP file before resuming.

## Evaluation

```bash
python -m scripts.main.eval --stage rl --split val \
  --checkpoint logs/sac-a100-production/rl/checkpoints/handoff.ckpt \
  --trust-checkpoint --save-predictions --output logs/manual-validation
python -m scripts.main.eval --stage rl --split test \
  --checkpoint logs/sac-a100-production/rl/checkpoints/handoff.ckpt \
  --trust-checkpoint --save-predictions --output logs/final-test
```

Both directions are evaluated with clean held-out examples. BLEU and chrF++
use 0-100; COMET is reported raw. SAC and fused reward are reported separately.
Changing curriculum alpha can change reward without a quality improvement.
Do not use the official test set to choose checkpoints.

## Verify before a long run

```bash
python -m pytest tests/grpo -q
python -m scripts.verify.pipeline --config configs/runs/a100.yaml \
  --synthetic --output logs/five-step-probe --steps 5
```

This creates isolated synthetic data and fresh output/export directories,
removes the production step schedule, and sets the fusion horizon to five.
It refuses to overwrite an existing probe directory. It requires GPU and API
access, verifies exactly five updates in each stage, and leaves production
outputs untouched. A long run is a separate command after the probe passes.

Build a source-only upload with `python -m scripts.tools.package`. Datasets,
credentials, checkpoints, caches, archives and Git history are excluded.
The dataset must be uploaded separately. No training is launched by packaging.
