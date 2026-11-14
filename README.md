# GRPO-MT

Machine translation with Hydra, PyTorch Lightning and Hugging Face Transformers.
The new [NLLB correction profile](docs/correction_training.md) pairs failed
translations with judge-verified targets, with measured GPU memory and a dated
rental/API cost plan. The existing profiles below remain reproducible.
The [larger NLLB trials](docs/nllb_scale_trials.md) cover distilled 1.3B and dense
3.3B with model-specific memory limits.

The primary workflow is **2,000 supervised updates -> 10,000 GRPO updates** on
separate EN-VI partitions, with frozen-backbone LoRA and bilingual evaluation.

The current reward combines **COMETKiwi** and **OpenRouter SAC** using a dynamic
sigmoid fusion. EN->VI and VI->EN candidates decode together; symmetric endpoint
matching replaces sequential back-translation. API feedback defaults to
`openai/gpt-6-luna` through `ANTHROPIC_AUTH_TOKEN`. A self-hosted judge is optional.

## Run

Use the existing Conda environment at `../venv`:

```bash
conda activate ../venv
python -m pip install -r requirements.txt
python -m scripts.main.train --dry-run
python -m scripts.main.train --config configs/runs/a100.yaml
```

Keep `ANTHROPIC_AUTH_TOKEN`, `HUGGING_FACE_API_KEY` and `WANDB_API_KEY` in the
process environment; see [.env.example](.env.example). W&B project:
**Machine Translation**. COMETKiwi requires accepted model conditions and an
HF token with access to `Unbabel/wmt22-cometkiwi-da`.

On a prepared Linux GPU rental:

```bash
bash scripts/rental/setup.sh
bash scripts/rental/start.sh --config configs/runs/a100.yaml
tmux attach -t mt-train
```

Detach with **Ctrl-b, d**. The scripts never rent, stop or destroy an instance.
No budget watchdog is enabled by default. A new run downloads its foundation
through Hugging Face; checkpoints are used only when explicitly resuming.

## Evaluate

```bash
python -m scripts.main.eval --config configs/runs/a100.yaml \
  --checkpoint logs/sac-a100-production/rl/checkpoints/handoff.ckpt \
  --stage rl --split test --trust-checkpoint --save-predictions
```

Evaluation reports corpus BLEU, chrF++, COMETKiwi, SAC, and fused reward in both
directions. Prediction exports contain source, reference, candidate and scores.
The official test set is reserved for explicit final evaluation.

## Project map

| Path | Purpose |
|---|---|
| `src/` | Lightning modules, feedback, rewards, data loading and optimizers |
| `configs/runs/` | Launcher profiles: `a100`, `rtx4090`, `sac`, `full`, `small`, `cycles` |
| `configs/` | Hydra model, data, trainer, optimizer and callback groups |
| `scripts/main/` | Train, evaluate, prepare and optional cyclic training |
| `scripts/data/` | Leakage-safe data preparation |
| `scripts/benchmarks/` | Timing, candidate batching and memory tools |
| `scripts/rental/` | Manual setup, tmux launch, monitoring and snapshots |
| `scripts/verify/` | Synthetic assets and bounded pipeline/GPU verification |
| `scripts/tools/` | Export, source packaging, checkpoint migration, cache cleanup |
| `tests/grpo/` | Offline tests using local tiny models and mocked feedback |
| `docs/` | Current manuals and research notes |
| `archive/` | Historical launchers, reports and the retired MNIST template |
| `data/`, `logs/` | User datasets, checkpoints and results; excluded from source bundles |

Start with the [execution manual](docs/execution.md), [data guide](docs/data.md),
[reward/training details](docs/training.md), and [development guide](docs/development.md).
The [procedure handoff](docs/PROCEDURE_HANDOFF.md) records current instructions,
effective settings and the last verified deployment state.
The [script index](scripts/README.md) lists maintenance commands.
[Migration notes](docs/migration.md) map commands from the old flat scripts directory.

## Verification status

The legacy round-trip objective passed a real Qwen-7B/COMET/live-SAC 4090 pilot.
The symmetric fusion objective has local optimizer/resume tests; its real-GPU
verification and production start remain unverified. The last observed A100
worker was downloading weights. Consult the procedure handoff for the snapshot;
historical reports are evidence, not current status.
