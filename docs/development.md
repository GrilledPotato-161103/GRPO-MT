# Development

Install development tools into the existing environment:

```bash
python -m pip install -e '.[dev,lora,comet,logging]'
python -m pytest tests/grpo -q
python -m ruff check src scripts tests/grpo
python -m mypy
```

Tests build tiny local Transformers models and mock API feedback. They do not
rent GPUs, download foundation weights, send user text to APIs, or establish
real-model throughput. GPU checks are explicit commands in `scripts/verify/`.
CI uses the same offline suite. See `docs/training.md` for the distinction
between local fusion verification and the earlier real-model round-trip pilot.

`src/models/` contains SFT, GRPO, cyclic and self-play Lightning modules.
`src/models/components/` contains objectives, rewards, adapters and optimizers.
`src/data/` owns paired translation data, augmentation, length sampling and
memory-mapped indexes. `src/feedback/` provides API/local judges and SAC parsing.
`src/utils/` holds timing, telemetry and generic Lightning/Hydra utilities.
Launcher code lives in `scripts/main/`; reusable maintenance commands are
organized by purpose under `scripts/`.

Config files in `configs/runs/` use relative `base_config` inheritance and feed
the Hydra groups in the rest of `configs/`. `a100.yaml` and `rtx4090.yaml`
inherit `sac.yaml`, which selects the current objective over shared `full.yaml`
settings. `small.yaml` and `cycles.yaml` preserve earlier experiments.

```bash
python -m scripts.tools.clean          # preview
python -m scripts.tools.clean --apply  # caches and coverage only
python -m scripts.tools.package --output logs/grpo-mt-source.tar.gz
```

Cleanup validates every deletion target stays within the project. It never
walks `data/`, `logs/`, `archive/` or the Conda environment. Source packaging
includes code/configs, manuals, tests and synthetic fixtures; it excludes user
data, environment files containing keys, model weights and experiment results.
Historical material is preserved under `archive/`, outside active test discovery
and installation. Do not rewrite old checkpoints just to change script paths.
