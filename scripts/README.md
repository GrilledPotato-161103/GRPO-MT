# Commands

Run Python commands from the repository root. `python -m scripts.main.train`
and `python -m scripts.main.eval` remain the stable entry points. Editable
settings live in `configs/runs/`, not beside the Python files.

| Command | Purpose |
|---|---|
| `python -m scripts.main.train --help` | SFT, RL, complete pipeline and resume |
| `python -m scripts.main.eval --help` | Validation/test and prediction export |
| `python -m scripts.main.prepare --help` | Full PhoMT preparation and indexes |
| `python -m scripts.main.cycles --help` | Optional sampled SFT/RL cycles |
| `python -m scripts.data.prepare_stages --help` | Split an existing training/dev/test corpus |
| `python -m scripts.data.filter_token_pairs --help` | Select complete pairs fitting both language/token budgets |
| `python -m scripts.verify.pipeline --help` | Bounded real-model SFT/RL verification |
| `python -m scripts.verify.gpu --help` | Synthetic GPU smoke checks |
| `python -m scripts.verify.comet --help` | Real COMETKiwi inference check |
| `python -m scripts.benchmarks.grpo --help` | Candidate graph batching benchmark |
| `python -m scripts.benchmarks.stages --help` | SFT/GRPO timing comparison |
| `python -m scripts.benchmarks.memory --help` | Adapter VRAM measurement |
| `python -m scripts.tools.export_sft --help` | Reconstruct the merged SFT foundation |
| `python -m scripts.tools.package --output logs/grpo-mt-source.tar.gz` | Portable source bundle |
| `python -m scripts.tools.clean` | Preview disposable caches |
| `python -m scripts.tools.clean --apply` | Remove only those caches |

Additional helpers stay in their category: length/scoring audits and tuning
under `benchmarks`, dataset conversion under `data`, snapshots and W&B relay
under `rental`. `rental.supervisor` is the historical optional budget supervisor;
it is never started by the normal workflow. Old instance-specific launchers are
archived and unsupported. See [execution](../docs/execution.md) for setup,
tmux, resume, credentials and graceful stop commands.
