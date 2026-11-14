# Command migration

The stable train/eval Python entry points remain `scripts.main.train` and
`scripts.main.eval`. Launcher YAML files moved to `configs/runs/`.
The default now selects the current A100 COMET/SAC fusion pipeline, with no
budget watchdog. The original flat-directory commands are intentionally not
kept as duplicate wrapper files.

| Previous path | Current path |
|---|---|
| `scripts/train_bilingual_rental.py` | `scripts/main/pipeline.py` |
| `scripts/prepare_full_phomt.py` | `scripts/data/prepare_full.py` |
| `scripts/prepare_phomt.py` | `scripts/data/prepare_phomt.py` |
| `scripts/prepare_training_stages.py` | `scripts/data/prepare_stages.py` |
| `scripts/audit_scoring_precision.py` | `scripts/benchmarks/scoring_precision.py` |
| `scripts/audit_translation_lengths.py` | `scripts/benchmarks/lengths.py` |
| `scripts/benchmark_grpo_batching.py` | `scripts/benchmarks/grpo.py` |
| `scripts/benchmark_self_play.py` | `scripts/benchmarks/self_play.py` |
| `scripts/benchmark_translation_stages.py` | `scripts/benchmarks/stages.py` |
| `scripts/estimate_self_play.py` | `scripts/benchmarks/estimate.py` |
| `scripts/profile_adapter_memory.py` | `scripts/benchmarks/memory.py` |
| `scripts/summarize_stage_profiles.py` | `scripts/benchmarks/summarize.py` |
| `scripts/tune_batch_size.py` | `scripts/benchmarks/batch_size.py` |
| `scripts/tune_token_budget.py` | `scripts/benchmarks/token_budget.py` |
| `scripts/export_sft.py` | `scripts/tools/export_sft.py` |
| `scripts/rebatch_checkpoint.py` | `scripts/tools/rebatch_checkpoint.py` |
| `scripts/package_gpu_test.py` | `scripts/tools/package.py` |
| `scripts/create_smoke_assets.py` | `scripts/verify/assets.py` |
| `scripts/gpu_smoke.py` | `scripts/verify/gpu.py` |
| `scripts/verify_cometkiwi.py` | `scripts/verify/comet.py` |
| `scripts/verify_full_pipeline.py` | `scripts/verify/pipeline.py` |
| `scripts/rental_snapshot.py` | `scripts/rental/snapshot.py` |
| `scripts/relay_wandb.py` | `scripts/rental/wandb_relay.py` |
| `scripts/watch_vast_training.py` | `scripts/rental/supervisor.py` |
| `scripts/monitor_rental.ps1` | `scripts/rental/monitor.ps1` |
| `scripts/main/config.yaml` | `configs/runs/small.yaml` |
| `scripts/main/full.yaml` | `configs/runs/full.yaml` |
| `scripts/main/cycles.yaml` | `configs/runs/cycles.yaml` |
| `scripts/main/sac.yaml` | `configs/runs/sac.yaml` |
| `scripts/main/sac_a100.yaml` | `configs/runs/a100.yaml` |
| `scripts/main/sac_4090.yaml` | `configs/runs/rtx4090.yaml` |

Use dotted module names for Python files, for example
`python -m scripts.data.prepare_full` or `python -m scripts.benchmarks.grpo`.
Instance-specific old launchers and all dated handoff reports are archived.
Their old endpoints are not active rental configuration.

The full file map, including original SHA-256 checksums, is in
[archive/MIGRATION.json](../archive/MIGRATION.json). Existing data, model
weights, checkpoints, logs and user secrets were left in place.
