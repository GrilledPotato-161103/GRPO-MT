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

## Run Qwen or NLLB

Run the following Bash commands from the `GRPO-MT` directory. Locally, use the
existing Conda environment at `../venv`; on a Linux GPU rental, use its prepared
PyTorch environment and run `bash scripts/rental/setup.sh` instead.

```bash
conda activate ../venv
python -m pip install -r requirements.txt
```

Keep `ANTHROPIC_AUTH_TOKEN`, `HUGGING_FACE_API_KEY` and `WANDB_API_KEY` in the
process environment; see [.env.example](.env.example). W&B project:
**Machine Translation**. COMETKiwi requires accepted model conditions and an
HF token with access to `Unbabel/wmt22-cometkiwi-da`. `ANTHROPIC_AUTH_TOKEN` must
contain an OpenRouter API key for these profiles.

### Prepare PhoMT once

Place the original CSVs, with `en,vi` columns, in `data/full/`. For a fresh
workspace, prepare the disjoint 80/20 SFT/RL partitions:

```bash
python -m scripts.main.prepare \
  --train-file data/full/PhoMT_training.csv \
  --test-file data/full/PhoMT_test.csv --output data/full_prepared
```

If `data/full_prepared/manifest.json` already exists, reuse it with
`python -m scripts.main.prepare --index-only` instead. Preparation refuses to
overwrite existing splits. Validation is held out before training; the official
test remains separate.

### Qwen family

For Qwen3-1.7B with native non-thinking chat prompts and complete pairs capped at
256 content tokens on each side:

```bash
python -m scripts.data.filter_token_pairs --format chat \
  --tokenizer Qwen/Qwen3-1.7B --model-name Qwen/Qwen3-1.7B \
  --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
  --input data/full_prepared --test-file data/full/PhoMT_test.csv \
  --output data/token256_qwen3

PROFILE=configs/runs/qwen3_1p7b_4090_precise.yaml
RUN_DIR=logs/qwen3-1p7b-manual
```

Skip filtering when its matching manifest already exists. This profile measures
the instruction/chat overhead in addition to the 256 content tokens, disables
online augmentation, and uses AdamW with an RL learning rate of `2e-5` and
ReduceLROnPlateau on validation COMETKiwi.

For the existing Qwen2.5-7B-Instruct LoRA profile, use the unfiltered prepared
splits and select these variables instead:

```bash
PROFILE=configs/runs/rtx4090.yaml
RUN_DIR=logs/qwen25-7b-manual
```

This 7B profile inherits its model and pinned revision from
`configs/runs/full.yaml` and its SAC reward from `sac.yaml`. Its settings differ
from the Qwen3 precise-prompt experiment. For another Qwen model, set both
`model_name` and `policy_revision` in a copied profile; rebuild any token-filter
manifest with that model's tokenizer and matching prompt. Do not reuse adapters
or optimizer state across model sizes.

### NLLB family

NLLB uses an encoder-decoder policy with native `eng_Latn`/`vie_Latn` language
controls and the raw source text, rather than a chat instruction. First prepare
the shared bounded NLLB data:

```bash
python -m scripts.data.filter_token_pairs --format nllb \
  --tokenizer facebook/nllb-200-distilled-600M \
  --model-name facebook/nllb-200-distilled-600M \
  --revision f8d333a098d19b4fd9a8b18f94170487ad3f821d \
  --input data/full_prepared --test-file data/full/PhoMT_test.csv \
  --output data/token256_nllb
```

For **NLLB-1.3B**, create its compatibility manifest and select its profile:

```bash
python -m scripts.data.filter_token_pairs --format nllb \
  --tokenizer facebook/nllb-200-distilled-1.3B \
  --model-name facebook/nllb-200-distilled-1.3B \
  --revision 7be3e24664b38ce1cac29b8aeed6911aa0cf0576 \
  --reuse-manifest data/token256_nllb/manifest.json \
  --output data/token256_nllb_1p3b

PROFILE=configs/runs/nllb_1p3b_correction.yaml
RUN_DIR=logs/nllb-1p3b-manual
```

Manifest reuse verifies tokenizer compatibility and CSV hashes; the 1.3B profile
continues to read the shared `data/token256_nllb/` CSVs. Skip these preparation
commands when the corresponding verified manifests already exist.

Other NLLB sizes use the same launcher:

| Model | Profile | Preparation |
|---|---|---|
| `facebook/nllb-200-distilled-600M` | `configs/runs/nllb_600m_correction.yaml` | Shared bounded data above |
| `facebook/nllb-200-distilled-1.3B` | `configs/runs/nllb_1p3b_correction.yaml` | Compatibility manifest above |
| `facebook/nllb-200-3.3B` | `configs/runs/nllb_3p3b_correction.yaml` | [3.3B manifest command](docs/nllb_scale_trials.md#commands) |

Set `PROFILE` and a fresh `RUN_DIR` for the size you choose. These profiles use
rank-8 all-linear LoRA, 2,000 SFT updates followed by 10,000 hybrid GRPO/correction
updates, and an RL learning rate of `2e-5`. They are experimental correction
profiles, with a different reward/routing setup from the Qwen SAC-fusion profile.
See [the objective](docs/correction_training.md) and
[model-specific memory limits](docs/nllb_scale_trials.md) before changing batches.

### Train the selected profile

After selecting `PROFILE` and `RUN_DIR` in one of the sections above, keep the
same arguments for training, resume and evaluation:

```bash
RUN_ARGS=(--config "$PROFILE"
  --set "output_dir=$RUN_DIR"
  --set hf_home=logs/assets/huggingface
  --set comet_checkpoint=logs/assets/cometkiwi/checkpoints/model.ckpt)

# Resolve both stages without loading weights or making feedback requests.
python -m scripts.main.train "${RUN_ARGS[@]}" --dry-run

# Fresh foundation from Hugging Face -> SFT export -> RL.
python -m scripts.main.train "${RUN_ARGS[@]}" --stage pipeline
```

Use a new output directory for every fresh experiment. The launcher refuses to
overwrite existing results or an SFT export. The precise Qwen3 and NLLB profiles
need their matching token-filter manifest even for `--dry-run`.

On Linux, run the following **instead of** the foreground training command to
start a detached tmux session with Rich progress:

```bash
bash scripts/rental/start.sh "${RUN_ARGS[@]}" --stage pipeline
tmux attach -t mt-train
```

Detach with **Ctrl-b, d**. Console logs are saved under `logs/sessions/`; W&B
uses project **Machine Translation**. The scripts never rent, stop or destroy an
instance. No budget watchdog is enabled by default.

To run the stages separately, use `--stage sft` followed by `--stage rl` with
the same `RUN_ARGS`. RL needs the merged foundation in `$RUN_DIR/sft_export`.
To resume an interrupted RL stage, replace the path below with your checkpoint:

```bash
python -m scripts.main.train "${RUN_ARGS[@]}" --stage rl \
  --resume "$RUN_DIR/rl/checkpoints/interrupted.ckpt" --trust-checkpoint
```

Keep the SFT export and matching model/data/adapter settings when resuming.
For a graceful stop, create `$RUN_DIR/STOP`; remove it before resuming.

## Evaluate

```bash
# Use the same PROFILE, RUN_DIR and RUN_ARGS as the training run.
python -m scripts.main.eval "${RUN_ARGS[@]}" \
  --checkpoint "$RUN_DIR/rl/checkpoints/handoff.ckpt" \
  --stage rl --split test --trust-checkpoint --save-predictions \
  --output "$RUN_DIR/final-test"
```

Evaluation reports corpus BLEU, chrF++, COMETKiwi, SAC, and fused reward in both
directions. Prediction exports contain source, reference, candidate and scores.
The official test set is reserved for explicit final evaluation. The bounded
profiles select `test_bounded.csv` and limit evaluation to 200 batches by default.
For every official test pair in both directions, append
`--set data_files.test=data/full/PhoMT_test.csv --set trainer.limit_test_batches=1.0`.
The configured generation caps still apply. To evaluate SFT instead, use
`--stage sft` and a checkpoint from `$RUN_DIR/sft/checkpoints/`. Select checkpoints
on validation (`--split val`), not the official test.

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
