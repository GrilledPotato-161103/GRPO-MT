# Full PhoMT training and conditional reward controls

Prepared from `data/full/PhoMT_training.csv` and `data/full/PhoMT_test.csv`.
Run commands from `GRPO-MT`, using the existing Conda environment at `../venv`.

## Prepared data

| Split | Parent pairs | EN↔VI directional examples |
|---|---:|---:|
| SFT | 2,373,127 | 4,746,254 |
| GRPO | 593,282 | 1,186,564 |
| Validation | 2,000 | 4,000 |
| Official test, unchanged | 19,151 | 38,302 |

The input training CSV has 2,977,999 rows. Preparation removed 1,439 normalized
duplicate pairs and excluded 8,151 pairs in components sharing either language
with the official test set. Connected components sharing normalized English or
Vietnamese stay together, including transitive overlaps. Validation is reserved
first; the remaining training pairs split 80/20 between SFT and RL. Mirroring
directions and online augmentation happen afterward.

Original CSVs remain intact. Hashes, counts, grouping rules, and the small
group-boundary rounding difference are recorded in
`data/full_prepared/manifest.json`. Preparation refuses an existing output
directory rather than overwriting a previous split.

```powershell
# Already completed in this workspace; use a new --output to prepare another version.
../venv/python.exe -m scripts.main.prepare

# Reuse prepared CSVs and create/reuse disk-backed indexes only.
../venv/python.exe -m scripts.main.prepare --index-only

# Inspect clean, uncapped token lengths using only the public tokenizer.
../venv/python.exe -m scripts.audit_translation_lengths --local-only
```

CSV indexes handle quoted newlines and reopen memory maps per worker. Workers
do not receive a Python list containing millions of translation strings. Token
length caches include the data fingerprint, direction settings, prompt, caps,
tokenizer revision and tokenizer state; they are reused without consuming the
augmentation RNG. Treat prepared files as immutable during training.

The local token audit sampled 10,000 pairs per large split and all 2,000
validation pairs. The 256 source / 128 target caps cover more than 99.7% of the
sampled clean targets. This does not bound generated lengths or augmented/NFD
inputs; cap-hit and source-truncation metrics remain necessary.

## Full-data configuration

Use [`scripts/main/full.yaml`](../scripts/main/full.yaml). It preserves rank-8
q/k/v/o LoRA, BF16, gradient checkpointing, length buckets, four candidates,
two updates per rollout, logprob batches up to eight, and 16 persistent data
workers. The frozen foundation downloads through Transformers. COMETKiwi runs
on the same GPU. W&B project is `Machine Translation`.

This profile uses fixed KL beta **0.01**, with a **0.04** matched control
available as an override. It has no budget deadline or rental watchdog. It runs
one epoch per stage unless overridden. The official test is evaluated only
through the explicit evaluation command, not automatically after RL pilots.

```bash
# No models or credentials needed for configuration preview.
python -m scripts.main.train --config scripts/main/full.yaml --dry-run

# Full SFT → export → GRPO pipeline; this is a manual command, not run by preparation.
python -m scripts.main.train --config scripts/main/full.yaml

# Independent clean validation of your trusted checkpoint.
python -m scripts.main.eval --config scripts/main/full.yaml \
  --checkpoint /path/to/rl.ckpt --trust-checkpoint --stage rl --split val

# Final test after selecting a checkpoint on development data.
python -m scripts.main.eval --config scripts/main/full.yaml \
  --checkpoint /path/to/selected.ckpt --trust-checkpoint --stage rl --split test
```

The complete dataset is much larger than the earlier 950-pair RL pool. At the
maximum configured batch sizes, one pass is approximately 197,761 SFT updates
and 148,322 GRPO updates. These are work counts, not wall-time or cost estimates;
packing limits, evaluation and actual sequence lengths affect runtime. Benchmark
the actual data/model before scheduling a complete pass.

## Matched KL experiment

Complete SFT once, then use the same merged SFT export and prepared splits in
both RL arms. Keep seed, learning rate, reward, sampling, caps, LoRA and update
budget equal. Do not attribute differences from the old small-data run solely
to KL: the dataset and SFT base changed too.

```bash
python -m scripts.main.train --config scripts/main/full.yaml --stage rl \
  --set output_dir=logs/full-beta004 \
  --set sft_export_dir=logs/full-phomt/sft_export \
  --set beta=0.04 --set trainer.max_steps=200

python -m scripts.main.train --config scripts/main/full.yaml --stage rl \
  --set output_dir=logs/full-beta001 \
  --set sft_export_dir=logs/full-phomt/sft_export \
  --set beta=0.01 --set trainer.max_steps=200
```

Two hundred updates can screen gross failures; they do not establish convergence.
The manual optimization loop honors an odd `trainer.max_steps` exactly even
when each rollout ordinarily has two updates.

## Adaptive reference KL

`kl_control.enabled` defaults to false. Enabling it requires an explicitly
calibrated positive `kl_control.target`; leaving the target null fails preflight.
The controller uses an EMA of the sampled reference-divergence proxy, bounded
coefficients, and a configurable update interval. It does not control
`approx_kl_to_old`, and its estimate is not an exact full-vocabulary KL.

Optional `quality_floors` map validation metric names to required minima, for
example `val/en_vi/chrf` and `val/vi_en/chrf`. If provided, beta cannot decrease
until all protected metrics exist and meet their floors. Strengthening the
penalty remains permitted. Derive floors from the new SFT baseline; do not reuse
scores from the earlier 250-pair development split as calibrated thresholds.

Controller coefficient, EMA, observation count and relaxation status are saved
with the Lightning checkpoint. Resume requires the same controller settings;
start an explicit new ablation to change them. Older checkpoints without
controller state use the newly supplied configuration. The SFT reference is
never silently re-anchored. On-step metrics include the coefficient used and
the next coefficient.

## Structured criterion feedback

`reward_harness.enabled` defaults to false so the first KL comparison remains
matched and makes no LLM-feedback requests. Enabling the harness uses the
existing API/local backend interface:

- API default: OpenRouter, configured model alias `deepseek_v4.1_flash_1017`,
  `CLAUDE_AUTH_KEY` read from the environment; no token belongs in YAML.
- Self-hosted: replace `reward_harness.backend` with a mapping containing
  `kind: self_hosted`, `model_name_or_path`, `device`, `dtype`, and the other
  `TransformersChatBackend` options. Choose an explicit suitable judge model.
- Cache-only: set `cache_only: true` and point `cache_dir` at validated existing
  annotations. A missing entry raises an error; it never fabricates a score.

Feedback has a versioned array of named criteria with applicability, nullable
scores, nullable confidence, severity and evidence. Parsing rejects malformed
values and unsupported errors. Cache identity includes source, candidate,
direction, provider/model and prompt version. No API key is cached. Raw judge
confidence is not an EN–VI calibration result.

With `calibrated: false`, the enabled harness is in **shadow mode**: it records
criterion evidence and preserves the original reward and group weight. API
calls still cost money in shadow mode. Calibrate judgments and confidence on
blinded bilingual examples before setting `calibrated: true`.

When calibrated, evidenced major/critical semantic, numerical, or language
failures enter a lower reward tier. Fluent surface bonuses cannot compensate
for that tier. Reliable, high-fidelity candidates can receive a relaxed lexical
preference. Unknown semantic categories are logged and lower the group's
policy-advantage weight after normalization. The KL penalty remains active.
Groups where every candidate fails have zero policy-advantage weight and enter
`diagnostics/repair_queue.jsonl` by source hash/direction for verified correction.
No correction is synthesized or accepted automatically.

Validation continues to report the fixed COMETKiwi/BLEU audit reward and corpus
BLEU/chrF++ independently of the conditional training objective, without making
criterion-judge API requests. The official test remains held out.

Implemented now: typed criterion feedback, caching, conditional tiers,
post-normalization confidence, repair routing, and checkpointed KL control.
Further work: human calibration, typed entity/unit checkers, explicit glossary
and format contracts, offline multi-pair preference training, and error-span
rewards. These are not silently enabled by the profile.

## Five-step A100 verification

`python -m scripts.verify_full_pipeline --config /workspace/pilot.yaml --steps 5`
checks five actual optimizer calls in **each** stage, frozen-backbone LoRA,
finite metrics, both evaluation directions, real COMETKiwi/BLEU/chrF++ outputs,
checkpoints and per-update GRPO diagnostics. `five_step_verification.json`
contains phase timings and validation metrics.

The pilot uses 128 SFT and 32 RL pairs sampled from their respective new full
splits. Its 16-pair validation and 16-pair functionality holdout come from
distinct development groups. The official test and complete raw corpus are not
uploaded. This test measures software functionality, not trained translation
quality or full-corpus convergence. No full-data training begins afterward.
