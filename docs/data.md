# Data

The raw full PhoMT CSVs live in `data/full/`. The existing preparation produces:

| Partition | Parallel pairs |
|---|---:|
| SFT | 2,373,127 |
| RL | 593,282 |
| Validation | 2,000 |
| Official test | 19,151 |

The 80/20 SFT/RL split is applied before bilingual expansion and augmentation.
Transitive groups sharing normalized English or Vietnamese text stay together.
Training groups that overlap the official test are excluded. SHA-256 hashes
and counts are recorded in `data/full_prepared/manifest.json`; the launcher
checks those hashes before training. The official test is not rewritten.

Reuse the existing prepared corpus and build/reuse disk-backed indexes:

```bash
python -m scripts.main.prepare --index-only
```

For a new copy of the raw corpus, choose a fresh destination:

```bash
python -m scripts.main.prepare \
  --train-file data/full/PhoMT_training.csv \
  --test-file data/full/PhoMT_test.csv --output data/full_prepared
```

The preparer refuses to overwrite prepared output. For another corpus with
explicit development/test files, use `python -m scripts.data.prepare_stages --help`.
Paths in run YAML files are resolved relative to the repository.

Both EN->VI and VI->EN are views of each parallel pair. The SAC profile uses
`paired_batches=true`: both views stay in the same batch through shuffling,
length bucketing and collation. Batch sizes must be even. Augmentation draws
concatenation once per pair (10%, capped at 768 characters per side), then adds
independent whitespace and Unicode-normalization noise to inputs (20% each).
Clean originals remain available for scoring. Concatenated partners come only
from the current training partition. Validation/test are clean.

This workflow does not generate LLM paraphrases. Symmetric endpoint scoring
uses the existing aligned pair and one bilingual generation batch; it does not
create sequential cycle translations or introduce held-out references into
training. `data/` is ignored except for three synthetic example JSONL fixtures.
