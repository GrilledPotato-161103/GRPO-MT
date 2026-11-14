# Final evaluation and rental cleanup ? November 11, 2026

Vast.ai instance **60031289** was destroyed after the checkpoints, all metric logs and all test predictions were downloaded and verified. The destroy command returned success, and the instance was absent from the subsequent Vast.ai instance list.

Training completed all **10 sampled SFT/GRPO cycles**, adding 10,000 SFT and 2,000 RL optimizer updates to the starting SFT checkpoint at step 24,393: **36,393 total updates**. The backbone and fixed reference stayed frozen.

## Official test

The untouched PhoMT test has **19,151 pairs**, evaluated in both EN?VI and VI?EN: **38,302 translations**. Evaluation used greedy decoding, source cap 256, generated-token cap 128, batch size 128, and no augmentation. COMET denotes reference-free `Unbabel/wmt22-cometkiwi-da`; BLEU and chrF++ use their 0?100 corpus scales.

| Direction | BLEU | chrF++ | COMETKiwi |
|---|---:|---:|---:|
| EN ? VI | 33.8769 | 51.9334 | 0.829187 |
| VI ? EN | 33.9106 | 55.9312 | 0.843326 |
| Macro | 33.8937 | 53.9323 | 0.836257 |

All prediction indices, directions, sources and references were checked against the original test pairs. CSV and JSONL predictions match exactly. Test-data SHA256: `c052a087e61fcbc5a5324a99a3adb086417cf54cbcff5790e1a499311816cab9`.

## Downloaded artifacts

All paths below are relative to the project root. Every archive checksum and every archived file checksum was verified. All 224 LoRA tensors in each saved checkpoint are finite, optimizer state is present, and the standalone adapter exactly matches the latest checkpoint. Its tokenizer also loads locally.

- Predictions: `logs/a100_close_20261111/predictions/test_predictions.csv` and `test_predictions.jsonl`. Rows include example index, direction, source, reference, prediction, reward and COMET. The CSV is UTF-8 with BOM.
- Latest checkpoint: `logs/a100_close_20261111/artifacts/cyclic-checkpoints-20261111/checkpoints/latest.ckpt`.
- The same folder includes `last_sft.ckpt`, `best_chrf.ckpt` (cycle 8 RL, selected on validation), and `reference.ckpt` (initial SFT, step 24,393).
- Standalone inference adapter and tokenizer: `logs/a100_close_20261111/artifacts/cyclic-checkpoints-20261111/adapter/`.
- Complete metric archive: `logs/a100_close_20261111/artifacts/all-metric-logs-20261111.tar.gz`: **542 metric/log files**, plus source/configuration context; **717 files verified** overall. Includes earlier SFT, every cycle phase, validation, official test, raw timing events, W&B metric histories and console output. Model weights, datasets, caches and non-metric debug logs are excluded.
- Readable results: `logs/a100_close_20261111/official_test_metrics.json`, `cycle_metrics.csv`, `cycle_metric_effects.json` and `cycle_metrics.png`.

The original Hugging Face foundation is reconstructed from `Qwen/Qwen2.5-7B-Instruct`, revision `a09a35458c702b33eeacc393d103063234e8bc28`; its dense weights are not duplicated in the checkpoint archive. Raw checkpoints retain their historical rental paths. When loading a cyclic checkpoint locally, override `reference_checkpoint` with the local `reference.ckpt` path. The PEFT inference adapter needs no frozen-reference file.

```python
from src.models.cyclic_module import CyclicTranslationModule

model = CyclicTranslationModule.load_from_checkpoint(
    "logs/a100_close_20261111/artifacts/cyclic-checkpoints-20261111/checkpoints/latest.ckpt",
    reference_checkpoint="logs/a100_close_20261111/artifacts/cyclic-checkpoints-20261111/checkpoints/reference.ckpt",
    weights_only=False,
)
```

Standalone evaluation now supports `--stage cyclic`, `--reference-checkpoint` and `--save-predictions`. The export was tested for ordering, Unicode/reference preservation and repeated evaluation without duplicate rows; the actual full GPU test verified all 38,302 exports.

## Validation and phase effects

The full 2,000-pair validation split was also evaluated before the official test. Comparisons below use the same full validation set; the per-cycle 1,000-pair subset is separate.

| Checkpoint | Macro BLEU | Macro chrF++ | Macro COMETKiwi |
|---|---:|---:|---:|
| Starting SFT, 24,393 updates | 32.7859 | 52.1649 | 0.827175 |
| Final cycle, 36,393 updates | 32.4816 | 52.7886 | 0.834192 |

Across the fixed 1,000-pair cycle evaluations, RL improved chrF++ and COMET in all 10 cycles. Mean RL-minus-SFT changes: **+0.9312 chrF++, +0.007147 COMET, ?0.0903 BLEU**; BLEU improved in 4/10 cycles. These are observed phase effects, not a controlled attribution or significance test.

[Cycle comparison chart](../logs/a100_close_20261111/cycle_metrics.png).

Cleanup evidence: `logs/a100_close_20261111/pre-destruction-verification.json` and `destruction-verification.json`. The user?s SSH key was preserved.
