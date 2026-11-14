# Stopped SFT checkpoint evaluation — November 10, 2026

Training stopped at user request at **SFT optimizer step 24,393**, 12.33% of the configured epoch. GRPO did not start. The STOP file remains present. The A100 rental remains running under user control; no automatic stop or destruction was configured.

The checkpoint was evaluated on all **2,000 held-out validation pairs in both directions**, without augmentation. This is validation, not the official 19,151-pair test. Generation retains the 256-source-token and 128-generated-token limits. COMET is `Unbabel/wmt22-cometkiwi-da` (reference-free COMETKiwi); BLEU and chrF++ are corpus scores on a 0–100 scale.

| Direction | BLEU | chrF++ | COMETKiwi |
|---|---:|---:|---:|
| EN → VI | 31.6701 | 49.4774 | 0.821120 |
| VI → EN | 33.9016 | 54.8523 | 0.833231 |

| SFT step | Macro BLEU | Macro chrF++ | Macro COMETKiwi | Reward |
|---|---:|---:|---:|---:|
| 10,000 | 32.1914 | 51.6640 | 0.825441 | 0.661408 |
| 20,000 | 32.0953 | 51.5188 | 0.823948 | 0.659993 |
| 24,393 | 32.7859 | 52.1649 | 0.827175 | 0.663764 |

Validation scores had already declined slightly between 10,000 and 20,000 updates. These observations support stopping further SFT; small differences alone do not establish statistical significance. Logged training-loss windows fluctuate around 1.26–1.31 after the early improvement and are not themselves a held-out convergence test.

The fixed reward is 70% raw COMETKiwi plus 30% sentence BLEU divided by 100; corpus BLEU is reported separately. The LLM feedback harness is disabled.

Remote checkpoint: `/workspace/grpo-mt/logs/full-phomt-20261110/sft/checkpoints/interrupted.ckpt`. Local backup: `logs/a100_full_test_20261110/stopped-checkpoint/sft-step-24393.ckpt`. Its SHA256 and pinned foundation revision are in the adjacent `manifest.json`. All 224 saved LoRA tensors are finite, and optimizer state is preserved.

Remote evaluation: `/workspace/grpo-mt/logs/full-phomt-20261110/evaluation/manual-stop-val-20261110`. Process exited successfully; metrics and timings remain on the rental, with scalar reports copied locally.

[W&B evaluation run](https://wandb.ai/25025069-vietnam-national-university-hanoi/Machine%20Translation/runs/11uzyggq).
