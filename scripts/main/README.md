# Train and evaluate

```bash
python -m scripts.main.train --config configs/runs/a100.yaml --dry-run
python -m scripts.main.train --config configs/runs/a100.yaml
python -m scripts.main.eval --config configs/runs/a100.yaml \
  --checkpoint logs/sac-a100-production/rl/checkpoints/handoff.ckpt \
  --stage rl --split test --trust-checkpoint --save-predictions
```

`train.sh` and `eval.sh` forward arguments to these Python entry points.
The default profile is `configs/runs/a100.yaml`. Use `rtx4090.yaml` for 24 GiB.
See the [execution manual](../../docs/execution.md) and
[configuration/reward details](../../docs/training.md).

For the original-pair experiment with 256 content tokens per language, measured
instruction overhead, and native non-thinking chat prompts, see the
[preparation, model choices and LoRA audit](../../docs/research/precise_translation_and_lora_20261112.md).
Its profiles are `qwen3_1p7b_4090_precise.yaml` and the matched
`qwen3_1p7b_4090_precise_all_linear.yaml` placement ablation.

For native NLLB with GRPO/correction routing, use
`configs/runs/nllb_600m_correction.yaml`. See the
[objective, benchmark, rental-cost plan and commands](../../docs/correction_training.md).

The [1.3B and 3.3B trials](../../docs/nllb_scale_trials.md) add size-specific
profiles, exact SFT microbatch accumulation and updated GPU recommendations.
