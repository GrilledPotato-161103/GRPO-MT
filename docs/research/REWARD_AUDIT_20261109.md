# Reward and validation audit — 9 November 2026

Instance 59820325, A100 40 GB. Evaluated GRPO checkpoint **step 2800** against its frozen, merged SFT base. No training settings or weights were changed. The same training process resumed after evaluation and the zero-update gradient probe; no rental lifecycle actions were performed.

## Held-out comparison

All 250 validation pairs were translated in both directions: **500 translations per policy**. Both policies used identical examples, prompts, greedy decoding, and source/target caps of 256/128. Disabling the RL adapter exposes exactly its merged SFT base. These are validation results; the 3,500-pair test set was not evaluated in this audit. BLEU and chrF++ are corpus scores on 0–100; COMETKiwi is the raw model score. Reward is mean segment reward, 0.7 clipped COMETKiwi + 0.3 sentence BLEU/100; it cannot be reconstructed from corpus BLEU.

| Direction | Policy | BLEU | chrF++ | COMETKiwi | Mean reward |
|---|---|---:|---:|---:|---:|
| en_vi | sft_base | 34.52 | 53.00 | 0.8238 | 0.6694 |
| en_vi | rl | 35.17 | 53.96 | 0.8316 | 0.6778 |
| vi_en | sft_base | 35.34 | 57.40 | 0.8377 | 0.6807 |
| vi_en | rl | 36.15 | 58.70 | 0.8424 | 0.6867 |
| all | sft_base | 34.94 | 55.24 | 0.8308 | 0.6750 |
| all | rl | 35.94 | 56.38 | 0.8370 | 0.6822 |

Paired reward differences, RL minus SFT (2,000 bootstrap draws; pooled draws retain both directions of each source pair):

- en_vi: +0.00837; 95% interval [+0.00333, +0.01377].
- vi_en: +0.00598; 95% interval [+0.00200, +0.01007].
- all: +0.00717; 95% interval [+0.00406, +0.01047].

## What the logs establish

1. Reward improved early and then plateaued. Sample-weighted rollout reward was **0.6036 in epoch 0**, 0.6376 in epoch 1, 0.6440 in epoch 6, and **0.6455 in the observed part of epoch 11**. These are sampled, augmented training rewards; compare held-out quality using the table above.
2. Recent candidate diversity is 98.52%; zero-variance groups are 0.094%. Mean first-iteration gradient norm is 0.287. Empty gradients or identical candidates do not explain the plateau.
3. BLEU and COMET disagree on **42.48% of comparable within-source candidate pairs**. This means genuine ranking tradeoffs, not proof that either scorer is wrong. Weighted group standard deviations are 0.0192 for BLEU and 0.0260 for COMET; the latter has more influence on within-group variation. chrF++ has zero training reward weight.
4. The per-tensor drift ceiling was first reached at update 1100. At checkpoint 2800, **7/112 A matrices and 11/112 B matrices** were at their limits. This is a partial restriction, not evidence that every adapter is frozen. Zero-initialized B tensors have a radius of 0.1 because the reference-norm floor is 1.0.
5. The backbone already is frozen. Only rank-8 attention LoRA weights train. The RL split has **950 distinct pairs**, revisited about 12 times. Current augmentation changes whitespace/Unicode and sometimes concatenates aligned pairs; it does not add LLM-generated semantic diversity. Exact pair overlap between all splits is zero. The sampled original/exported tokenizer consistency check found 0 differing encodings.
6. Prompt truncation is 0.000% in the recent window, and 1.15% of candidates hit the output cap. Severe pervasive truncation is not supported. Scheduled RL validation is every 10,000 updates, so none was due at step 2800 before this manual audit.

## Gradient probe

This probe uses four fixed held-out source pairs in both directions (32 sampled candidates), the saved checkpoint, and unchanged reward weights. It performs forward/backward computations on an isolated model copy and **zero optimizer updates**. Scoring uses one candidate per graph for memory headroom, versus eight in production; BF16 shape sensitivity and the small sample limit extrapolation to production gradients. Its small sample diagnoses local gradient balance; it does not establish dataset-wide causality.

- Reward-policy gradient norm: **0.44734**.
- Weighted KL gradient norm (beta=0.04): **0.66132**, ratio **1.478** of the reward-policy gradient.
- Cosine between policy and KL gradients: **-0.532**; negative values mean opposing directions.
- Combined norm: 0.56804; L2-SP gradient norm: 0.00114.
- A hypothetical Adam proposal from saved moments retains 99.90% of its L2 step norm after projection, across 11 projected tensors. This is one diagnostic proposal, not a measured fraction of all training progress.

**Interpretation:** the strongest measured optimization hypothesis is KL opposing reward learning: its gradient is larger and points partly against the reward gradient in this probe. The trust-region projection has only a small effect on this hypothetical update, so the fact that some tensors touch their limits is insufficient to blame projection as the main bottleneck. The live run did learn and generalize; it is now making smaller gains under conservative regularization, repeated scarce data, and conflicting ranking criteria. A matched beta ablation is needed to establish a causal fix.

The first reuse iteration has a near-zero centered policy surrogate and ratio exactly one before its update; the second has only about 0.3% clipped tokens. A near-zero scalar policy loss does not imply a zero gradient. Comparing the scalar KL loss to that scalar surrogate would be misleading. This behavior follows the group-normalized objective in [the original GRPO paper](https://arxiv.org/html/2402.03300v3#S4.SS1).

## Recommended controlled experiments

Keep the current run unchanged until choosing an experiment. Use the same checkpoint, data order, seed, and a fixed development split for each short comparison; select improvements on both translation directions rather than on the sampled training curve.

1. Compare the existing beta=0.04 with beta=0.01 while keeping all other settings fixed. This tests whether the observed KL/reward gradient opposition is constraining gains; it is a hypothesis until this ablation succeeds.
2. Separately compare drift limit 0.1 with 0.3 while retaining KL and L2-SP. Avoid changing both restrictions at once. Raw LoRA factor norms are parameterization-dependent, so monitor held-out translation quality as well as parameter drift.
3. Add verified, diverse parallel pairs or validated back-translations within the training partition. Formatting perturbations alone do not provide substantial new semantic coverage. Preserve the untouched validation/test splits.
4. Audit actual BLEU/COMET disagreements and compare reward mixtures on fixed validation examples. More weight on one score is a preference change, not a general remedy. Do not freeze additional layers merely to address this curve: the foundation is already frozen, and reduced adapter capacity could make fitting harder.

These are proposed ablations, not deployed changes or proven causal fixes. [PEFT documents that LoRA keeps foundation weights frozen](https://huggingface.co/docs/peft/developer_guides/lora).

## Artifacts and monitoring

- Local scalar evidence: `logs/training_fast_20261109/completed_reward_audit.json`, `reward_summary.json`, `weighted_reward_summary.json`, `data_audit.json`.
- Remote detailed predictions and metrics: `/workspace/grpo-mt/logs/production/reward_audit_20261109/`.
- Monitor resumed training: `./scripts/rental/monitor.ps1 status` or `ssh -t grpo-a100 "tmux attach -t mt-train"`.
- Post-audit verification: training process 2510 is running; latest progress snapshot is optimizer step 2862, epoch 12.
- [W&B paired validation charts](https://wandb.ai/25025069-vietnam-national-university-hanoi/Machine%20Translation/runs/phomt-fast-user-20261109-091316-reward-audit).
