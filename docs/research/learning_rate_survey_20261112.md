# Learning-rate survey for Qwen3-1.7B translation LoRA

Reviewed 12 November 2026. Recommendation: use **1e-5 as the first GRPO LoRA
candidate**, retaining **5e-6 as the matched control**; test **2e-5** next if
the first comparison is stable. **5e-5 is an exploratory candidate**, not a
validated production optimum. The current SFT rate, **1e-4**, is already a
reasonable supervised-LoRA starting point. No live configuration, model,
rental or learning rate was changed for this survey; no GPU pilot or paid
feedback requests were started.

The claim that 5e-6 is necessarily too small is not established. It is half
the current TRL LoRA-GRPO recommendation and above its generic GRPO default.
The best rate depends on the objective, adapter scaling, batch, reward noise
and update reuse, rather than model size alone.

| Primary source | Setting | Learning rate | Relevance |
|---|---|---:|---|
| [TRL PEFT guide](https://huggingface.co/docs/trl/peft_integration#learning-rate-considerations) | Supervised LoRA | 2e-4 | SFT reference; not an online-RL prescription |
| [Qwen training guide](https://qwen.readthedocs.io/en/latest/training/ms_swift.html#supervised-fine-tuning-sft) | Qwen3-8B SFT, rank-8 LoRA | 1e-4 | Supports the current SFT starting rate; different task and model size |
| [TRL GRPO configuration](https://huggingface.co/docs/trl/grpo_trainer#grpoconfig) | Generic GRPO default | 1e-6 | Default, not an optimum for this task |
| [TRL PEFT guide](https://huggingface.co/docs/trl/peft_integration#learning-rate-considerations) | GRPO with LoRA | 1e-5 | Closest general starting-point guidance |
| [TRL LoRA Without Regret example](https://huggingface.co/docs/trl/lora_without_regret#reinforcement-learning-grpo) | Qwen3-0.6B math GRPO, all-linear rank 1 / alpha 32 | 5e-5 | Demonstrates a larger viable example; architecture, reward and adapter scaling differ |
| [M2PO, section 5.2](https://aclanthology.org/2026.acl-long.469.pdf) | Qwen3-4B/Gemma 9B translation preference optimization, rank 32 / alpha 64 | 5e-5 | Closest translation domain; offline multi-pair objective, not our online GRPO |

M2PO also reports batch 32, sequence length 512, 10% warmup and gradient
clipping 1.0. Its preference-objective beta is not interchangeable with this
project's reference-KL coefficient. It supplies evidence worth testing, not
grounds for copying its rate into a different optimizer objective.

[Thinking Machines' original study](https://thinkingmachines.ai/blog/lora/)
finds that learning-rate tuning and adapter placement matter, including
advantages from adapting MLP layers as well as attention. This motivates a
separate capacity/placement ablation if needed. It does not justify changing
target modules and learning rate together, or claim an EN–VI optimum.

The current production setup is Qwen3-1.7B with a frozen foundation and
3,211,264 trainable attention-LoRA parameters, rank 8, alpha 16, BF16 and
AdamW. SFT uses batch 12; RL uses 16 directional sources, four candidates per
source and two optimizer updates per rollout. RL uses beta zero, an 8-update
warmup followed by a constant rate, and normalized COMET/SAC fusion. A small
sampled batch and categorical judge introduce uncertainty into updates; removing
KL also makes a large rate increase a more substantial change. A higher rate
can accelerate both desired adaptation and exploitation of a flawed reward.

The training reward mean is not a direct progress ruler: the code normalizes
each reward stream within groups, applies sigmoid fusion, and then computes
group-relative advantages. Mean reward can remain near 0.5 despite changes
in absolute component quality. Use fixed held-out raw scores and fidelity
checks when deciding whether a rate is helping. The older 7B analysis also
found objective disagreement; increasing the rate cannot by itself reconcile
BLEU, QE and factual preservation.

A controlled, cost-conscious screening experiment would proceed as follows:

1. Preserve one completed RL checkpoint and its Adam moments, counters and
   sampler position. Branch a 5e-6 control and a 1e-5 candidate from that same
   state, using identical source order and seeds. Generated candidates will
   diverge as the policies change; that is expected in online RL. Give each
   branch the same number of fresh rollouts, not merely logger records or
   different numbers of reused updates.
2. Keep LoRA, batch sizes, token caps, two updates/rollout, reward model,
   curriculum, clipping and existing warmup policy fixed. Use 100–200 optimizer
   updates for an initial quality screen. Five or ten steps can check numerical
   stability but cannot establish translation improvement. A promising result
   still needs a longer confirmation and, when practical, repeated seeds.
3. Evaluate both branches before and after training on the **same fixed 1,000
   validation pairs**, clean and bidirectional, with greedy decoding. At batch
   16 this is 125 batches, below the user's 200-batch cap. Report per-direction
   COMETKiwi, SAC, BLEU and chrF++, plus independently checked omissions,
   additions, entities, numbers and negation. Use a fixed-alpha reward in
   comparisons; changing curriculum weights otherwise change the ruler.
4. Record actual optimizer LR, applied adapter-update norm, gradient norm,
   clipping fraction, KL to the rollout policy, candidate diversity and
   cap-hit rates. Stop on non-finite values or numerical failure. Assess shifts
   in these diagnostics relative to the control rather than declaring a
   universal safe clipping/KL threshold.
5. Prefer the smallest rate producing a material held-out gain without a
   meaningful direction-specific fidelity regression. If 1e-5 helps and remains
   stable, compare 2e-5 separately. Test 5e-5 only as a subsequent bounded arm.
   More training and independent evaluation are needed to claim a best rate.

Evaluation size needs care in the current pipeline. Trainer validation uses
the stage's data batch size: SFT 12 versus RL 16. A limit of 200 batches can
therefore cover 2,400 versus 3,200 directional examples from the full 4,000-view
validation set. Those aggregate scores are not a matched SFT-to-RL comparison.
Using the fixed 1,000-pair CSV above ensures both stages cover the same complete
subset within the existing cap. Standalone evaluation uses its separately
configured batch size.

For a **fresh future run**, the first candidate setting would be:

```yaml
learning_rate: {sft: 1.0e-4, rl: 1.0e-5}
```

For **resuming the active run**, that setting alone is insufficient. Lightning
restores optimizer parameter groups from the checkpoint, and
`GRPOModule._optimizer_step` uses their saved `initial_lr` during warmup handling.
A deliberate new-rate branch must override and record both `lr` and
`initial_lr` after restoration, preserve Adam moments and counters, and assert
the effective rate after the first real optimizer step. Changing an on-disk
YAML also does not update an already-running process. This survey has not
implemented or deployed a live rate switch.

Code inspected: `configs/runs/full.yaml`, `configs/runs/qwen3_1p7b_4090.yaml`,
`scripts/main/pipeline.py`, `src/models/grpo_module.py`, and
`src/data/translation_datamodule.py`. Existing checkpoint/probe evidence:
`logs/rtx4090_60296212/repair_summary.json` and
`logs/rtx4090_60296212/model_identity_verified.json`.
