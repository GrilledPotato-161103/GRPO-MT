# Feedback-learning investigation, November 11, 2026

The model learned its scalar reward, but repeated SFT offset most of the measured
RL gains. The configured DeepSeek judge was disabled throughout the run. The
executed pipeline therefore learned from COMETKiwi and sentence BLEU, with no
LLM annotations or verified corrections entering training.

This investigation used the downloaded artifacts only. No rental, training job
or paid feedback request was started, and no training policy was changed.

## Scope and reproducibility

Evidence covers all 20 phase evaluations, 2,000 RL optimizer-update records,
1,000 rollouts, 16,000 directional source groups and 64,000 candidate scores.
Four trusted checkpoints provided the latest RL weights, preceding SFT weights,
best-chrF weights and original reference, together with optimizer states.

Reproduce from the project root using the existing Conda environment:

```powershell
../venv/python.exe diagnostics/feedback_learning_20261111/analyze.py
```

Results: [`analysis.json`](../../diagnostics/feedback_learning_20261111/analysis.json),
[`phase_effects.csv`](../../diagnostics/feedback_learning_20261111/phase_effects.csv)
and [phase comparison chart](../../diagnostics/feedback_learning_20261111/phase_effects.png).
The script asserts checkpoint-layout compatibility, verifies optimizer replay
against the closed-form projection, and checks that zero centered loss can
still produce a nonzero policy gradient.

## 1. The LLM feedback path was inactive

Every archived RL plan has `reward_harness.enabled: false` and
`calibrated: false`. There are no criterion records, repair queues or structured
feedback timing events in the cycle run. With the harness disabled, the code
never constructs the API judge and returns the original reward components.

The actual reward was:

```text
0.7 * clipped COMETKiwi + 0.3 * sentence BLEU / 100
```

chrF++ was evaluated and logged, with zero training-reward weight. COMETKiwi
returns a source/translation quality score; it is not a textual correction
teacher. This agrees with the [model card](https://huggingface.co/Unbabel/wmt22-cometkiwi-da).

Two further distinctions matter before enabling feedback:

- `enabled: true` with `calibrated: false` is a shadow annotation mode. It records
  judgments while preserving the original training reward and group weights.
  Merely turning on API calls will not introduce a new learning signal.
- The calibrated harness converts criteria into scalar rewards and group
  confidence. It does not insert critiques into the translation prompt or train
  on corrected translations. The repair queue currently records identifiers;
  there is no queue consumer that turns them into verified SFT targets.

Code: [`full.yaml`](../../configs/runs/full.yaml),
[`reward_harness.py`](../../src/models/components/reward_harness.py),
[`TranslationReward`](../../src/models/components/translation_reward.py), and
[`GRPOModule`](../../src/models/grpo_module.py). This audit does not establish the
availability or quality of the configured OpenRouter model: it was not called.

## 2. Repeated SFT offset the RL improvements

All phase evaluations used the same fixed 1,000 validation pairs, in both
directions, without augmentation and with evaluation batch size 16.

| Macro metric | Mean change during RL, 10 phases | Mean change during following SFT, 9 phases | RL improved | Following SFT degraded |
|---|---:|---:|---:|---:|
| COMETKiwi | +0.007147 | -0.007154 | 10/10 | 9/9 |
| chrF++ | +0.931215 | -0.948024 | 10/10 | 9/9 |
| Reward | +0.004611 | -0.004577 | 10/10 | 9/9 |
| BLEU | -0.090322 | +0.091295 | 4/10 | 4/9 |

Summed COMET gains during RL were +0.071473; intervening SFT changes summed to
-0.064385. Their ratio is **90.08% offset**. The corresponding chrF++ offset is
**91.62%**. These are descriptive sums of adjacent phase changes, not a claim
about the result of a hypothetical RL-only run or statistical significance.

The schedule gave SFT 1,000 updates at LR 1e-4, followed by 200 RL updates at
LR 5e-6. SFT therefore had five times the updates and twenty times the learning
rate. LR-times-step counts are not measured gradient strength, but the observed
phase changes show that this rehearsal pressure was poorly balanced for
retaining the RL objectives.

There was a real net improvement on the complete 2,000-pair validation set:
COMET 0.827175 to 0.834192 and chrF++ 52.1649 to 52.7886, while macro BLEU fell
from 32.7859 to 32.4816. The model's learning ability is present; the accumulated
benefit is small relative to the repeated phase swings.

## 3. A direction-specific tradeoff is hidden by macro BLEU

| RL-minus-SFT change | EN-VI | VI-EN |
|---|---:|---:|
| Mean BLEU change | +0.811832 | -0.992475 |
| Cycles with BLEU improvement | 10/10 | 0/10 |
| Mean chrF++ change | +1.485049 | +0.377381 |
| Mean COMET change | +0.005885 | +0.008409 |

COMET and BLEU disagreed on **31,788 / 82,230 comparable within-source candidate
pairs: 38.66%**. This is a ranking-disagreement rate, not an annotation error
rate. Improving this mixed reward need not improve each metric in each direction.

Mixed reward's within-source correlation with COMET was 0.9367, versus 0.6204
with BLEU. COMET's mean weighted within-group standard deviation was 2.11 times
BLEU's. These raw variations do not imply a 2.11-times gradient contribution:
GRPO normalizes rewards separately within each source group.

Among groups with differing rewards, the best-reward candidate was shorter than
the group average by 3.58 tokens for EN-VI and 2.25 for VI-EN. Within-source
length/reward correlations were -0.47 and -0.65. This could reflect rejection of
long, poor completions, valid concision, or omissions. Scores and lengths alone
cannot identify which explanation is correct. A blunt length bonus would not
resolve that uncertainty.

[M2PO](https://aclanthology.org/2026.acl-long.469/) specifically studies QE blind
spots around partial omissions and hallucinations and proposes faithfulness
assessment plus multi-pair preferences. That motivates a fidelity audit here;
it does not prove this checkpoint is reward-hacking, nor establish EN-VI gains.

## 4. Protection constraints restrict adaptation, but are not the sole explanation

The backbone was frozen in all 20 phases. Only 5,046,272 attention-LoRA parameters
were trained. Freezing more foundation layers would not address missing feedback.

Both optimizers used L2-SP 1e-3 and a per-tensor drift radius of 10%, but preserved
different anchors. The SFT anchor still exactly matches the original SFT
optimizer's starting anchor. The RL optimizer retains its own initial anchor.

At the last SFT checkpoint, **221/224 adapter tensors** were at least 99.9% of
their allowed SFT radius. At the last RL checkpoint, **80/224 tensors** were
outside that SFT optimizer's radius. The last SFT weights had **10/224 tensors**
outside the RL optimizer's radius.

A replay with the real `TransferAdamW`, zero learning rate and zero task
gradients still moved weights through projection. Switching the last RL weights
to the SFT constraint produced parameter-space movement of 0.05854 L2; the reverse
switch produced 0.03261. Corresponding effective LoRA-matrix changes were
nonzero: 0.02852 and 0.03765 Frobenius norm.

For scale, the actual last RL phase changed parameters by 0.36054 L2 and effective
LoRA matrices by 0.26684. Cross-anchor projection is a concrete additional force,
but these probes do not establish that it caused the observed validation drops.
Tiny bitwise changes inside a radius also occur from floating-point recomposition;
the outside-radius counts above exclude that roundoff.

## 5. What did not fail

- Combined gradient norms were nonzero throughout the 2,000 logged RL updates:
  mean 0.8665, minimum 0.1785. Gradient-norm clipping applied to 28.2% of updates;
  its mean scale was 0.9353.
- PPO clipping affected only 0.31% of scored tokens on average. It is not
  saturating the whole update path.
- Only 4.23% of source groups had zero reward variance, and mean candidate
  uniqueness was 90.08%. There is no general all-ties or exploration-collapse
  explanation in these logs.
- Optimizer counters were preserved: SFT 34,393 updates; RL 2,000. The reference
  checkpoint hash stayed identical across all 20 phases.
- First-reuse policy loss averaged approximately zero, as expected from centered
  advantages when current and old policy scores agree. A local tensor probe
  reproduced loss 0 with gradient norm 0.2886. Zero scalar loss does not mean
  zero learning signal; see the [GRPO formulation](https://arxiv.org/html/2402.03300v3).
- Prompt truncation averaged 0.10% and capped completions 2.44%. Neither explains
  a general absence of learning.
- The run used 8,000 unique RL pairs, only 1.35% of its available 593,282-pair RL
  partition. It used 120,000 unique SFT pairs during the cycles. Fresh training
  data remain available; LLM augmentation is not yet established as necessary.

There is a minor telemetry issue: `optimizer/warmup_fraction` uses phase-local
steps, while the cyclic optimizer uses cumulative stage steps. In **63 rows**,
the chart reports partial warmup despite the actual LR already being full.
This is misleading logging, not a warmup restart. The executed LR and saved
optimizer counters verify the distinction.

KL cannot yet be ranked as the main blocker. Mean sampled reference KL was
0.1602 with fixed beta 0.01; rare spikes occurred. Logs do not separate policy
and KL gradient norms or their cosine, so ratios of scalar loss values do not
establish which gradient dominated.

## Proposed next experiments

1. **Isolate phase interference first.** Fork the saved initial SFT checkpoint
   and run an RL-only control, evaluating every 200 updates on the same fixed
   validation set. Keep reward, KL, sampling and optimizer protection unchanged.
   Treat this as a diagnostic control, not permission to ignore VI-EN BLEU.
2. **Reduce SFT pressure in a separate branch.** Change only its learning rate
   from 1e-4 to 5e-6 initially. If rehearsal is retained later, make it small and
   conditional on measured direction-specific quality drops, rather than an
   unconditional 1,000-update reset.
3. **Test hard projection separately.** Compare `max_relative_drift: null` with
   0.1, retaining the frozen backbone, fixed reference KL and L2-SP. Measure
   proposed/applied update norms, active projection fraction and per-direction
   quality. Do not remove every protection at once.
4. **Connect real feedback deliberately.** Validate the provider route, collect
   training-only EN-VI annotations in shadow mode, verify omission/negation/number
   judgments against reviewed labels, and then test calibrated reward shaping.
   An alternative is verified preferred/corrected targets with a multi-pair
   preference objective. Merely caching critiques does not teach the policy.
5. **Improve observability before another long run.** Record feedback mode,
   judge-call count, calibrated reward-change fraction, corrected-pair consumption,
   stage-cumulative warmup and step counts. On occasional diagnostic batches,
   measure policy/KL gradient norms and cosine plus projection removal. Avoid
   doubling backward work on every update.

The official test has already been evaluated. Use training/validation data for
these decisions and a fresh held-out evaluation for subsequent model selection.
Future semantic calibration should not use the official-test predictions as
training labels.
