# Qwen2.5-7B: loss convergence and translation quality

Analysis date: 12 November 2026, Asia/Saigon. This review used archived local
artifacts. It started no rental, GPU inference, training or paid feedback calls
and changed no production configuration. The separate Qwen3-1.7B repair is not
part of these measurements.

The latest 7B run shows early SFT loss flattening followed by an RL tradeoff:
COMETKiwi and semantic-alignment scores improve, while reference-overlap BLEU
falls. Evaluation is therefore not uniformly unchanged. An older cyclic run
also shows a different problem: following SFT phases reverse most RL gains.
These are separate experiments with different rewards and optimizers.

The latest run, `sac-a100-60228333`, used the same 2,000 validation pairs in both
directions at each completed evaluation. The table uses direction-macro scores,
not the slightly different scores computed by pooling both corpora. Its final
SFT evaluation was repeated without a weight update; that duplicate was removed.

| Validation checkpoint | BLEU | chrF++ | COMETKiwi | Direct SAC | Fused reward |
|---|---:|---:|---:|---:|---:|
| SFT 1,000 | 35.3227 | 54.0033 | 0.828452 | 0.876400 | 0.694337 |
| SFT 2,000 | 35.3801 | 53.9411 | 0.829153 | 0.879275 | 0.694509 |
| RL 1,000 | 32.9797 | 53.7014 | 0.838619 | 0.919650 | 0.695998 |

There are only two unique SFT evaluations and one completed RL evaluation in
this export. RL failed at update 1,416 after a malformed SAC reply. The table
does not establish a long-run RL convergence curve. The full test of RL 1,416
gave macro BLEU 32.8351, chrF++ 53.9921 and COMETKiwi 0.842808 on 19,151 pairs;
there is no matched SFT test baseline here. Test SAC was disabled, so test
`reward` is COMET-only and cannot be compared with the fused validation reward.

![Archived loss, evaluation and phase-interference curves](../../diagnostics/qwen7b_convergence_20261112/convergence.png)

**Why SFT loss can improve without better decoded translations.** SFT
`train/loss_step` is teacher-forced target-token cross entropy, including EOS
and excluding prompt/padding. It contains no COMET, SAC, BLEU or KL reward.
The model receives the correct reference prefix while learning; evaluation
generates its own prefix. Increasing probability assigned to reference tokens
need not change the highest-probability decoded token, and small probability
improvements do not guarantee better complete sentences. These logs show a
200-step mean loss of 1.8045 initially, 1.2175 at steps 200–399 and 1.1767 at
1,800–1,999. From steps 800–999 to the final window it falls only 0.76%.
The corresponding SFT 1,000-to-2,000 validation changes are BLEU +0.0574,
chrF++ -0.0622 and COMET +0.00070. This supports diminishing measured benefit,
not a proof of overfitting: held-out teacher-forced loss was not recorded.

**Why RL loss is not a translation-quality curve.** In RL the same
`train/loss_step` label describes the clipped group-relative policy surrogate
plus weighted KL, rather than SFT cross entropy. Beta was zero in this latest
run. The logged training loss follows the last reuse iteration, while detailed
`grpo/loss` diagnostics describe each update. GRPO centers advantages within
each candidate group, so the first surrogate can be approximately zero even
with nonzero gradients. This follows the
[GRPO formulation](https://arxiv.org/html/2402.03300v3#S4.SS1).
Joining SFT and RL losses into one notion of convergence is invalid.

**Why evaluation criteria disagree.** Relative to SFT 2,000, RL 1,000 gains
0.00947 COMET and 0.04038 direct SAC, while losing 2.4004 macro BLEU and
0.2397 chrF++. VI–EN contributes most of the BLEU loss:

| RL-minus-SFT change | EN–VI | VI–EN |
|---|---:|---:|
| BLEU | -1.6236 | -3.1772 |
| chrF++ | +0.2972 | -0.7766 |
| COMETKiwi | +0.00445 | +0.01448 |
| Direct SAC | +0.04295 | +0.03780 |

BLEU/chrF measure overlap with a particular reference; COMETKiwi and SAC
estimate aspects of translation quality and semantic alignment. Faithfulness
means preserving the source's propositions, entities, numbers, negation and
scope. Fluency means natural target-language expression. Reference overlap is
a third property. Faithfulness and fluency can coexist; they are not inherently
opposed. A valid paraphrase can lower overlap, and a fluent omission can fool
an imperfect judge. Aggregate scores cannot distinguish those explanations.
[M2PO](https://aclanthology.org/2026.acl-long.469/) specifically identifies
partial omissions and hallucinations as QE blind spots; that motivates an
independent fidelity audit, not a conclusion that this model is reward-hacking.
The current COMET/SAC adaptation is not the paper's complete multi-pair objective.

**Why the reward chart can look especially flat.** Training separately
standardizes COMET and SAC within each four-candidate group before sigmoid
fusion, then GRPO standardizes the fused reward for advantages. Adding the
same quality improvement to every candidate leaves the first normalized
stream unchanged. Evaluation instead fuses raw scores because singleton
normalization is uninformative. Training and validation reward curves thus
use different transformations. Furthermore, fusion alpha changes from 0.10
at SFT evaluation to 0.18 at RL 1,000. The +0.00149 fused-reward difference
is not a measurement on one fixed ruler. Recompute with fixed alpha for a
controlled comparison; aggregate component means cannot exactly reproduce
mean sigmoid rewards. SAC is also coarse: its four scores are 0.1/0.3/0.7/1.0.
The 0.91965 direct validation mean implies at least 73.2% Full-match labels,
because every other label scores at most 0.7. This is a bound on validation
judgments, not evidence about training-group tie rates or judge correctness.

**Why earlier cyclic evaluations oscillated instead of accumulating gains.**
The older 10-cycle experiment used 1,000 SFT updates at LR 1e-4 followed by
200 RL updates at LR 5e-6. Its LLM criterion judge was disabled; the reward
was COMET/BLEU. RL improved COMET in 10/10 phases, but following SFT degraded
it in 9/9 phases. Summed SFT changes offset 90.08% of summed RL COMET gains
and 91.62% of chrF++ gains. Different SFT/RL projection anchors added
measurable parameter movement; their causal contribution to validation loss
was not isolated. See the
[archived investigation](feedback_learning_investigation_20261111.md).
The latest SAC run uses plain AdamW with beta zero and no projection, so
older KL/projection diagnoses must not be carried over to it.

The evidence does not support blaming insufficient freezing. The latest
foundation was already frozen; only 5,046,272 LoRA parameters, about 0.066%
of the model, were trainable. Neither does the expanded dataset establish
scarcity: its prepared partitions contain 2,373,127 SFT and 593,282 RL pairs.
The earlier cycle audit found substantial unused data. Restricting still more
layers or buying synthetic data should follow an identified failure category.

The next controlled analysis should use a fixed validation subset in both
directions, fixed decoding and fixed-alpha reporting; log validation NLL
alongside raw COMET, SAC, BLEU and chrF. Inspect paired SFT/RL translations
where COMET rises but BLEU falls, including omissions, additions, names,
numbers, negation and valid paraphrases. Calibrate the judge against reviewed
examples independently of its reward. If fidelity is retained, lower BLEU
may be an acceptable paraphrase change; if source facts disappear, the reward
needs targeted correction. Prefer an RL-only control from the selected SFT
checkpoint before reintroducing strong recurring SFT phases. Use validation,
not the already-inspected official test, for these choices. Estimate uncertainty
with paired resampling and judge material changes per direction.

Evidence: [latest SFT history](../../../runs/a100-60228333-20261112/validation/sft/history.jsonl),
[latest RL history](../../../runs/a100-60228333-20261112/validation/rl/history.jsonl),
[loss-window audit](../../logs/a100_60228333/loss-diagnosis.json),
[computed summary](../../diagnostics/qwen7b_convergence_20261112/analysis.json).
Reproduce locally with `../venv/python.exe diagnostics/qwen7b_convergence_20261112/analyze.py`.
