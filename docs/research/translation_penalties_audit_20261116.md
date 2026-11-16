# Translation penalties under imperfect references

Date: 16 November 2026. Scope: the latest locally saved NLLB configuration,
current loss implementations, the opt-in noise-aware profile, and the selected
MQM vector design. No new training, judge calls, or rental. This is an audit and
prioritized design; it does not change training defaults.

**Recommendation:** fix checkpoint selection and overconfident sequence-level
correction first. Add source-grounded local error contrasts, calibrated eligibility,
and drift/retention controls. Use direct negative token supervision only when the
specific action is demonstrably wrong in its actual context. Larger scalar
penalties alone do not resolve incorrect judgments or poor credit assignment.

Related: [MQM observations to criterion-specific gradients](mqm_vector_optimization_20261116.md).
The [consolidated training scheme](translation_training_scheme_20261116.md)
integrates these decisions and records the official benchmark's prior inspection.

## 1. Reward and penalty share the same optimization problem

Writing a reward as

\[
 R'(x,y)=R(x,y)-\sum_k\lambda_k C_k(x,y)
\]

does not introduce a new learning mechanism. The effects depend on the validity
of each cost, the estimator, where gradients act, and what alternatives the model
can learn. A penalty that is constant across a group cancels under centering:

\[
 \operatorname{center}(R-c)=\operatorname{center}(R).
\]

Even all-negative rewards can produce positive relative advantages. A harsher
number cannot by itself identify an omitted phrase or teach a better word choice.
An arbitrary negative return is also not a differentiable model loss until it is
connected to action probabilities by an estimator or a supervised contrast.

Distinguish three mechanisms: task-error costs, distribution/retention losses,
and optimizer/update constraints. They solve different problems and should be
logged separately. PPO clipping and gradient clipping are update controls, not
proof of translation fidelity.

## 2. What the saved run actually used

Evidence is the
[resolved configuration](../../runs/nllb-1p3b-60596557-20261115/run/rl/resolved_config.yaml)
and the current source linked below. “Active” here means this saved configuration,
not a currently running rental. The noise-aware and MQM paths are separate stages
of development and must not be credited with this run's results.

| Mechanism | Saved run / current implementation | Limitation |
|---|---|---|
| Direct semantic reward reduction | `clip(COMETKiwi,0,1) × SAC`, with SAC in `{1,.7,.3,.1}` | Coarse source–candidate assessment; not separate costs for fluency, wrong sense, omission or casing |
| Relative negative policy coefficients | Group-centered/scaled advantages on groups routed to GRPO | Negative advantage discourages the whole sampled completion; all-failed samples can still get positive relative coefficients |
| Rejected-versus-anchor correction | Weighted `-log sigmoid(beta_correction × (mean_logp_anchor − mean_logp_rejected))`, plus anchor NLL | Genuine negative contrast, but sequence-wide; accepts only SAC `<.7` as rejection in the legacy route |
| Anchor likelihood | `anchor_weight=1`, legacy likelihood weight 1 | No saturation in the saved mode; can repeatedly reinforce an imperfect anchor |
| PPO ratio clipping | `clip_epsilon=.2` on the GRPO branch | Not a hard trust region; correction preference/NLL terms are outside this clipping |
| Gradient norm clipping | `max_grad_norm=1` | Bounds gradient norm before the optimizer, not actual parameter displacement or semantic damage |
| KL to frozen policy | `model.beta=0`, adaptive control disabled | No reference-policy drift penalty in the saved objective |
| Weight decay | Configured optimizer factory uses `weight_decay=0` | Top-level `model.weight_decay=.01` is unused by that factory; there is no active AdamW decay penalty |
| Round-trip / symmetric consistency | `consistency_weight=0`, fusion disabled | Implemented alternatives, inactive; references in endpoint scoring are not used here |
| Signed-SAC negative severity | `polarity.enabled=false` | Inactive and forbidden by the anchored objective's direct-reward requirements |
| Structured criterion feedback | Conditional harness disabled; correction evidence defaults disabled | No active category-specific orthography/calque penalty; MQM vector losses remain a design |
| Empty / capped outputs | Empty text gets reward zero; anchors exclude empty/capped candidates | Legacy rejection is determined by SAC `<.7`; explicit empty/capped rejection is added only in noise-aware routing |
| Entropy, repetition, copying, source-unit coverage | No dedicated active loss | These failure modes receive only whatever the proxy/SAC happens to detect |

The two `beta` values are different: `model.beta=0` disables KL, while
`model.correction.beta=1` scales preference margins.

Sources: [reward construction](../../src/models/components/translation_reward.py),
[GRPO loss](../../src/models/components/grpo.py),
[correction objective](../../src/models/components/correction_objective.py),
[routing and anchored backward](../../src/models/anchored_grpo_module.py),
and [optimizer/generation/evaluation](../../src/models/grpo_module.py).

Two implementation details matter for proposed fixes:

1. `AnchoredGRPOModule.__init__` explicitly rejects nonzero KL beta, and its
   `_backward_rollout` passes a literal zero into `grpo_loss`. Turning on a YAML
   coefficient is insufficient. A correct retention implementation must update
   construction, reference scoring, the coefficient graph, loss logging and resume
   behavior, including the correction branch.
2. Legacy correction can add anchor NLL even if no rejected pair survives the
   quality-gap screen. The noise-aware route prevents this when it has no
   supported contrast. This is an important self-imitation safeguard, not another
   task penalty.

## 3. BLEU and chrF still affect decisions

Both lexical reward weights are zero, so low BLEU/chrF is **not currently a
gradient penalty** in the saved run. However:

- The best-checkpoint callback monitors `val/chrf_macro`.
- The learning-rate plateau controller monitors `val/comet_macro`.
- The resolved `optimized_metric` is `val/chrf`; whether an external caller uses
  that field depends on the entry point. It is distinct from the callback monitor.

The [pipeline](../../scripts/main/pipeline.py) can hand off the callback's best
checkpoint by default, with an explicit last-checkpoint override. Thus the reward,
schedule, and model-selection criteria are inconsistent. The callback alone is
enough to establish reference-dependent checkpoint selection in this configuration;
it does not establish which historical checkpoint every manual evaluation used.

For these data, “ill-conditioned” is better understood as **an unreliable quality
proxy under reference errors and acceptable variation**, not necessarily numerical
ill-conditioning. chrF's character overlap does not establish semantic correctness.
Corpus-level scores can remain useful regression diagnostics on audited references;
sentence-level overlap should not certify an anchor, reject a faithful paraphrase,
or select the final model on its own. Reference construction affecting automatic
rankings is documented in
[Freitag et al., EMNLP 2020](https://aclanthology.org/2020.emnlp-main.5/).

Replacing the checkpoint monitor with the trained COMETKiwi proxy alone is also
insufficient. QE reward increases can coexist with declining translation quality;
[He et al., NAACL 2024](https://aclanthology.org/2024.naacl-long.451/)
demonstrate this problem and test length/off-target filtering. Their heuristic
thresholds do not transfer automatically to Vietnamese or establish that our
current errors have the same cause.

## 4. Fixes and additions, in priority order

### P0: repair the decisions surrounding the loss

**Unify eligibility and model selection around source-grounded review.** Major
meaning errors must block a positive anchor. Unknown is neither an error nor a
pass. Preserve checkpoints periodically and select on a fixed development panel:
first satisfy declared adequacy noninferiority/error-rate criteria, then compare
fluency/orthography among eligible checkpoints. Report coverage and intervals
separately in both directions. Use development data to choose thresholds and seal
new eligible sources for independent confirmation. The official benchmark has
already been inspected and remains useful with that disclosure. Until the panel is calibrated, retain
candidate checkpoints for reviewed selection rather than inventing a trustworthy
new numeric monitor.

**Use supported contrasts and stop excessive anchor imitation.** Carry forward
the implemented noise-aware safeguards: no anchor NLL without a supported pair,
provenance/evidence weights, soft preference targets, and an NLL floor. The
[opt-in profile](../../configs/experiment/nllb_robust_correction.yaml) is an
ablation, not a tuned recipe. Its `.02` dead band suppresses 77.62% of eligible
archived groups; it is not an estimated judge standard deviation. Calibrate
precision/coverage rather than simply enabling all conservative thresholds.

**Retain sample provenance.** A cap-hit output is censored by the generation
budget, not automatically a linguistic failure. Retry under an adequate bounded
budget or route it separately; do not train a generic short-output preference.
Malformed annotations/API failures require explicit failure handling, not a
fabricated bad translation score. Unknown groups need coverage accounting so
abstention does not silently remove all difficult sources.

### P1: add criterion-specific negative contrasts and positive repairs

The selected MQM design supplies a vector of losses rather than one repeated
sentence coefficient. See the linked derivation for
\(g_k=J_\theta^{\mathsf T}c_k\). Prioritize these verified event costs:

| Error cost | Supervision | Eligibility / exceptions |
|---|---|---|
| Unsupported addition / omission | Contrast a minimal source-supported repair with the failed continuation | Omissions need inserted tokens; deletion/insertion need aligned continuation or EOS boundaries |
| Wrong sense, participants, negation/scope | Contrast the affected phrase or clause | Use only supplied context; penalize only errors robust to unresolved interpretations |
| Numbers, entities, placeholders | Contrast a verified corrected invariant | Numeric localization, identity-preserving transliteration and explicit contracts govern validity |
| Literal calques / unnatural phrasing | Contrast a verified more natural target phrase | Preserve meaning and qualifiers; avoid promoting fluent semantic drift |
| Casing / orthography | Contrast target-convention-correct forms | Judge actual sentence/name context; fragments and mixed-case identifiers are legitimate |
| Wrong language / copying | Contrast a faithful target-language repair | Names, equations, borrowed terms and very short strings are not automatic violations |

Keep one primary owner per error event to prevent counting one number error
again as mistranslation and again as adequacy. A calibrated criterion label alone
does not certify a proposed correction. Preserve repair provenance and verify
that it removes the asserted error without creating another obligation violation.

**Optional direct unlikelihood:** where a specific token action \(b\) is known
wrong at its actual prefix \(h\), add

\[
 L_{\mathrm{UL},k}
 =-\frac1N\sum_{i,t} a_{ikt}r_{ikt}
       \log(1-p_\theta(b_{it}\mid x_i,h_{it})).
\]

Masks and reliability weights are detached. This discourages a known action
without group-centering cancellation. It should be a separate ablation after
repair contrasts, not an automatic addition to every rejected sentence. It
redistributes probability to alternatives without identifying a correct one,
and may duplicate the pair loss on the same event. Prefer a verified replacement
when available. Do not blacklist a word everywhere because it was wrong once,
or place an omission penalty on a nonexistent token. Use a stable implementation
near probability one and check gradient signs on real aligned token examples.
[Unlikelihood training](https://arxiv.org/abs/1908.04319) is an established method
for text-generation degeneracy; its reported repetition results do not validate
automatic EN/VI semantic rejection labels.

### P1: add distribution and capability retention

A small, measured drift/retention term should accompany stronger local negative
supervision. Choose an explicit frozen SFT policy and clean source-conditioned
prefixes. One option is forward KL on their next-token distributions:

\[
 L_{\mathrm{retain}}=
 \mathbb E_{h\sim\mathcal D_{\mathrm{clean}}}
 \sum_v p_0(v\mid h)\log\frac{p_0(v\mid h)}{p_\theta(v\mid h)}.
\]

Here \(p_0\) is frozen, and both policies see the same actual source/prefix.
This is a distributional retention objective on specified prefixes, not a
universal guarantee that translations preserve meaning. Exclude known bad
positions from imitation constraints and measure the conflict with desired edits.
Clean supervised exposure is a complementary control if its targets were checked.

Do not conflate this with PPO old-policy clipping or claim that logged zero KL
means no drift when KL computation is disabled. A sampled divergence estimate
also depends on the sampling policy; repeated updates on old rollouts do not
turn it into exact current-policy KL. Define the estimator and evaluated contexts.

The coefficient must be measured in a pilot; an oversized value preserves the
starting calques/casing errors. The frozen policy and any controller/optimizer
state must survive checkpoint resume. If adding the previously proposed joint
gradient protection, constrain actual optimizer displacement or use a verified
compatible update rule; raw-gradient projection followed by Adam is insufficient.

### P2: add narrowly validated degeneracy costs

**Repetition loops and unsupported copying:** first measure incidence and precision
of detection on model outputs. A repeated phrase can be source-supported emphasis
or a recurring name. Flag a model-created loop only after accounting for the
source/contract. Prefer a local contrast or high-specificity action penalty over
an unconditional repeated-n-gram tax.

**Length and coverage:** use directional, tokenizer-specific length distributions
as anomaly triggers, conditional on source length/text type. Escalate unusual
cases to source-unit coverage assessment. Do not make 1:1 length or a narrow
reference-derived ratio a truth criterion. Vietnamese whitespace units often
represent syllables; over half the raw training English sources have at most ten
units, so short-text exceptions matter. Attention coverage is not verified semantic
coverage, and raw train/test reference statistics do not calibrate a candidate
error detector. Evidence: the
[full-corpus audit](../../runs/comet-reference-20261115/dataset_analysis/report.html).

**Exploration:** there is no dedicated entropy bonus now. Add one only if measured
collapse prevents finding useful alternatives. Increasing entropy indiscriminately
can increase wrong-language, copying and nonsense outputs. Candidate diversity,
useful-contrast discovery, entropy and error incidence must be measured together.
LoRA capacity and decoding temperature are not substitutes for label reliability.

Generation currently avoids nucleus/top-k filtering and repetition processors to
preserve the policy used by GRPO ratios. Adding decoding-time penalties or hard
constraints requires scoring the actual sampling distribution consistently; do
not quietly compare a processed sampler against unprocessed log probabilities.

## 5. Prefer error budgets to compensating quality scores

An intended policy-level formulation is

\[
 \max_\theta J_{\mathrm{naturalness}}(\theta)
 \quad\text{subject to}\quad
 \mathbb E[C_k(x,Y)]\le\epsilon_k\quad(k\text{ semantic}),
 \qquad D_{\mathrm{retain}}\le\delta.
\]

This expresses that a surface gain should not purchase additional negation or
omission errors. Define event costs, budgets and reference populations with
bilingual calibration. Do not interpret a partially observed, accepted-only
error rate as the full-population expectation. Report unknown fractions and
review a representative sample of abstentions; without that information the
population constraint is not established.

Lagrange multipliers can increase when a measured cost exceeds its budget,
\(\lambda_k\leftarrow[\lambda_k+\eta_\lambda(\widehat C_k-\epsilon_k)]_+\),
but this is still a scalarized update at each fixed multiplier setting. It needs
stable, calibrated cost estimates and does not itself solve token credit.
For this project, local repair losses plus explicit adequacy eligibility and
measured gradient conflicts are the more concrete first implementation.
[Constrained Policy Optimization](https://proceedings.mlr.press/v70/achiam17a.html)
provides research context; this proposed surrogate does not inherit its guarantees.

## 6. Avoid adding these penalties blindly

- Negative BLEU/chrF distance from one imperfect reference.
- A larger global SAC penalty followed by unchanged group normalization.
- A blanket penalty for lowercase starts, missing Vietnamese diacritics, copied
  strings, repeated words, or nonmatching numbers without context/localization.
- A narrow length-ratio target, a universal EOS tax, or a forced-length bonus.
- A universal cycle-consistency penalty: two directions can agree on the same
  omission or choose a reversible but unnatural representation.
- High KL that locks in existing defects, unconditional entropy rewards, or
  categorical “critical” penalties based only on an uncalibrated judge assertion.

## 7. Validation before deployment

Use one fixed, reviewed candidate pool to compare the saved objective, noise-aware
controls, local MQM contrasts, and then retention/unlikelihood additions separately.
Match update tokens and feedback budget. Validate criterion gradient signs,
token alignment, correction/retention masking, and actual optimizer displacement
on a tiny real model before running a quality experiment.

On disjoint development sources, report both directions separately: semantic
error rates, calque/wrong-sense and orthography rates, faithful-paraphrase false
rejection, negative-label/repair precision, unknown coverage, EOS/cap-hit behavior,
and retention drift. Report reviewed paired differences with uncertainty and
source clustering. Keep BLEU/chrF and untrained QE metrics as complementary
diagnostics with their limitations. A penalty is successful only when independently
reviewed errors decrease without unacceptable semantic/coverage regressions.

This audit establishes which controls exist and which were inactive. It does
not establish that any proposed coefficient or penalty has improved EN/VI quality.
