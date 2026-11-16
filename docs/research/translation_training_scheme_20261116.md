# Consolidated Vietnamese–English optimization and training scheme

Date: 16 November 2026. Scope: EN→VI and VI→EN, starting from the NLLB 1.3B SFT
policy used for the existing comparisons. This is the current design and
implementation handoff. It supersedes earlier recommendations for a new
experiment where they conflict; historical results and runtime defaults remain
as recorded. It is not a runnable MQM training profile.

**Decision:** develop source-grounded MQM grading, verified local repair contrasts,
criterion-specific gradients, and measured retention. Establish reliable feedback
before strengthening optimization. Test each addition separately. Keep the old
scalar correction as a control, and defer a new on-policy ranking branch until
grading, supervision and independent evaluation have passed their checks.

The latest NLLB SAC already compares source and candidate directly. The new
design changes its granularity and connection to learning. Imperfect references
still matter through SFT, fallback anchors and lexical checkpoint selection.

## 1. What is established, and what is pending

| Component | Established status | Remaining work |
|---|---|---|
| Official PhoMT reference audit | Full train/test scoring and descriptive statistics complete, both directions | Do not interpret QE as human accuracy or reference truth |
| Existing anchored correction | Implemented sequence-level preference + anchor likelihood, with GRPO on other eligible groups | Coarse SAC can miss calques/casing; no active retention |
| Noise-aware scalar controls | Implemented opt-in; historical focused suite: 54 CPU tests passed, 4 deselected | Calibrate thresholds and measure translation quality |
| MQM grading engine 1 | Selected; prompt and output-contract designs documented | Runtime parser, coverage/event validation, cache revision and bilingual calibration |
| Local criterion losses | Formula and 16 synthetic numerical checks complete | PyTorch/tokenizer integration and real-model gradient checks |
| Retention and joint gradient protection | Specified | Trainer implementation, optimizer semantics, memory measurements and ablations |
| New model-selection rule | Specified | Reviewed development panel and selector; current saved callback still uses chrF |
| Infrastructure closure | Reference rental destroyed; local backup verified | No new rental or training was started for this scheme |

The [reference report](../../runs/comet-reference-20261115/dataset_analysis/report.html),
[software validation log](../../diagnostics/noisy_objective_20261115/validation.log),
[mathematical verification](../../diagnostics/mqm_vector_20261116/verification.json)
and [rental closure](../../runs/comet-reference-20261115/closure.json) establish
different facts. None establishes a translation-quality improvement from MQM.

## 2. Data and the baseline

Raw reference COMETKiwi means are:

| Official split | Parallel pairs | EN→VI | VI→EN |
|---|---:|---:|---:|
| Training | 2,977,999 | 0.799835 | 0.798563 |
| Test | 19,151 | 0.808565 | 0.805733 |

These are reference-only scores, not a quality ceiling, human error rates or
acceptance thresholds. The older model comparison used 1,600 matched pairs per
direction at RL step 3,600. Its proxy gains do not prove independent quality
gains, and it does not represent every later checkpoint or the full official test.

Data decisions:

- Verify fingerprints and connected-group exclusions on the **actual prepared
  SFT/RL/development outputs**. Raw official overlaps alone do not prove leakage
  in the already prepared project partitions. Do not reuse historical filtered
  counts as verification of a new preparation.
- Retain both directional views, but judge and route them independently. An
  incomplete Vietnamese source cannot support recovering facts from its English
  partner just because that partner is the dataset target.
- Keep native complete token-bounded examples. Do not truncate a clause and then
  train its original full target. Preserve valid paraphrase alternatives.
- Use length, copying, lowercase and punctuation flags for stratified inspection.
  Short fragments, names and formulas need exceptions; Vietnamese whitespace
  units are often syllables. Do not convert raw flags into error labels.
- Do not invent row-level domains or neighboring document context where the
  downloaded archive does not supply reliable provenance. A policy and its grader
  must have the same disambiguating source context for a mandatory training label.

Start every comparison from the same fingerprinted, reload-verified SFT export.
Verify the export's local availability and provenance before a future run; the
example path in an old YAML is not evidence that an export exists there. Retain
the existing RL model as a comparison, rather than using its weights as the
starting point of only some arms. Additional clean SFT recovery is optional and
must be a separate arm with reviewed targets, not another pass over all references.

Use training-only sources for grader/repair calibration and disjoint development
sources for tuning and checkpoint selection. The 240-item training worksheet is
blank reference-review scaffolding: add actual model candidates, accepted
paraphrases, erroneous contrasts and verified repairs. Measure per-criterion
precision/recall, span accuracy, repair validity and abstention by direction and
source-length/context stratum. Judge self-confidence is not this calibration.

**Evaluation history matters:** official-test reference statistics and some model
predictions have already informed this investigation. Continue reporting the
official benchmark transparently, but do not call it a newly untouched
confirmatory set. For an independent confirmation, seal new audited source groups
excluded from SFT, RL and development before further tuning. Alternatively use
groups whose exclusion and prior nonuse can actually be established. A new split
of already fitted training examples does not create such a set.

## 3. Grading and repair eligibility

Use [proposal 1](sac_grading_engines_20261115.md#1-source-grounded-mqm-style-error-vector),
composed with its common policy and the
[coverage/event extension](mqm_vector_optimization_20261116.md#3-define-the-observation-before-defining-its-cost).
The judge sees actual source, candidate, direction, genuine context and explicit
translation contract. Hide dataset references, model identity and prior proxy
scores from the primary assessment.

Retain separate observations for omission, addition, mistranslation, negation/scope,
numbers, entities, target language, fluency, orthography, applicable terminology
and format. Adequacy is a summary, not another charge for the same semantic event.
Every criterion has assessed/unknown/inapplicable coverage. Error events have a
primary owner, severity, exact quoted span occurrences and concise evidence.
Ordinal grades remain ordinal; unknown is neither pass nor failure.

Route each observation as follows:

| Evidence | Training action |
|---|---|
| Verified semantic error and verified repair | Local semantic contrast; the repaired sentence is not automatically a globally faithful anchor |
| Verified surface error, resolved adequacy, no known semantic defect | Meaning-preserving fluency/casing contrast |
| Multiple acceptable translations, no verified difference | No artificial winner or compulsory imitation |
| Unknown interpretation or disputed repair | Abstain for that criterion; preserve review/coverage accounting |
| All candidates fail and no verified replacement exists | No positive anchor; bounded repair/resampling attempt, then quarantine |
| Invalid JSON, unsupported span, missing coverage or backend failure | Explicit controller failure/retry; never fabricate a reward |
| Generation hits its cap | Treat as censored; bounded budget retry or separate route, rather than a universal linguistic penalty |

If there is no eligible task signal and no scheduled retention signal, skip the
optimizer step, including momentum/decay effects; log the attempt separately from
optimizer updates. Bound consecutive unusable rollouts explicitly so the pilot
cannot spend indefinitely while its optimizer counter stays fixed.

Repair candidates may come from reviewers, checked transformations, or a bounded
editor. Their eligibility requires calibrated verification against the source
and applicable obligations. Rephrasing a grader prompt is not statistical
independence. A reference can be considered as a candidate repair only after the
same checks; it has no privileged status.

## 4. The objective and its gradient

For source \(i\), criterion \(k\), verified repair \(y^+\) and original \(y^-\),
score both complete sequences under their own prefixes. Define local mean log
probabilities \(s_{ik}^{\pm}\) using independently aligned nonempty token masks.
Let \(w_{ik}=a_{ik}r_{ik}v_{k,s}\): eligibility, empirically supported reliability
and bounded severity cost, all detached. The selected objective is

\[
 L_k=\frac1N\sum_i w_{ik}\left[
 \operatorname{BCEWithLogits}(\beta_k(s_{ik}^+-s_{ik}^-),q_k)
 +\frac{\alpha_k}{2}[-s_{ik}^+-\tau_k]_+^2\right],
 \qquad g_k=\nabla_\theta L_k=J_\theta^{\mathsf T}c_k.
\]

The soft preference target \(q_k\in(.5,1)\), preference scale \(\beta_k>0\),
anchor weight \(\alpha_k\ge0\), and NLL floor \(\tau_k\ge0\) require calibration.
The squared positive-part NLL term is continuously differentiable and stops
pushing once the selected repair is sufficiently likely. It is not bounded above;
step/gradient controls are still required. This new squared-hinge term differs
from the implemented scalar noise-aware branch's linear hinge.

\(N\) counts all scheduled directional source records, including abstentions.
Keep this denominator and the scheduled direction mixture fixed across chunks
and ranks; do not amplify weak evidence by renormalizing surviving weights.
Select one verified event per `(source, criterion)` or average multiple events
explicitly. Do not multiply exposure through duplicated error annotations.

This is a supervised local contrast surrogate. No gradient passes through the
judge or sampled token IDs. It is not an unbiased token policy-gradient estimator
or a reproduction of DPO/CPO. A fixed sum of sentence rewards still collapses to
a scalar; the additional information comes from distinct targets, masks and
gradient handling. Masking a loss site does not isolate a semantic module in NLLB.

Alignment must validate Unicode offsets, tokenizer normalization and actual token
IDs. Omission masks include inserted content and the original divergent
continuation/EOS. Deletion masks include the repaired continuation/EOS. Expanded
scope is required for reordered clauses. Exclude prompts, padding and forced
language tokens; EOS counts only when it is an actual selected action. Do not
splice token sequences and reuse stale suffix probabilities.

## 5. Penalties, retention and the update rule

Use the [penalty audit](translation_penalties_audit_20261116.md) as the detailed
inventory. The decisions for the new scheme are:

| Mechanism | Decision |
|---|---|
| Semantic errors and surface defects | Verified local negative contrasts plus positive repairs |
| Unsupported anchor imitation | Require supported contrasts; soften preference targets and saturate anchor NLL |
| Retention | Add measured frozen-SFT distribution retention and/or reviewed clean exposure as an explicit ablation |
| PPO/gradient clipping | Retain where applicable; do not interpret as semantic protection |
| Direct token unlikelihood | Optional later ablation for demonstrably wrong actions at actual prefixes; avoid double-charging the same event |
| Copying/repetition/length | Detect and review; penalize only calibrated high-specificity errors |
| Entropy bonus | Defer unless useful-candidate discovery demonstrably collapses |
| Lexical penalties, global harsher SAC, blanket casing/ratio rules | Exclude from the primary scheme |
| Symmetric endpoints/cycle consistency/fusion | Keep disabled in the primary experiment |

Retain a frozen SFT policy on reviewed clean source/prefix contexts, for example
with \(\mathbb E_h\operatorname{KL}(p_0(\cdot\mid h)\|p_\theta(\cdot\mid h))\).
Use matched contexts, measure drift, and do not require edited positions to imitate
known starting errors. Retention does not establish faithfulness and excessive
retention can preserve defects. Its coefficient is a measured setting, not an
inherited universal KL constant.

The existing anchored trainer rejects nonzero `model.beta` and passes literal
zero to its KL loss. Retention requires code integration across construction,
scoring, loss coefficients, logging and checkpoints, including correction updates.
It cannot be enabled correctly by changing that YAML field alone.

Initially preserve criterion gradients to log norms/cosines, then use a declared
fixed sum as the transparent control. When testing conflict handling, jointly
project the **actual proposed displacement** \(d_0\):

\[
 d^*=\arg\min_d\tfrac12\|d-d_0\|^2
 \quad\text{s.t. }g_p^{\mathsf T}d\le0\ (p\text{ protected semantic}),
 \quad\|d\|\le\rho,\qquad\theta\leftarrow\theta+d^*.
\]

Use matched plain-SGD controls for the first projection comparison, or implement
and verify optimizer-displacement projection explicitly. Projecting raw gradients
and then using Adam does not preserve these constraints. Zero semantic gradients
supply no protection, and first-order batch constraints are not guarantees for
finite steps or individual translations. Apply finite-step clean/retention probes;
rejected steps must restore parameters and optimizer state.

## 6. Training stages and pilot envelope

| Stage | Work | Exit condition |
|---|---|---|
| A — baseline and data | Reload SFT; verify preparation, partitions, decoding and baseline predictions | Fingerprinted reproducible baseline and valid exclusions |
| B — feedback calibration | Review training-only candidate contrasts/repairs; freeze prompt/parser/calibration | Accepted category precision, coverage and repair validity; unsupported categories stay review-only |
| C — correctness | Implement MQM loss, masks, coefficient replay, retention and diagnostics | Tiny real-model gradients, optimizer/resume and mask tests pass |
| D — fixed-pool pilot | Compare sequence versus local supervision on the same cached evidence | Independent development evidence supports local supervision |
| E — retention and conflicts | Add retention; separately compare joint gradient handling | Improved tradeoff without unacceptable adequacy/coverage regression |
| F — online candidates | Refresh candidates from the current policy; optionally test restricted on-policy ranking | Independent quality improves under fixed budgets and stopping rules |
| G — final report/export | Freeze selector and settings, evaluate benchmark and sealed confirmation | Matched bilingual report; verified inference and resume artifacts |

The first MQM pilot learns from generated/verified corrections; this is
policy-assisted online correction training, not pure on-policy RL. First keep
the candidate pool fixed to isolate the objective. Treat true policy-gradient
ranking as a later, separately measured branch. Permit it only for calibrated,
fully assessed eligible groups with a supported quality distinction. Equal or
unresolved candidates provide no ranking signal. Keep external repairs outside
old-policy importance ratios and avoid opposing GRPO/correction signals on the
same candidate set. Do not restore per-group standard-deviation amplification
of tiny score differences.

Planning defaults for a **future** bounded pilot, not validated optima:

| Setting | Choice / accounting |
|---|---|
| Policy | Same NLLB 1.3B SFT export in every arm; frozen backbone, all-linear rank-8 LoRA, dropout 0 |
| Precision and memory | BF16 model, FP32 probability/loss calculations, activation checkpointing |
| Inputs | Native language codes, raw source, complete 256-content-token bounded pairs |
| Candidate generation | Four candidates/directional source; temperature 1; full-support sampling with scored distribution matching sampling |
| Scheduled batch | Up to 16 directional records = 8 paired bilingual rows; four candidates gives up to 64 original completions before deduplication/repairs |
| Scoring microbatches | Start with existing 4,096 padded source+target token cap; count repairs and retention work too; measure full-pipeline peak memory |
| Update reuse | Start with one update/fresh rollout for online MQM pilot; keep fixed evidence pools explicitly versioned for offline arms |
| Optimizer control | Existing AdamW at LR `2e-5`, 100-update warmup, decay 0, gradient norm cap 1 as a reproducible comparison; projection arms use matched SGD controls with development-chosen LR |
| Pilot ceiling | At most 1,000 optimizer updates/arm; validation every 100; earlier stop for invalid evidence, inadequate exposure or reviewed regression |
| Evaluation | Fixed source IDs, independent greedy evaluation and explicit counts; separate full benchmark reporting from any bounded development panel |

Length bucketing can reduce the scheduled batch, so log actual directional source
records, tokens, candidates, eligible events, rollouts and optimizer updates. A
1,000-update ceiling is not an epoch or a commitment to train that long. With full
16-record batches and one fresh rollout/update, it schedules at most 16,000
directional views and 64,000 original completions; these are not distinct corpus
counts and exclude repairs/retention. The local-loss numeric examples in the
verifier are not chosen production hyperparameters.

The old `.02` noise floor and `.75` QE anchor screen remain scalar-control
sensitivity settings. The former suppresses 77.62% of eligible archived groups;
neither is a calibrated MQM setting. Do not copy the old 32-token SAC response
budget for structured grading. Start offline response-length profiling around
the existing 2,048-token criterion budget and handle truncation explicitly.

Keep gradients on trainable adapters. K full-size gradient buffers for full
1.3B fine-tuning are not a plausible unmeasured 24 GB plan. Profile criterion
backward/recomputation costs and exact parameter-gradient dot products. Token
coefficient cosines are not a substitute for parameter-gradient cosines.
The completed 4090 reference-scoring run does not benchmark this training load.

## 7. Ablations and success criteria

Use sequential comparisons, stopping unpromising branches rather than launching
every combination:

| Comparison | Controlled question |
|---|---|
| Legacy scalar vs noise-aware scalar | Do noise controls help, using the same scalar feedback? |
| MQM-verified sequence contrasts vs local criterion contrasts | Does localization help, with the same repairs/evidence and optimizer? |
| Local contrasts vs local contrasts + retention | Does retention improve the quality/drift tradeoff? |
| Matched SGD sum vs matched SGD joint projection | Does conflict handling help beyond the optimizer change? |
| Best supported scheme vs optional unlikelihood or restricted on-policy ranking | Does that individual addition help? |

Match source IDs, starting weights, generated pools, decoding, effective update
tokens and seeds within each controlled comparison. Legacy and MQM feedback have
different costs: record annotation, repair, API tokens/calls, wall time and GPU
work rather than claiming identical budgets automatically. Reuse immutable
evidence only when its exact inputs, prompt/parser/model and verification
provenance match. Refresh gradients at current weights; never reuse stale
coefficients after an update.

Before model selection, predeclare adequacy noninferiority and surface-improvement
margins with a justified sample size. Compare reviewed paired error differences
with source-clustered uncertainty. Selection is lexicographic: satisfy semantic
criteria in **both** directions first, then improve fluency/orthography with
acceptable exposure/cost. A macro average cannot hide a failing direction.

Report omissions, additions, participant/scope/negation errors, numeric/entity
errors, wrong senses/calques, applicable casing defects, faithful-paraphrase
false rejection, repair precision, unknowns and source coverage. Log gradient
norms/conflicts, likelihood margins, anchor saturation, stalled/projected updates,
actual displacement and retention drift. Supplement with COMET variants and
corpus BLEU/chrF, while identifying the trained proxy and imperfect references.

The current `val/chrf_macro` best-checkpoint callback and COMET-only plateau
decisions do not implement that rule. Add the reviewed selector and align all
checkpoint/handoff decisions with it. Keep periodic checkpoints; until review
exists, do not invent a reliable replacement score. A training-reward increase
alone is not a release condition.

## 8. Implementation and reproducibility handoff

Implement in this dependency order:

1. MQM adapter/parser, explicit coverage/events, exact-span validation and cache
   revision in the feedback layer; collect external calibration evidence.
2. Repair ingestion/verification and tokenizer-aware insertion/deletion/casing
   masks, including changed prefixes and normalization failures.
3. Token-level loss vector and criterion coefficient replay alongside existing
   correction code; exact dense/microbatched and per-criterion autograd checks.
4. Retention integration and checkpointed state; real-model optimizer/resume tests.
5. Reviewed development selector, per-direction coverage/quality reporting and
   callback/handoff integration.
6. Optional joint displacement solver, then optional unlikelihood/on-policy arms,
   each behind a separate objective configuration and verified against its control.

Pin model/tokenizer/dataset revisions; record prepared split hashes, source IDs,
candidate/repair text hashes, context/contract, prompt/parser/calibration versions,
actual token IDs/masks, event ownership, eligibility and sampling. Checkpoint
adapters/base provenance, optimizer/scheduler/controller state, data cursor and
random state. Verify both inference reload and exact supported training resume.

The immediate next work is calibration and runtime integration, not a longer
scalar-reward run. The
[existing scalar-only profile](../../configs/experiment/nllb_robust_correction.yaml)
is a control and does not implement this MQM scheme. No new launch command is
provided until the required runtime path and reviewed parameters exist.

The reference evaluation's complete local backup and destruction record remain
available. For a future authorized rental, preserve final weights, resumable
state, evidence, manifests and logs locally with verified hashes before teardown;
do not rely on an ephemeral remote checkpoint.
