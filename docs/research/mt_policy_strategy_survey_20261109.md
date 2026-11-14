# Reinforcement strategies and conditional reward policies for EN–VI translation

Research date: 9 November 2026. Targeted survey using primary papers, current repository code, and the saved training audit. This document proposes experiments and interfaces; it does not implement the proposed policies or establish their performance. Training and rental 59820325 were stopped at update 3002. The quality comparison below evaluates checkpoint 2800, not the final stopped checkpoint.

## Recommendation

Keep the existing frozen backbone and LoRA configuration for the first comparison. Test a smaller KL coefficient before changing the GRPO algorithm. Then replace the single unconditional quality mixture with a harness that distinguishes semantic fidelity, lexical agreement, output validity, and uncertainty. Preserve strict treatment of verified meaning errors while relaxing surface preferences where alternatives are valid. Use cached, structured preference feedback to improve data coverage before introducing more expensive gradient methods.

There are three different things to control:

- **Translation criteria:** adequacy, omissions/additions, fluency, terminology, entities/numbers, language, formatting, and context.
- **Update constraints:** reference KL, old-policy ratio clipping, L2-SP, and adapter drift projection. These do not directly certify translation correctness.
- **Training strategy:** GRPO, preference optimization, minimum-risk training, or self-play. Changing the optimizer does not repair an unreliable evaluator.

## Evidence from the previous phase

Sources: [completed audit](REWARD_AUDIT_20261109.md), [scalar evidence](../../logs/training_fast_20261109/completed_reward_audit.json), [sample-weighted history](../../logs/training_fast_20261109/weighted_reward_summary.json), and [data audit](../../logs/training_fast_20261109/data_audit.json).

| Observation | Supported interpretation | What it does not establish |
|---|---|---|
| Training reward 0.6036 in epoch 0, 0.6376 in epoch 1, approximately 0.644–0.646 later | Early learning followed by diminishing gains | Training failure or monotonically improving held-out quality |
| Full validation reward 0.6750 → 0.6822; paired 95% interval for the difference +0.0041 to +0.0105 | Improvement over the merged SFT base on this validation split | Generalization to every domain or improvement of checkpoint 3002 |
| BLEU 34.94 → 35.94; chrF++ 55.24 → 56.38; COMETKiwi 0.8308 → 0.8370 | Both lexical and semantic proxy metrics improved | Absence of partial omissions, hallucinations, or terminology errors |
| Recent BLEU–COMET candidate ranking disagreement 42.48% | The rewards frequently prefer different alternatives | A 42.48% translation-error rate, or proof that COMET/BLEU is wrong |
| Weighted within-group standard deviation: BLEU 0.0192, COMET 0.0260 | COMET has more within-group variation after mixing | Its 70% coefficient gives exactly 70% of the learning signal; covariance and normalization matter |
| KL gradient norm 0.6613 versus reward-policy 0.4473, cosine −0.532 | The gradients oppose each other in the diagnostic sample | Production-wide dominance: the probe used four pairs and scoring microbatch 1 rather than 8 |
| Projection retained 99.90% of the hypothetical Adam step norm; 18/224 tensors at their drift bounds | Projection is a weaker suspect than KL in this probe | Projection is harmless everywhere; retained norm is not retained task progress |
| Candidate uniqueness 98.52%, zero-variance groups 0.094%, second-iteration clip fraction 0.31% | No clear diversity collapse, tie starvation, or pervasive clipping | Token entropy is healthy, or exploration is sufficient for every source |
| 950 RL pairs, approximately 12 passes; augmentation is formatting plus aligned concatenation | Repetition and limited semantic coverage plausibly limit improvement | Proven overfitting; this needs checkpoint histories and independent error evaluation |

The measured negative gradient cosine concerns **combined reward versus KL**, not BLEU versus COMET gradients. Candidate-ranking disagreement and gradient interference are different measurements. Directional reward levels also need calibration: a lower EN→VI score does not, by itself, prove unfair treatment or negative transfer.

The run optimized `0.7 * clip(COMETKiwi, 0, 1) + 0.3 * sentence_BLEU/100`. chrF++ was diagnostic only. It used parallel-data GRPO; the separate LLM-feedback self-play path was not active. Scheduled validation was every 10,000 optimizer updates, so the audited early plateau had no scheduled RL validation point yet.

## Which criteria each reward emphasizes

The following conditional rules are **proposals for this project**, not properties guaranteed by the cited metrics.

| Criterion | Useful evidence | Main conflict or blind spot | Proposed conditional treatment |
|---|---|---|---|
| Semantic adequacy | Source-aware QE plus structured bilingual assessment | A fluent partial translation can receive a high holistic score | Use separate omission, addition, mistranslation, and negation labels; do not let fluency compensate for a verified major meaning error |
| Lexical fidelity | Sentence BLEU and chrF++ | Penalizes legitimate paraphrases, alternate word order, and reference imperfections | Reduce the lexical preference when independent evidence confirms equivalence; retain it when exact wording is required |
| Fluency and readability | A separate fluency rubric; limited use of LM likelihood | Naturalness can reward invented or omitted content | Rank fluency within a comparable fidelity tier; do not use the policy's own confidence as proof of truth |
| Numbers, units, dates, entities | Typed extraction, source alignment, normalization, bilingual checks | Valid localization, transliteration, unit conversion, and name conventions defeat raw string equality | Compare canonical values and explicitly permitted transformations; abstain when parsing or alignment is uncertain |
| Terminology | A supplied glossary, domain, and context | A single preferred term can be wrong outside its sense or domain | Apply only to active, unambiguous glossary obligations; support approved inflections and alternatives |
| Target language | Language ID plus script/context evidence | Names, short fragments, borrowed words, and code switching are ambiguous | Apply a strong penalty only with calibrated evidence; otherwise mark unknown and retain other criteria |
| Completeness and length | Semantic coverage plus direction/length-conditioned reference distributions | Uniform length ratios punish normal EN–VI expansion; token caps can create censored outputs | Use a soft calibrated band for triage, not a 1:1 rule; route capped candidates to a bounded longer decode or repair queue |
| Format and placeholders | Explicit output contract, placeholder parser, markup validation | Translation-only prose and code/document translation require different invariants | Enforce only declared obligations; preserve required placeholders and markup exactly where specified |
| Repetition and copying | Duplicate clauses, source alignment, semantic coverage | Lists and repeated technical terms can be correct | Penalize unsupported duplication; do not penalize every repeated n-gram or identical named entity |
| Discourse, register, pronouns | Document context and task-specific rubric | Isolated sentence metrics cannot resolve unavailable context | Assess only with supplied context; record ambiguous cases instead of fabricating certainty |

[BLEU](https://aclanthology.org/P02-1040/) measures reference overlap and brevity; [chrF++](https://aclanthology.org/W17-4770/) combines character and word n-grams. They are inexpensive lexical signals rather than semantic verifiers. The deployed [COMETKiwi model card](https://huggingface.co/Unbabel/wmt22-cometkiwi-da) specifies source-plus-translation quality estimation: it does not require or consume the reference used by BLEU.

MT-specific research supports inspecting evaluator failures. [He et al.](https://aclanthology.org/2024.naacl-long.451/) report QE overoptimization, including off-target and length-related failures. [WALAR](https://aclanthology.org/2026.findings-acl.1682/) adds word and language alignment to address QE weaknesses in multilingual RL. These motivate tests here; they do not prove those failures occurred in our run or supply universal EN–VI thresholds.

## Reinforcement and preference strategies

Priority reflects relevance to our evidence and compute constraints, not a universal ranking.

| Strategy | What it changes or focuses on | Fit to this run | Main limitation |
|---|---|---|---|
| GRPO with lower or adaptive reference KL | Balances reward improvement against deviation from the SFT translator | **First experiment:** gradient evidence points here; existing batching and LoRA remain usable | Reward hacking remains possible; adaptive control needs a calibrated target |
| Conditional, constrained multi-objective GRPO | Separates desired quality from explicit violation budgets | **Preferred harness direction:** some criteria should become active only for applicable sources | Estimated constraints are only as reliable as their judges; more interacting controls |
| M2PO-style preference construction | Separates fidelity evidence and quality preferences; retains informative candidate contrasts | **High priority for scarce data and partial errors** | Requires reliable labels and a distinct preference-training path |
| DPO / contrastive preference optimization | Learns chosen/rejected preferences from a fixed candidate dataset | Cache annotations and remove online sampling/judging from the optimizer loop | Can overfit noisy preferences; offline data eventually become stale |
| Dr. GRPO | Removes response-length and per-group standard-deviation normalization | Test if length-dependent or low-variance noisy updates are observed | Evidence originates in reasoning; changed reduction changes gradient scale and may require retuning |
| DAPO | Asymmetric clipping, dynamic sampling, token-level reduction, truncation treatment | Lower priority: current diversity and clipping logs do not identify its main failure modes | Its removal of reference KL in long-CoT reasoning is not an MT retention result |
| Minimum-risk training / REINFORCE baseline | Optimizes expected sequence-level MT loss with a sampled baseline | A useful simpler control for whether GRPO machinery adds value | Still depends on reward quality and can have high gradient variance |
| Error-span/token-level RL | Gives localized credit using error severity | Useful if partial errors survive the improved sequence harness | Requires validated span-to-policy-token alignment and additional evaluator computation |
| Self-play curriculum | Generates tasks around the solver's learning frontier | Use later to broaden verified source coverage | Translation lacks an executable correctness oracle; proposer and judge can reinforce shared errors |
| PCGrad or separate adapters | Handles measured soft-objective or direction gradient conflict | Diagnostic/late-stage options | Additional gradient passes are costly; do not project away critical-error protections |

[Original GRPO](https://arxiv.org/html/2402.03300v3#S4.SS1) uses within-group relative advantages, clipping, and a reference penalty. [Dr. GRPO](https://arxiv.org/html/2503.20783v1) analyzes length and group-standard-deviation biases. [DAPO](https://arxiv.org/html/2503.14476v1) addresses exploration, ineffective groups, and truncation in reasoning training. Their algorithmic changes should be ablated separately; changing all of them would obscure the cause of any MT gain.

[DPO](https://arxiv.org/html/2305.18290v2) uses offline preferences rather than online generation during fine-tuning. [Contrastive Preference Optimization](https://arxiv.org/abs/2401.08417) is an MT-specific preference method with an NLL component. Its acronym **CPO** differs from **Constrained Policy Optimization** by [Achiam et al.](https://proceedings.mlr.press/v70/achiam17a.html). The latter motivates constraint-based optimization; its formal guarantees do not automatically transfer to our proposed GRPO adaptation or imperfect language judges.

[Minimum-risk training](https://aclanthology.org/P16-1159/) is an established sequence-level MT baseline. [Fine-Grained Reward Optimization](https://aclanthology.org/2026.tacl-1.33/) studies token-level xCOMET error-severity rewards; [xCOMET](https://aclanthology.org/2024.tacl-1.54/) provides error-span evidence. Neither is equivalent to broadcasting a replacement scalar across every generated token.

[PCGrad](https://arxiv.org/html/2001.06782v1) projects conflicting task gradients. Its applicability here is an experiment, not a consequence of the 42.48% ranking statistic. [Rewarded soups](https://arxiv.org/html/2306.04488v2) offers another multi-policy tradeoff approach, but model interpolation is not a semantic-correctness guarantee and is not the cheapest first experiment.

## M2PO: relevant mechanisms and remaining coverage

[M2PO](https://aclanthology.org/2026.acl-long.469.pdf), Sections 4.2–4.3, combines external/base candidates; multiplies QE by four semantic-alignment tiers (1.0, 0.7, 0.3, 0.1); fuses standardized external scores with length-normalized policy confidence using a 0.1→0.9 curriculum; and uses K/2 margin-weighted preference pairs plus listwise ranking. Its main objective is contrastive preference optimization with NLL, not our GRPO loss. The paper uses eight candidates and demonstrates preference data across EN↔ZH/DE/JA. The inspected [official repository](https://github.com/AIDC-AI/Marco-MT/tree/master/MMPO) currently presents a release-preparation README, so it is not a ready-to-run dependency in this project.

Our coverage assessment is qualitative; no defensible percentage of criteria is established for EN–VI:

| Project criterion | Expected relevance of that feedback design | Extra project-specific evidence needed |
|---|---|---|
| Partial omissions, additions, mistranslations | Strong conceptual match | Bilingual EN–VI calibration and error adjudication |
| General adequacy/fluency balance | Strong conceptual match | Independence of the judge from the selected evaluator |
| Numbers, negation, units, names | May be detected within fidelity assessment | Explicit typed checks and targeted minimal pairs |
| Glossary, register, document consistency | Requires additional task conditioning | Explicit obligations, context, and category-level scoring |
| Language, format, placeholder invariants | Cannot be assumed from a holistic alignment label | Dedicated validity checks with abstention |
| Foundation retention and optimization stability | Different purpose | Independent retention evaluation, KL and update diagnostics |

For our adaptation, keep critical-error constraints outside any growing self-confidence weight. Evaluate external confidence and policy likelihood as separate features before combining them. Generating more candidate translations for the same 950 source pairs improves preference coverage, but does not create an equivalent number of distinct semantic situations. New verified sources are still needed. No M2PO throughput claim is used to predict A100 speed or feedback latency here.

The earlier [Absolute Zero Reasoner paper](https://arxiv.org/html/2505.03335v1) trains proposer/solver roles using executable coding tasks and grounded verification. An MT self-play extension replaces that oracle with uncertain semantic assessment. Use its curriculum idea to generate useful challenges, while retaining independently validated translation supervision.

## A conditional harness suitable for this framework

### 1. Return evidence per criterion

The current [translation judge](../../src/feedback/judge.py) returns `valid`, one `score`, and text feedback. Although its prompt names several criteria, those criteria are not independently measured. Extend the design to return, per criterion: `applicable`, `value` or `unknown`, calibrated reliability, severity, evidence spans, and reason codes. Retain raw scores and the judge/model/prompt versions. A verbal confidence score is not calibrated probability.

Examples of categories are omission, unsupported addition, negation reversal, number/unit error, terminology violation, wrong language, format violation, fluency error, and uncertainty. Distinguish a translation failure from a malformed API response or unavailable evaluator. Missing feedback must not silently become either a perfect score or a translation-error penalty.

Use one schema for the API and self-hosted backends. Cache immutable annotations by source, candidate, direction, context/glossary, scorer revision, and prompt revision. Policy-dependent likelihood must be recomputed when model weights change. Use [GEMBA-MQM](https://aclanthology.org/2023.wmt-1.64/) as a reference for structured error annotation, not proof that any particular API model is a reliable EN–VI judge.

### 2. Separate obligations from preferences

Use strong, non-compensatory treatment only for verified applicable obligations: for example a declared placeholder or unambiguous numerical value, or an adjudicated major semantic error. Use bounded penalties and abstention for uncertain learned detections. Among sufficiently faithful candidates, rank semantic quality and fluency; use lexical overlap as a secondary preference unless the task requires exact terminology.

A fluent output must not outrank a faithful one solely by accumulating surface bonuses after a confirmed critical error. Implement this as an explicit candidate eligibility tier or lexicographic preference, rather than hoping arbitrary scalar weights enforce it. Record which rule changed each candidate ranking.

Conditional relaxation examples:

- Valid paraphrase: relax BLEU preference, preserve coverage and factual checks.
- Localized date or number formatting: relax string matching, preserve canonical meaning.
- Source consisting mainly of a name or code: abstain from a confident language penalty.
- Requested technical glossary: activate its obligations; do not impose the glossary globally.
- Uncertain judge disagreement: reduce the update weight or seek adjudication; do not equate disagreement with candidate error.
- All candidates fail: resample under a bounded budget or train from a separately verified correction. Do not simply keep the highest fluent failure.

### 3. Use adaptive constraints, with explicit limits

A proposed objective is:

```text
maximize expected calibrated quality
subject to E[c_j | direction, applicable source class] <= epsilon_j
           and a monitored divergence budget to the frozen SFT reference.

J = E[Q] - sum_j lambda_j * (E[c_j] - epsilon_j) - beta * D_reference
lambda_j <- max(0, lambda_j + eta_j * (EMA(c_j) - epsilon_j))
```

The costs, applicability masks, budgets, and update periods need development-set calibration. Aggregate constraints cannot guarantee every translation is correct. Keep independent checks for critical failures and track per-direction violation rates so one direction cannot hide the other's deterioration.

First compare fixed beta 0.01 against the existing 0.04. Only then test an adaptive controller. [PPO](https://arxiv.org/pdf/1707.06347), Section 4, establishes an adaptive KL-penalty baseline; applying a related controller to a **frozen SFT reference** is our proposed extension. PPO's old-policy divergence and our reference divergence have different roles.

One possible controller, not a calibrated default, is:

```text
beta_next = clip(beta * exp(eta * (EMA(D_reference) / target - 1)),
                 beta_min, beta_max)
```

Set the target from a quality-checked pilot, retain a positive lower bound, update slowly, and allow relaxation only while protected development criteria remain acceptable. If those criteria deteriorate, suspend relaxation, restore protection, or reject the checkpoint. Freeze the reference identity; do not silently re-anchor it to hide accumulated drift.

The logged k3 quantity is a sampled divergence proxy. It is not an exact full-vocabulary KL, and rollout reuse introduces an old/current sampling mismatch. Monitor it separately from `approx_kl_to_old`, clip fraction, and the occasional reward/KL gradient probe. Do not transplant a numeric target from a different estimator or temperature.

### 4. Handle group-relative normalization correctly

For current GRPO, `A_i = (r_i - mean_group(r)) / (std_group(r) + eps)`.

- Adding a constant penalty to every candidate in a group cancels exactly. All candidates being wrong in the same way can therefore produce no relative learning signal for that criterion.
- Multiplying all rewards in a group by a positive confidence factor largely cancels too, except for the epsilon effect. To downweight an uncertain group, use a detached, externally calibrated factor on the **advantage or loss**, not just on all rewards before normalization.
- An extremely small group variance can amplify differences that are smaller than judge noise. Calibrated tie handling, post-normalization group weights, or a separately tested Dr. GRPO reduction are preferable to treating tiny differences as certain.
- All-fail/tied groups need verified repair examples or bounded resampling. Repeatedly discarding difficult groups can bias the training distribution.
- External-teacher candidates are useful for preference construction. Do not insert them into on-policy GRPO rollouts with fabricated old-policy likelihoods. Use a separate preference/SFT objective or a justified off-policy correction.

These points follow from the implemented normalization and sampling contract; they are not extra claims of correctness from a new algorithm.

### 5. Keep the added compute manageable

Score cheap deterministic criteria first. Compute source-aware semantic scores next. Route selected **whole candidate groups** to expensive structured feedback; annotate all candidates in a selected group consistently. Include a random audit stream alongside disagreement-triggered selection, so easy-looking errors are not invisible. Estimate population error rates using a representative sample or known selection probabilities.

Initially use four candidates, two reuse updates, and the existing length-bucketed/logprob batching. Cache offline candidates and annotations for preference experiments. Do not add per-criterion backward passes on every step merely because metrics disagree; measure gradient conflict periodically before considering PCGrad. Dynamic KL and most rule checks require no additional policy backward pass.

Offline preferences remove online generation/judge waits from the update loop, but still require candidate preparation and policy forward/backward scoring. Multi-pair methods can score more candidate tokens. Compare end-to-end time, annotated candidates, scored tokens, and held-out gains; optimizer-step counts or another paper's H100 training duration do not establish a speedup on this A100.

## Validation and staged experiment plan

Maintain the disjoint 80/20 SFT/RL partition. Keep both directions and all augmented siblings of an original pair in the same partition. Judge-generated negatives, including deliberate number or negation errors, must be labeled as rejected candidates rather than added as correct SFT translations. Verified new sources and back-translations require their own provenance and filtering.

Start with 100 training source pairs in both directions: 200 source-direction groups. Four policy candidates plus two teacher candidates per group produce **1,200 candidate translations** to annotate and cache. This is a proposed collection size, not an authorized API job or a runtime estimate. Calibrate each criterion with blinded bilingual review, including unflagged controls, valid paraphrases/localizations, partial omissions, number/negation changes, and prompt-injection-like source text.

| Order | Comparison | Purpose |
|---|---|---|
| 0 | Saved SFT base and checkpoint 2800 on the new structured error audit | Establish which errors the existing proxies miss |
| 1 | Beta 0.04 versus 0.01; same checkpoint, seed, data, reward, LoRA, and update budget | Test the leading measured optimization hypothesis |
| 2 | Selected fixed beta versus adaptive reference-KL control | Test conditional relaxation without changing the evaluator |
| 3 | Add calibrated fidelity evidence, then applicability-aware validity checks separately | Measure each reward-policy contribution and false-positive cost |
| 4 | Fixed-annotation preference baseline versus multi-pair preference training on identical candidates | Test better use of scarce labels without conflating a new judge |
| 5 | Group-std and response-length reduction ablations, then complete Dr. GRPO if justified | Distinguish normalization effects from reward design |
| 6 | Relaxed drift bound or direction-specific adapters only if direct diagnostics support them | Avoid solving an unmeasured bottleneck |

A 200-update pilot can screen functionality and gross regressions; it cannot establish convergence. Use repeated seeds and longer confirmation only for promising arms. For online RL, report the complete rollout-plus-reuse time; compare methods by sampled/scored tokens and wall time as well as updates.

Evaluate clean EN→VI and VI→EN separately, macro averages, corpus BLEU/chrF++, a reference-based semantic evaluator when available, semantic-error severity, exact obligations, and a blinded human subset. COMETKiwi used as reward should not be the sole checkpoint selector. Hold the 3,500-pair test set for final selection-free reporting; the validation set is only 250 pairs, so repeated tuning can overfit it.

Publish both a **fixed audit score** with an immutable scorer definition and the changing training objective. Log criteria applicability, missingness, confidence/calibration, false-positive rates, candidate ranking changes, conditioned penalties, direction, length, cap hits, and scorer/API latency. Schedule changes must be versioned; a higher reward after changing weights is not comparable to the earlier curve.

## Proposed repository backlog

| Priority | Work | Acceptance evidence |
|---|---|---|
| P0 | Structured criterion result schema shared by API/self-hosted judges | Missing/unknown handling; typed severity and spans; scorer/prompt versioning; no scalar success fabricated on parsing failure |
| P0 | Offline replay and bilingual calibration set | Applicable/inapplicable cases; per-direction category precision/recall; valid-alternative false-positive analysis |
| P1 | Conditional reward harness around `TranslationReward` | Provenance for every ranking; severe-error non-compensation; canonical number/placeholder handling; separate fixed audit metrics |
| P1 | Post-normalization group weighting and all-fail repair routing | Demonstrate constant-penalty cancellation, confidence weighting, tie behavior, and bounded resampling in focused CPU tests |
| P1 | Checkpointed adaptive reference-KL controller | EMA and coefficient state restored on resume; fixed reference; separated old-policy divergence logs; controller ablation |
| P2 | Offline preference datamodule and objective | Teacher/current candidate provenance; consistent tokenization; cached labels; on-policy GRPO remains separate |
| P2 | Error-span/local reward prototype | Unicode-aware offsets and policy-token alignment; omissions handled explicitly; measured scoring and backward overhead |
| P3 | Direction adapters, PCGrad, or broader self-play | Direct evidence of direction/soft-objective interference; comparison against simpler controls |

The survey proposes interfaces around [reward scoring](../../src/models/components/translation_reward.py), [GRPO advantages/loss](../../src/models/components/grpo.py), [the Lightning module](../../src/models/grpo_module.py), [reward diagnostics](../../src/models/components/reward_diagnostics.py), and [feedback](../../src/feedback/judge.py). No proposed controller, new reward policy, or preference objective has been activated by this research work.
