# From MQM observations to criterion-specific translation gradients

Date: 16 November 2026. Decision: develop proposal 1, the
[source-grounded MQM-style error vector](sac_grading_engines_20261115.md#1-source-grounded-mqm-style-error-vector).
Status: mathematical design with a numerically verified reference calculation.
The training engine, feedback parser and prompts in production are unchanged.
No judge calls, model training or GPU rental were performed for this design.

The [consolidated training scheme](translation_training_scheme_20261116.md)
places this objective in the complete data, calibration, experiment and evaluation
procedure, including the already-inspected official benchmark's limitations.

**Recommendation:** retain the error vector through validation, local repair
contrasts, token coefficients and separate parameter gradients. Use verified
meaning-preserving edits to teach the missing action. Measure conflicts before
forming the final parameter update, and protect adequacy against surface-quality
tradeoffs. A fixed weighted sum of sentence rewards cannot supply these distinctions.

This is a proposed adaptation for EN→VI and VI→EN, not an established result on
PhoMT. Numeric objective settings in the verifier are synthetic examples.

The [penalty audit](translation_penalties_audit_20261116.md) distinguishes active
controls from proposed negative supervision and retention. In particular, the
anchored trainer currently requires KL beta zero; retention needs implementation,
and the saved checkpoint callback still selects by reference-dependent chrF.

## 1. What can actually be differentiated

Let the translation policy be

\[
 p_\theta(y\mid x)=\prod_{t=1}^{T}p_\theta(y_t\mid x,y_{<t}).
\]

A black-box MQM judge returns discrete observations
\(e_k(x,y)\): criterion, severity, error span, applicability and unresolved
context. There is no usable pathwise derivative through its API, decoded text,
or sampled token IDs. Making `severity` a floating-point tensor does not create
one. The observation must instead specify a **differentiable training objective**.

There are two valid routes, with different requirements:

1. **Policy gradient:** discrete criterion reward \(R_k(x,y)\) can train the
   policy through the score-function identity:

   \[
   \nabla_\theta\mathbb E_{y\sim p_\theta}[R_k(x,y)]
   =\mathbb E[(R_k(x,y)-b_k(x))\sum_t\nabla_\theta\log p_\theta(y_t\mid x,y_{<t})].
   \]

   Here the judge and source-only baseline are detached. An action-dependent
   baseline requires a valid estimator of its own. The identity assumes samples
   from the stated policy and a fixed reward function; arbitrary replay, clipping,
   decoding filters and length-normalized surrogates need separate treatment.
   This route needs no corrected translation, but a terminal reward still provides
   weak localization and may not expose a good action in an all-failed group.

2. **Verified contrast supervision:** an observed error identifies a rejected
   phrase and a verified replacement under the source. Teacher-force both complete
   sequences and differentiate their selected token log probabilities. This supplies
   a concrete action, including inserted information for omissions. It is the
   recommended first implementation given this project's correction branch.

An arbitrary terminal-error mask is **not** an unbiased policy-gradient estimator
of expected MQM. Earlier choices can cause later errors. The localized objective
below is explicitly a supervised surrogate; its utility must be tested rather
than justified by calling each masked coefficient a true action advantage.

## 2. Why merely keeping K rewards is insufficient

For fixed weights \(w_k\), common token masks and linear aggregation:

\[
 \sum_k w_k\nabla_\theta\mathbb E[R_k]
 =\nabla_\theta\mathbb E[\sum_k w_kR_k].
\]

Separate logging or separate backward calls followed by an unchanged weighted
sum do not defeat this equivalence. Criterion-specific centering/scaling can
change coefficients, but can still reduce to one effective scalar per completion
if every token receives the same resulting coefficient. Shared PPO clipping also
does not magically establish criterion-specific credit.

The proposed difference is **where supervision acts and how conflicts are
handled**:

- Each criterion selects different validated target actions and token positions.
- Adequacy errors determine eligibility for a surface improvement; fluency cannot
  compensate for a mistranslation by increasing its weight.
- Criterion gradients remain separate for interference measurement and, when
  necessary, a constrained update rather than an unconditional weighted sum.

The optimizer necessarily applies one displacement in parameter space. The aim
is to preserve information until that decision, not to avoid producing a final
update or every scalar loss anywhere in the system. A final differentiable loss
can still be scalar while carrying distinct token-level supervision.

For example, criterion coefficients `[+1, 0]` and `[0, -1]` can sum to
`[+1, -1]`: a zero sentence sum with a nonzero learning signal. One sentence
coefficient cannot represent that distinction under a uniform token mask.

## 3. Define the observation before defining its cost

Use atomic error events plus explicit criterion coverage. Suggested primary
criteria are omission, unsupported addition, mistranslation, negation/scope,
numbers, entities, target language, fluency, orthography, terminology and format.
Treat adequacy as a derived summary over semantic events rather than a second
charge for every semantic error.

For each event retain:

\[
 e=(\text{id},k,s,\text{source span},\text{candidate span},
     \text{reason},\text{interpretation status}).
\]

Severity \(s\in\{\text{minor,major,critical}\}\) is ordinal. `unknown` and
`inapplicable` are separate states, not points on this severity scale. A dimension
grade of 4 is not four times as good as a grade of 1, and its difference is not
automatically a valid policy advantage.

The grading envelope can retain `sac-grade-v1` with a versioned
`details.mqm_vector_version`. Extend proposal 1 with a coverage record for every
criterion, event identifiers, and exact raw-text span occurrence identifiers.
An empty errors list alone must never imply that every criterion was checked.
The controller resolves quotes to character offsets and validates them; do not
trust judge-generated subword indices. Overlapping kinds describing the same
defect get one primary owner and secondary diagnostic tags. For example, a
wrong digit is not three independent penalties under numbers, mistranslation
and adequacy. Distinct real defects may overlap and need adjudication.

Additional instruction for the chosen grader, composed with the existing common
policy and source-grounded role:

```text
Report coverage for every requested criterion as assessed, unknown, or
inapplicable. Unknown coverage needs a short explanation. Every clear error has
an event_id and one primary kind; use secondary_tags for overlapping descriptions
of the same defect. Preserve the common error fields and exact quoted spans.
For a repeated quote identify its occurrence in the raw input. Omission has no
candidate span; unsupported addition may have no source span. Do not invent
token indices, a numeric training reward, a corrected target, or calibrated
confidence. Put these additions in details, with mqm_vector_version=1.
```

This is a new research prompt revision, not a compatible runtime response to
`parse_sac` or the current `parse_criteria`. Version its prompt, parser and cache
together. Grading remains engine 1; repair construction below is a training-data
operation and does not make the original judge its own truth validator.

## 4. Connect observations to learnable model behavior

There is no known fixed partition of NLLB weights into an omission module, a word
choice module and a capitalization module. A token mask selects **loss sites**;
its gradient can reach encoder/decoder attention, embeddings and output weights
through the network. Restricting the loss site does not restrict all affected
parameters, or prove a causal localization of the model's defect.

| Observation | Behavior to teach | Eligible supervision and extent | Failure to avoid |
|---|---|---|---|
| Orthography / capitals | Target-form choice in the actual sentence or name context | Verified casing/spelling variant; aligned affected subwords | Mechanically capitalizing every subtitle fragment or copying English name conventions |
| Fluency / literal calque | Idiomatic collocation, syntax and target ordering with unchanged meaning | Minimal phrase/clause rewrite with semantic preservation verified | Rewarding a fluent paraphrase that drops a qualifier |
| Contextual wrong sense | Source-conditioned lexical choice | Replaced phrase and any affected agreement/ordering | Declaring one dictionary sense correct without the missing context |
| Omission | Generate a source-supported unit at an appropriate boundary | Inserted tokens plus the original divergent continuation or EOS | Placing a penalty on a nonexistent generated span |
| Unsupported addition | Continue without the unsupported content | Deleted region on original; next valid continuation or EOS on repaired side | Encouraging premature stopping as a universal deletion strategy |
| Negation / scope | Preserve polarity and its scope, participants and conditions | Full affected relation/clause, not necessarily only the word “not” | Fixing a negation token while leaving the opposite scope |
| Numbers / entities | Preserve identity/quantity under allowed localization | Verified semantic contrast and surrounding units | Treating `1,000` / `1.000` or transliteration as automatic mismatch |
| Target language | Generate the required language under the source task | Verified clause or full-sequence contrast | Flagging valid names, formulas or borrowed words as wrong language |
| Terminology / format | Honor a supplied glossary or explicit output contract | Bounded edit only when the obligation exists | Learning an arbitrary preferred term as universally mandatory |
| Unknown / inapplicable | No assertion about that criterion | No criterion update; retain coverage accounting | Converting missing context into a positive anchor or a negative reward |

For source ambiguity, a local correction can be eligible only if it holds across
the unresolved interpretations. Otherwise abstain. A locally verified semantic
repair can improve one error while other errors remain; this must not certify
the whole repaired sentence as a globally faithful anchor. In the first pilot,
surface-only edits require resolved adequacy assessment and no known semantic
defect. Later relaxation would be a separately audited intervention.

Do not first add criterion-specific LoRA modules or classification heads. A
head trained to recognize errors may learn grading without improving generation.
A frozen text critic still has no path through sampled token IDs. Relaxed tokens
or straight-through estimators introduce bias and input-distribution mismatch.
Adapters could impose a useful partition later, but their names do not establish
disentangled capabilities; all required adapters must also be active or routed
without an oracle error label at inference.

## 5. The proposed differentiable bridge

### 5.1 Build a valid contrast

For source \(x_i\), original output \(y_i^-\), criterion \(k\), and independently
verified minimal repair \(y_{ik}^+\), construct masks
\(m_{ikt}^{+}\) and \(m_{ikt}^{-}\) on **their own complete token sequences**.
Do not splice a phrase into the original token IDs and reuse the old suffix
probabilities. Both variants require fresh teacher forcing with their own prefixes.
The input includes only the context that the translation policy actually has.
Judge-only document context must not create a mandatory target distinction the
policy cannot observe; supply that context to both or withhold that training label.

The repair provider can be a bilingual reviewer, a carefully checked deterministic
transformation, or a bounded editor followed by an independently calibrated
verification process. A second prompt to the same model may share the first
model's errors; it is not independent evidence by designation. No candidate is
promoted merely because it is least bad in the group or is the PhoMT reference.

Let \(a_{ik}\) be the binary eligibility decision after that verification,
\(r_{ik}\in[0,1]\) an empirical reliability weight from held-out reviewed evidence,
and \(v_{k,s}>0\) a bounded severity cost chosen on development data:

\[
 w_{ik}=\operatorname{stopgrad}(a_{ik}r_{ik}v_{k,s}).
\]

Reliability is not the judge's self-reported confidence or a COMET threshold.
Without sufficient evidence, the contrast stays review-only or is withheld;
inventing a probability does not solve missing calibration. Severity costs need
not equal MQM reporting penalties. Preserve event ownership when several criteria
refer to one repair; split attribution explicitly or use one primary owner.

For the equations, use one selected verified event per source and criterion per
batch. Rotate among additional events with recorded, controlled sampling, or
average them explicitly within `(source, criterion)`. Do not let duplicate events
silently multiply a source's influence.

### 5.2 Define the criterion losses

Write token log probabilities as
\(\ell_{ikt}^{\pm}=\log p_\theta(y_{ikt}^{\pm}\mid x_i,y_{ik,<t}^{\pm})\).
For every active contrast, both mask masses must be positive:

\[
 M_{ik}^{\pm}=\sum_t m_{ikt}^{\pm}>0,\qquad
 s_{ik}^{\pm}=\frac{\sum_t m_{ikt}^{\pm}\ell_{ikt}^{\pm}}{M_{ik}^{\pm}},\qquad
 \Delta_{ik}=s_{ik}^{+}-s_{ik}^{-}.
\]

This is a comparison of local mean log probabilities, not a normalized
probability of an entire translation or a DPO derivation. It deliberately gives
an event bounded local exposure rather than scaling its weight by the number of
Vietnamese syllables or subword pieces. Insertion/deletion asymmetries and suffix
context still introduce bias; compare mean versus sum/window alternatives in
development rather than claiming universal length invariance.

Use a soft preference target \(q_k\in(1/2,1)\), positive temperature parameter
\(\beta_k\), nonnegative anchor weight \(\alpha_k\), and NLL floor
\(\tau_k\ge0\). The vector objective is

\[
 \mathbf L(\theta)=(L_1(\theta),\ldots,L_K(\theta)),
\]

\[
 \boxed{
 L_k=\frac1N\sum_i w_{ik}\left[
  \operatorname{BCEWithLogits}(\beta_k\Delta_{ik},q_k)
  +\frac{\alpha_k}{2}\big[{-s_{ik}^{+}-\tau_k}\big]_+^2
 \right].}
\]

Here \([z]_+=\max(z,0)\); its square is continuously differentiable, including
at zero. Discrete masks, labels, weights and repair strings are frozen during
the update. The log probabilities and therefore \(L_k\) are differentiable with
respect to policy parameters. `BCEWithLogits` must use its numerically stable form.
The bounded local NLL term supplies the action without indefinitely reinforcing
an already sufficiently likely repair. It is not bounded above as a loss; weights,
gradient norms and step size still need controls.

\(N\) is the number of scheduled directional source records, including records
with no eligible contrast. Keep that denominator fixed through microbatching
and distributed reduction. Renormalizing by the small surviving sum of weights
would erase a deliberate reduction in uncertain evidence. For two directions,
choose and record their scheduled mixture rather than letting different eligibility
rates accidentally determine the direction weights.

\(q_k\) is a chosen/calibrated noise allowance, not an automatically known
probability from a severity label. With no active anchor term, the pair term has
finite optimum \(\beta_k\Delta=\log(q_k/(1-q_k))\). The active NLL term can move
that optimum. All hyperparameters require ablation; the older scalar `0.02`
noise floor does not calibrate any of these parameters.

### 5.3 Explicit chain rule into the translation model

Define

\[
 u_{ik}=\beta_k[\sigma(\beta_k\Delta_{ik})-q_k],\qquad
 h_{ik}=[-s_{ik}^{+}-\tau_k]_+.
\]

The per-criterion token coefficients are

\[
 c_{ikt}^{+}=\frac{w_{ik}}N
       (u_{ik}-\alpha_kh_{ik})\frac{m_{ikt}^{+}}{M_{ik}^{+}},\qquad
 c_{ikt}^{-}=-\frac{w_{ik}}N
       u_{ik}\frac{m_{ikt}^{-}}{M_{ik}^{-}}.
\]

With repairs below the desired margin, the pair term encourages the selected
repair actions and discourages their rejected alternatives. It has no direct
coefficient on unrelated target positions. Shared parameters can nevertheless
change unrelated probabilities; a token mask does not guarantee behavioral isolation.

The desired mathematical relationship is

\[
 \boxed{
 g_k=\nabla_\theta L_k
 =\sum_{i,t,\pm}c_{ikt}^{\pm}\nabla_\theta\ell_{ikt}^{\pm}
 =J_\theta^{\mathsf T}c_k .}
\]

For a particular target-token logit \(z_{tj}\),

\[
 \frac{\partial\ell_t}{\partial z_{tj}}
 =\mathbb 1[j=y_t]-p_\theta(j\mid x,y_{<t}).
\]

Thus the MQM observation selects an action contrast and its coefficients;
the policy Jacobian determines which parameters change. No semantic neuron
assignment and no derivative through the judge are required.

In a practical PyTorch implementation, stack the K scalar losses and compute
their vector-Jacobian products separately. Masks and metadata have no graph.
For the existing memory-efficient scorer, coefficients can be evaluated on a
small detached token-score leaf, detached again, and replayed against fresh
policy forwards. This reproduces first derivatives **at the same parameters
and scoring state**; it is not a higher-order differentiable solver. Disable
dropout or replay identical randomness for coefficient and backward passes.
Do not update parameters between the two passes.

## 6. Omissions, deletion and token alignment are part of the objective

A missing phrase has no target token. For insertion, mask the new phrase and,
where needed, its first continuation on the repaired side. On the original side,
mask the first divergent continuation action, or EOS if it ended there. For a
deletion, include the rejected extra material and its divergent boundary; mask
the repaired continuation or EOS on the repaired side. An active contrast with
an empty side is rejected, not repaired numerically by dividing by `max(1, M)`.

Example with a source-supported time:

| Input | Text | Local supervision |
|---|---|---|
| Source | `The meeting starts at 9 a.m.` | Meaning obligation includes the time |
| Original | `Cuộc họp bắt đầu.` | Original ending action at the divergence |
| Verified repair | `Cuộc họp bắt đầu lúc 9 giờ sáng.` | Inserted time phrase and correct ending boundary |

For a casing error, `cuộc` → `Cuộc` can be a small target-form contrast when the
contract establishes a sentence start. For the previously observed calque
`ends up reflecting` → `kết thúc bằng cách phản ánh`, a verified replacement
must respect the complete source; it cannot be prescribed from the isolated
English phrase in every context.

Alignment requirements before any transformer pilot:

1. Resolve exact quoted occurrences to raw Unicode character intervals. Preserve
   normalization provenance; Python code points, UTF-8 bytes and JavaScript UTF-16
   units are not interchangeable offsets.
2. Align each original/repaired text with its actual tokenizer revision and token
   IDs. Retokenized decoded output may differ from the sampled IDs under
   SentencePiece normalization. Check the correspondence; reject or explicitly
   align differences rather than assuming round-trip identity.
3. Include every subword overlapping the affected interval and any validated edit
   boundary. Vietnamese whitespace units are not English-style word boundaries.
4. Score each complete sequence with the correct shifted decoder inputs. Exclude
   prompts, padding and NLLB language-prefix tokens. Include EOS only when it is
   an actual selected boundary action, never as padding.
5. If a semantic edit reorders a clause, expand to the verified scope or use a
   declared sequence-level contrast. Minimal string distance is not a causal
   guarantee that only one word matters.

The NumPy verifier uses prevalidated synthetic masks. It does **not** implement
or validate these tokenizer/linguistic alignment requirements.

## 7. Keep the vector until the parameter update

Initially measure criterion gradient norms and cosines:

\[
 C_{kj}=\frac{g_k^{\mathsf T}g_j}{\|g_k\|\|g_j\|}.
\]

Negative cosine signals a local surrogate conflict; it is not proof that two
translation capabilities are inherently incompatible. Do not unit-normalize a
tiny, unreliable gradient to make it as strong as well-supported evidence.

A transparent local update rule is a joint projection in the space of **actual
parameter displacements**. Let \(d_0\) be the proposed displacement, including
the optimizer's learning rate, preconditioning, momentum and weight decay. For
protected semantic criterion gradients \(g_p\), solve

\[
 \boxed{
 d^*=\arg\min_d \tfrac12\|d-d_0\|_2^2
 \quad\text{s.t.}\quad
 g_p^{\mathsf T}d\le0\ \ (p\in\mathcal P),\qquad \|d\|_2\le\rho .}
\]

Then apply \(\theta\leftarrow\theta+d^*\). Zero is feasible. Some conflicting
batches will stall; that is preferable to declaring a fluency gain a semantic
improvement. It also signals where the labels or proposed objectives need review.
The trust radius \(\rho\) must be selected on development evidence, not taken
from the synthetic verifier.

Why this is a meaningful protection, and its limits:

- For each protected **batch surrogate**, Taylor expansion gives
  \(L_p(\theta+d)\approx L_p(\theta)+g_p^{\mathsf T}d\). The constraint prevents
  a first-order increase for that surrogate. It does not guarantee better actual
  translations, judge correctness, per-example preservation or finite-step safety.
- Protect criteria jointly. One pass of pairwise projections can reintroduce a
  conflict removed earlier. PCGrad is an informative comparison, not a guarantee
  that every criterion improves on each step.
- Projecting raw gradients and then calling Adam can violate the constraints:
  its state and preconditioning change the displacement. A first correctness
  prototype should use plain SGD, or explicitly project a proposed optimizer
  displacement and define its state/rollback behavior. Do not quietly feed a
  projected gradient into the existing Adam trainer and claim protection.
- A zero semantic gradient supplies no protective information. In particular,
  a KL retention loss has zero gradient at exact equality to its reference while
  nonzero moves can increase it through curvature. A norm cap, finite-step
  divergence checks and fixed clean retention probes remain necessary.
- Averaging gradients across ranks before the constraint solve is required.
  Per-rank projection then all-reduction is a different algorithm. Gradient
  accumulation must finish before measuring conflicts or projecting.

Use scheduled clean source–target retention examples whose fidelity was checked,
with a separately measured contribution. A frozen-reference distribution can
help limit drift but may itself encode the starting errors; do not require the
very edited position to reproduce a known bad output. Finite-step probe rejection
must restore parameters **and optimizer state**; document scheduler/counter
semantics. These checks limit observed regressions, not all possible regressions.

For an initial research comparison, use the same vector/token loss with a fixed
weighted-sum SGD update as the transparent baseline, then add projection. This
separates the benefit of better supervision from the benefit or cost of conflict
handling. Do not attribute both to “vector reward.”

## 8. Integration with this repository

The relevant bottlenecks are explicit:

- [`CriterionJudge`](../../src/feedback/criteria.py) returns category values and
  textual evidence but not a validated event/coverage/repair contract. It needs
  a versioned parser and span validator for the chosen MQM design.
- [`CorrectionEvidence`](../../src/models/components/correction_evidence.py)
  currently routes judgments to pass/reject/unknown. Preserve this for old
  experiments; the new path must retain category and event information.
- [`correction_loss`](../../src/models/components/correction_objective.py)
  accepts per-sequence mean log probabilities. A localized path must accept
  token scores and per-criterion masks instead of reducing first.
- [`AnchoredGRPOModule._backward_rollout`](../../src/models/anchored_grpo_module.py)
  already forms detached token-score coefficients and replays microbatches. Extend
  this structure with an unreduced criterion axis; keep separate gradient buffers
  through any conflict analysis. One merged backward pass erases that information.

The proposed flow is:

```mermaid
flowchart LR
  A[Source and generated candidate] --> B[MQM error events and coverage]
  B --> C[Validated evidence and eligible repairs]
  C --> D[Own-tokenizer masks on both full sequences]
  D --> E[Vector of criterion losses]
  E --> F[Criterion token coefficients]
  F --> G[Criterion parameter gradients]
  G --> H[Measured conflicts and joint update rule]
  H --> I[Finite-step checks and update record]
```

The first implementation should be an opt-in objective, with no production
defaults changed. Off-policy repairs stay supervised contrasts; they must not
receive old-policy GRPO likelihood ratios. Retain ordinary sequence-level RL
only as a separately specified control or branch with explicit candidate
eligibility, never an opposing hidden signal on the same repair pair.

Per-criterion transformer gradients increase cost: retain/recompute backward
passes and potentially K parameter-sized buffers. On NLLB 1.3B full fine-tuning,
one dense FP32 gradient is about 5.2 GB before framework overhead; K such buffers
are not credible on a 24 GB 4090 alongside model/optimizer states. Prefer a
measured low-rank-adapter prototype or a few explicitly declared objective
families, with streamed dot products/recomputation where exact. Adapter gradients
reduce memory without proving semantic disentanglement. A coefficient-space
cosine is not a parameter-gradient cosine: the policy Jacobian changes the metric.
Do not substitute one and claim exact gradient protection.

Store annotation revision, prompt/model revisions, source/candidate hashes,
direction, actual context/contract, event ownership, repaired sequence and
verification provenance, tokenizer/mask revision, objective configuration,
denominators, gradient norms/cosines and accepted displacement. Checkpoint any
adaptive controller and optimizer state. Old scalar/criterion caches cannot be
silently reinterpreted as this new evidence.

## 9. Verification performed and experiment needed

Run the local numerical reference:

```powershell
python scripts/tools/verify_mqm_vector_objective.py
```

The [script](../../scripts/tools/verify_mqm_vector_objective.py) implements the
loss vector, analytic token coefficients, a tiny shared-softmax actor Jacobian
and a small active-set displacement projection. Its
[saved output](../../diagnostics/mqm_vector_20261116/verification.json) records
16 checks. Central finite differences agree with the token coefficients to
approximately `2.05e-11` maximum absolute error, and with the shared actor's
parameter gradients to approximately `4.08e-11`. Source microbatch and dense
gradients agree exactly for that synthetic example.

Additional checks cover zero update for abstention/no eligible repairs, distinct
criterion token support, an omission insertion boundary, empty-active-mask
rejection, fixed scalarization equivalence, nonzero localized coefficients despite
scalar cancellation, a finite soft-preference margin, a joint-projection example,
and counterexamples to single-pass projection, preconditioned-gradient protection
and zero-gradient retention guarantees. These are NumPy analytic/numerical checks,
**not PyTorch autograd, tokenizer integration or translation-quality tests**.

Before training, the meaningful integration checks are:

1. Exact PyTorch autograd agreement for each criterion, decoder shifting, prefix,
   padding/EOS masks, abstention and changed-prefix scoring on a tiny real model.
2. Equality of dense versus microbatched criterion gradients at fixed scoring
   state; distributed-global versus explicitly concatenated denominators.
3. A real insertion/deletion/casing contrast with Unicode and SentencePiece
   offsets verified against actual NLLB token IDs. Include a normalized
   round-trip mismatch that must fail explicitly.
4. Measured optimizer displacement obeys the chosen joint constraints within
   tolerance, including accumulated gradients and resumed state. Restore state
   correctly on a rejected finite-step probe.

For a translation experiment, compare under matched sources, decoding, update
tokens, starting checkpoint and feedback budget:

| Arm | What it isolates |
|---|---|
| Existing scalar correction | Current control |
| MQM-verified targets with sequence-level contrast | Better evidence/anchors without token localization |
| Same targets with criterion-specific local losses and fixed sum | Added value of localization |
| Same local losses with measured joint gradient handling | Added value and coverage cost of conflict handling |

Obtain bilingual reviewed **candidate contrasts**, not only reference grades.
Use training data for repair calibration, disjoint development sources for
selection/thresholds, and a newly sealed eligible set for confirmatory evaluation.
The official test remains a benchmark whose references and some predictions have
already been inspected; it must not be described as newly untouched. The
existing 240-item blank training review worksheet is a sampling scaffold, not
labeled calibration evidence. Full-corpus COMETKiwi means and the earlier
`0.02` replay sensitivity do not estimate MQM error reliability.

Report separately by direction: omission/addition/negation/numeric error rates,
wrong-sense and calque rates, capitalization/orthography defects where applicable,
and blinded adequacy/fluency judgments. Include repair validity, localization
precision, unknown/abstention rates, gradient conflicts, projected/stalled steps,
and effective exposure by length/context stratum. Report intervals for reviewed
error rates and paired differences, clustered by source when multiple candidates
share it. Independent COMET variants may supplement review; the trained proxy's
increase alone is not a success condition. Predeclare adequacy noninferiority
and surface improvement margins after a pilot supplies defensible variance and
sample-size estimates, before selecting the winning arm.

## 10. Primary research and scope of support

- [Fine-Grained Reward Optimization for MT using Error Severity Mappings
  (Ramos et al., TACL 2026)](https://aclanthology.org/2026.tacl-1.33/)
  studies XCOMET error spans and token-level reward mappings with NLLB and LLM
  translation models. Its language pairs include English/German and
  English/French, not Vietnamese. It motivates local feedback, not the reliability
  of our MQM prompt or this repair loss.
- [RLMEC (Chen et al., Findings ACL 2024)](https://aclanthology.org/2024.findings-acl.338/)
  uses minimal-edit rewriting for token-level supervision and an imitation term
  in reasoning tasks. It motivates action-bearing corrections; its results do
  not establish translation fidelity or our alignment/verification procedure.
- [Contrastive Preference Optimization for MT
  (Xu et al., ICML 2024)](https://proceedings.mlr.press/v235/xu24t.html)
  motivates preference contrasts and questions automatic trust in human reference
  translations. Our masked, noise-aware local mean-log-probability surrogate is
  not a reproduction of CPO or its derivation.
- [Gradient Surgery for Multi-Task Learning
  (Yu et al., NeurIPS 2020)](https://arxiv.org/html/2001.06782v4)
  studies conflicting task gradients and PCGrad. It motivates measuring conflicts;
  the joint displacement projection here is our proposed control, not PCGrad
  and not an imported guarantee of all-criterion improvement.
- [Constrained Policy Optimization
  (Achiam et al., ICML 2017)](https://proceedings.mlr.press/v70/achiam17a.html)
  provides a constrained-RL formulation with its own assumptions and guarantees.
  The local surrogate constraints proposed here do not inherit those guarantees.

The differentiable relationship and finite-difference checks are established for
the defined mathematical surrogate. Whether its labels, repairs, masks and
updates improve translation remains an empirical question.
