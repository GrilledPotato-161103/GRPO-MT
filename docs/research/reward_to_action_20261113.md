# Turning failed translations into useful updates

November 13, 2026: proposal and offline audit. November 14 update: the hybrid
objective is implemented and locally tested; see the
[execution and resource plan](../correction_training.md). Translation-quality
improvement remains unvalidated. No rental or paid feedback request was started
for this work. The signed-SAC profiles remain experimental reward-shaping ablations.

**Recommendation:** use failed translations as rejected examples paired with a
verified faithful translation. Train the pair with a contrastive objective and
teacher-forced likelihood on the verified translation. Retain GRPO initially only
for source groups without rejected candidates, so the two objectives cannot give
opposing sequence-level coefficients to the same failed candidate.

This changes the reward-to-action mapping. Changing the sign or sigmoid of a
scalar while retaining ordinary group centering cannot supply a missing correct
translation or reliably make every semantically bad sample a negative update.

## What the current model receives

The judge is outside the policy computation graph. In
`src/models/grpo_module.py`, scalar rewards become
`group_advantages(rewards, groups)`, then `_backward_rollout` passes those
advantages to `grpo_loss`. Feedback text is not a policy input or a corrected
training target. In outcome-supervised GRPO, the same completion advantage is
assigned to each generated token. This is the published construction in
[DeepSeekMath, section 4.1.2](https://arxiv.org/html/2402.03300v3#S4.SS1.SSS2).

For one source and four candidates:

\[
A_i = \frac{r_i-\bar r}{\operatorname{std}(r)+10^{-4}}.
\]

At the old policy, with clipping inactive and KL disabled, our loss derivative
with respect to each scored token log probability is
`-A_i / (number_of_candidates * completion_length_i)`. Thus:

- A positive advantage supplies a coefficient encouraging that sampled sequence.
- A negative advantage supplies a coefficient discouraging it, including its
  correct tokens. Different token contexts still have different parameter
  gradients; identical coefficients do not mean identical parameter updates.
- A tied group contributes no reward-policy gradient.
- Shared parameters, optimizer state, clipping and KL mean these signs do not
  guarantee an actual increase/decrease of each sequence probability after a step.

For rewards `[-1, -.5, -.5, -.5]`, the advantages are approximately
`[-1.73, +.577, +.577, +.577]`. All candidates can be unacceptable and three still
get positive coefficients. This is valid relative improvement, not an
implementation bug. It does not express the absolute instruction “all four fail.”

Subtracting a verified-anchor score **before another group normalization also
cancels**: `center(r - anchor) == center(r)`. Positive affine rescaling likewise
cancels apart from the denominator epsilon. Nonlinear shaping can change relative
weights, but does not introduce an acceptable action. The judge's sigmoid
derivative is not backpropagated into the translation policy.

An omission is especially revealing: an omitted phrase has no generated token
on which to place a local token penalty. A corrected target explicitly supplies
that phrase and the contexts in which to generate it.

## Evidence from the last downloaded run

Source: `runs/rtx4090-60341320-final-metrics-20261112/metrics/run/rl/diagnostics/rollouts.jsonl`
in the parent workspace. There are 397 fresh rollouts, 6,352 directional source
groups and 25,408 candidates. “Negative” here means **the existing direct judge
rated Low Partial match or No match**, not independently verified human error.

| Observation under the historical fused reward | Count |
|---|---:|
| Negative candidates with positive reconstructed advantage | 2,274 / 5,812 (39.13%) |
| Groups with four negative candidates | 538 / 6,352 (8.47%) |
| All-negative groups with at least one positive advantage | 517 / 538 (96.10%) |
| All-negative groups with zero reward-policy advantage | 21 / 538 |
| Groups tied on direct SAC category | 2,739 / 6,352 (43.12%) |
| Groups with both accepted-category and negative candidates | 2,106 / 6,352 |
| Groups with no Full match candidate | 1,962 / 6,352 (30.89%) |

The two opt-in signed-SAC counterfactuals, linear and centered inverse sigmoid,
both leave **275 negative candidates with positive advantages**. Both make
416/538 all-negative groups tied, while the other 122 still promote a less-bad
candidate. On a group containing only two negative severity levels, any strictly
ordered remapping is affine on those two values; after group normalization its
advantages are effectively the same, except for epsilon scaling. Neither formula
adds a correction target.

These are fixed-score replays, not alternative training runs. They cannot prove
which tokens were wrong, how actual parameter updates changed predictions, or
that the proposed objective improves translation. The archive contains no
source/prediction text or error-span alignments. The all-negative groups are only
8.47% of groups, so this mechanism alone cannot explain every flat evaluation.

Reproduce from `GRPO-MT` with the existing Conda environment:

```powershell
../venv/python.exe diagnostics/reward_action_20261113/analyze.py
```

The [JSON result](../../diagnostics/reward_action_20261113/analysis.json) includes
input SHA-256, direction-specific counts and actual-autograd checks of the current
GRPO loss. The checks cover all-negative/tied groups, constant-anchor cancellation,
zero scalar loss with nonzero gradients, and chosen/rejected score derivatives.

## What the literature supports

| Approach | How failure changes the model | Assessment for this project |
|---|---|---|
| Outcome GRPO | Reweights sampled token log probabilities relative to peers | Useful when samples contain meaningful quality variation; no absolute acceptance boundary |
| Fixed acceptance-baseline policy gradient | Uses `(reward - threshold) / fixed_scale` instead of the group mean | Can give all failures negative coefficients; still offers no explicit correction and changes the estimator/variance |
| DPO / CPO | Contrasts a chosen translation against a rejected one | Explicit relative action; requires a genuinely better chosen target |
| Correction likelihood plus contrastive learning | Learns the replacement while distinguishing it from failures | Preferred first experiment given available parallel data |
| Segment/token feedback | Routes reward to annotated parts of output | Promising later; requires trustworthy error localization, including a strategy for omissions |

[DPO](https://arxiv.org/abs/2305.18290) learns chosen-versus-rejected preferences
using a reference policy. [CPO for translation](https://arxiv.org/html/2401.08417v4)
removes that reference-model requirement and combines preference learning with a
behavior-cloning term on the chosen response. Its analysis also warns that human
references are not automatically the best translations; fallback references need
quality checks rather than blind promotion.

[M²PO](https://aclanthology.org/2026.acl-long.469.pdf), equations 1–6, does more
than reshape rewards. It constructs a mixed external/base-model candidate pool,
uses `QE * SAC` as the static score, and fuses its normalized score with normalized
policy log probability. It ranks candidates, forms multiple top-versus-bottom
pairs, weights CPO-plus-NLL terms by score gaps, and adds a listwise ranking loss.
The published fusion is **not** our historical COMET-versus-SAC interpolation.
The paper's candidate generation and preference objective matter to learning;
copying its sigmoid into GRPO is not a replication. Its reported training data
cover English with Chinese, German and Japanese, so English–Vietnamese/NLLB
benefits remain an experiment.

[Fine-Grained RLHF](https://arxiv.org/abs/2306.01693) supports using denser,
typed feedback rather than a single holistic score, with experiments on
detoxification and long-form QA. That evidence motivates a later MT span-credit
experiment; it does not validate an automatic MT error-span detector.

[SimPO](https://arxiv.org/html/2405.14734v2) uses average token log probability
for preference scores to address response-length effects. Its SFT-regularization
ablation reports task-dependent outcomes. Length normalization plus a correction
anchor is therefore a reasonable explicit variant, not universally better than
the original sequence-log-probability CPO objective.

A translation-specific
[empirical CPO study](https://arxiv.org/abs/2409.20059) finds improvements on the
alignment metric can coexist with instability across neural and lexical metrics.
Independent bidirectional evaluation remains necessary.

The newer [StalePO study](https://arxiv.org/abs/2609.16340) addresses legacy MT
post-edits that a newer base model can already surpass. It argues against blindly
increasing the likelihood of those edits and uses an anchored token-level
objective. Its English–Hindi/Turkish setting differs from ours, but reinforces an
important condition here: verify that a proposed correction really improves the
current translation before making it the chosen target.

## Proposed reward and action objective

This is a **hybrid proposal**, not original GRPO or an exact M²PO reproduction.
It uses existing SAC categories first; a new rubric alone would confound the test.

### 1. Preserve an absolute quality signal

Use a bounded, direction-calibrated COMET quality `c(x,y)` and semantic coefficient
`s(x,y)` from SAC. Start with:

\[
q(x,y)=c(x,y)\,s(x,y).
\]

This retains both severity and quality without a separate positive sigmoid gate
for errors. Keep `q` on a common scale within each direction; do not z-score it
inside the four-candidate group before deciding acceptance or confidence. A
high COMET score cannot on its own authorize a faithfulness-positive target.
Calibrate acceptance on manually inspected training examples in each direction.
The four SAC values are ordinal proxies, not calibrated probabilities.

### 2. Select a trustworthy replacement

For each source `x`, build an anchor `a` in this order:

1. A candidate judged fully faithful, in the requested language, with no detected
   omission, number/negation error, truncation or hallucination.
2. Otherwise the **RL training pair's reference**, checked against the source.
3. Only if that reference fails, a cached, verified correction from a teacher.

Never train on validation/test references. Using RL references in this way changes
the former strict SFT-then-pure-RL experiment into hybrid alignment; retain that
distinction in experiment names. Partial match alone does not establish a faithful
anchor. If no trustworthy replacement exists, quarantine the example instead of
declaring its least-bad output a positive target.

Both directions use their own source and target. Do not let an independently bad
reverse translation penalize a correct forward candidate.

### 3. Convert negative severity into contrastive weight

Let `n` be a rejected translation. Define a stopped-gradient score gap and weight:

\[
\Delta_n=q(x,a)-q(x,n),\qquad
w_n=c_{\rm pair}\,\operatorname{clip}(\Delta_n-\epsilon,0,1).
\]

`c_pair` is a calibrated reliability weight, not an unvalidated LLM confidence
claim. Set it to zero for ambiguous comparisons. Start with binary reliable/not
reliable decisions; choose `epsilon` from inspected examples, not an arbitrary
claim of statistical significance. Larger reliable gaps get stronger contrasts.
Do not divide each group's weights by their sum: doing so would restore a full
update even when every comparison has a tiny uncertain gap.

Use the length-normalized score
`ell_theta(y|x) = mean_t log pi_theta(y_t | x, y_<t)` and minimize:

\[
\mathcal L_{\rm failed}(x)=
-\frac{1}{\max(1,|N_x|)}
\sum_{n\in N_x} w_n
\log\sigma\!\left(\beta[\ell_\theta(a|x)-\ell_\theta(n|x)]\right)
-\lambda_{\rm anchor}\ell_\theta(a|x).
\]

The first term raises the anchor's score relative to each failed candidate. The
second supplies a direct teacher-forced target, including omitted content. The
sigmoid now sits **inside a differentiable preference loss over policy scores**;
this is different from applying a sigmoid to a detached judge penalty.

At fixed sequence-score variables, with `d = beta*(ell_a-ell_n)`, the preference
derivatives are `dL/dell_a = -w*beta*(1-sigmoid(d))` and
`dL/dell_n = +w*beta*(1-sigmoid(d))`, before averaging. The anchor NLL adds another
negative derivative for the chosen score. Shared-network interference still
prevents a universal guarantee of probability changes.

Using average scores makes this a length-normalized CPO-style variant. Compare it
against the published sum-log-probability CPO formulation with the same examples;
do not silently call it exact CPO/M²PO. Keep native NLLB language tokens out of
both score lengths and target loss, while retaining them in decoder context.

### 4. Route groups without opposing updates

| Source group | Initial update |
|---|---|
| No rejected candidates | Existing GRPO quality ranking |
| Mixed faithful/failed candidates | Anchor-versus-failed contrasts plus anchor likelihood; no GRPO on this group |
| All candidates failed | Verified training reference/correction versus failures plus anchor likelihood; no GRPO on this group |
| Unreliable judgment, reliable reference | Reference likelihood only |
| No reliable reference or judgment | Skip and record for review |

A teacher/reference output is off-policy. Do not append it to a GRPO group and
pretend it was sampled from the old policy. It belongs in the explicit auxiliary
objective. Preserve a fixed scheduled proportion of group types and log their
effective weights so the training distribution does not silently drift.

The archived labels imply at least 8.47% of groups would need an external-to-rollout
anchor because every candidate is negative. With the stricter Full match screen,
30.89% lack a candidate anchor, before checking references. These are lookup and
scoring needs, **not** necessarily new LLM generations: existing RL references
are the economical first fallback.

## Implementation and validation sequence

1. Keep the present architecture/reward baseline reproducible. Compare NLLB and
   Qwen with unchanged objectives before attributing any improvement to topology.
2. Build a small manually checked training-only calibration set, balanced across
   EN→VI and VI→EN, full/partial errors, omissions and hallucinations. Cache exact
   judgments and verified correction targets. Keep held-out evaluation separate.
3. Run equal-source/equal-token-budget ablations: existing GRPO as the historical
   control; GRPO with the shared quality score `q`; that same GRPO plus correction
   likelihood; and the proposed routed contrastive plus correction objective using
   the same `q`. This separates reward changes from objective changes. Keep LR,
   LoRA, sampling and prompt limits fixed.
4. In the contrastive branch, deduplicate exact candidates and compute each unique
   sequence score once per update. Reuse the anchor score across pairs. Accumulate
   one combined loss per bounded group microbatch and one optimizer step per
   global batch. Do not retain all rollout graphs or backpropagate per pair.
5. Log raw `q`, accepted/failed/tied fractions, anchor provenance, pair weights,
   chosen/rejected log probabilities, their margin, anchor NLL, branch gradient
   norms/cosines and wall time. Compare margins at identical before/after sources.
6. Evaluate a fixed disjoint set in both directions with COMET, chrF/BLEU and an
   independent faithfulness check. Track omission, hallucination, language accuracy
   and length; do not declare success from training reward or the judging model
   alone. Use paired uncertainty estimates for held-out metric changes.

The first trial can use `beta=1`, `lambda_anchor=1` with mean log-probabilities,
but these are starting values to ablate, not surveyed optimal hyperparameters.
An initial gradient check should confirm that the contrastive term does not
overwhelm correction likelihood. Measure tokens/second and API cache hit rate;
the offline audit cannot estimate end-to-end speed of this unimplemented branch.

Defer fine-grained span penalties until error localization is validated. They can
improve credit assignment further, but a wrong span annotation can directly
discourage a correct translation fragment. A verified correction is the more
defensible first action target with the data already available.
