# NLLB correction alignment and rental plan

Design update, 16 November: use the
[consolidated optimization and training scheme](research/translation_training_scheme_20261116.md)
for the selected MQM approach, penalties, calibration, evaluation and remaining
implementation. The dated 600M recipe and noise-aware scalar control below retain
their historical settings; they do not implement the MQM vector design.

November 14, 2026. Implementation is locally verified; no rental was created and
no paid judge requests were made for these tests. Translation-quality gains have
not yet been measured. See the [reward-to-action survey](research/reward_to_action_20261113.md)
for the objective and its limitations.

## What runs

Use [`nllb_600m_correction.yaml`](../configs/runs/nllb_600m_correction.yaml):

- Public `facebook/nllb-200-distilled-600M`, pinned revision, native per-example
  English/Vietnamese language tokens. Source/target content is bounded to 256
  tokens, preserving language tokens and EOS. No chat instruction is required.
- Frozen backbone, all-linear rank-8 LoRA: 4,325,376 trainable parameters;
  619,399,168 total including adapters. BF16 policy, checkpointed activations.
- 2,000 SFT updates on the 80% partition, then 10,000 hybrid updates on the 20%
  partition. This is a scheduled subset, **not one complete dataset epoch**.
- Batch sizes SFT/RL 16; four sampled translations per directional RL source.
  Two optimizer updates share each judged rollout: 5,000 fresh RL rollouts.
- Candidate scoring accepts up to 64 short rows per graph, with a **4,096 padded
  source-plus-target token budget**. At both maximum lengths this reduces to
  seven rows. Length bucketing and candidate sorting remain active.
- COMETKiwi on the same GPU, OpenRouter `openai/gpt-6-luna`, concurrency 16,
  persistent exact-response cache and bounded parse/transport retries.
- AdamW; SFT LR `1e-4`, RL LR `2e-5`, 100 warmup updates in each phase. The
  validation-COMET plateau scheduler remains configured for RL.
- Both-direction validation every 1,000 optimizer updates, capped at 200 batches,
  with an independent fixed evaluation batch size of 16. W&B project is
  **Machine Translation**. Original run profiles are unchanged.

The filtered NLLB manifest contains 2,372,643 SFT pairs, 593,185 RL pairs,
2,000 validation pairs and 19,150 bounded test pairs. Both training partitions
expand into both translation directions. Validation/test remain separate.

## How failed groups learn

[`AnchoredGRPOModule`](../src/models/anchored_grpo_module.py) routes groups with
SAC values below 0.7 away from GRPO. Their failed translations become rejected
examples against a Full-match candidate, or a judge-verified training reference.
It combines a quality-gap-weighted preference loss and anchor likelihood. Other
groups retain GRPO. Targets from references never enter the on-policy ratio.

The quality score is clipped COMET multiplied by direct SAC. Dynamic fusion,
signed penalties, symmetric endpoint scoring and KL are disabled in this profile.
Native target-language prefixes remain in decoder context but are excluded from
all scored-token losses.

Anchor checks require SAC 1.0 and COMET at least 0.5. The gap floor is 0.05.
These are conservative **uncalibrated heuristics**, not human verification or
proof that a target is correct. Inspect a training-only sample before a long run.
If the reference also fails verification or exceeds its complete token budget,
the group is quarantined. After 20 consecutive untrainable rollouts the run fails
explicitly; it never fabricates a positive target or silently applies AdamW decay
to a skipped rollout. Automatic teacher-generated corrections are not enabled.

Each exact completion in the gradient-scoring pool is scored once per source;
contrastive pairs share anchor scores. A tiny detached score graph supplies exact first-order gradient
coefficients to freshly recomputed candidate graphs. Dense-versus-microbatched
gradient tests verify equivalence. The first update reuses rollout scores;
subsequent updates refresh current scores. Only one transformer graph is held
at a time, with one combined backward per graph and one optimizer step per update.

## Measurements and VRAM × FLOPs

The real public model was tested on a local RTX 4060 Laptop 8 GB GPU. COMET/SAC
were mocked; their costs are not part of the measurements below. AdamW state was
included in memory probes. These are short warm measurements, not rental-GPU
throughput guarantees.

| Same 24-source/24-decoder-position workload | Peak allocated | Forward + backward |
|---|---:|---:|
| 4 rows | 1.50 GiB | 276 ms |
| 32 rows | 3.46 GiB | 353 ms |
| 64 rows | 5.70 GiB | 522 ms |
| 4 rows, full 258/258 positions | 4.22 GiB | 431 ms |
| SFT teacher forcing, 4 rows, full 258/258 | 3.74 GiB | 384 ms |

Scoring/backpropagating 64 short candidates together was approximately **8.46×
faster** than sixteen 4-row graphs at the same lengths on this GPU. That is a
section-level result, not an end-to-end training speedup; generation and API wait
remain. Five real GPU optimizer updates passed with both translation directions,
mocked feedback and finite gradients. Tiny-model tests cover both native NLLB
and causal policies, frozen LoRA parameters, checkpoint load/resume, mixed-group
routing, quarantine, and exact gradient accumulation.

From the measured full-cap graph, the configured seven-row limit, estimated
FP32 COMET weights/activations and 2 GiB headroom give **12.35 GiB planned RL
memory**. Extrapolating the separately measured SFT graph to its 16-row batch
gives **18.51 GiB planned SFT memory**, including the same allowances. COMET
coexistence has not been measured on a rental. A 24 GB GPU has
adequate planning margin; spending on A100 memory is unnecessary for this model.
Do not increase the cap merely to fill VRAM before measuring complete-pipeline
peak memory and examples/second.

Matrix-operation accounting at average padded lengths 24/24 estimates **8.02
TFLOP per RL optimizer update for the policy**, including generation, old-policy
scoring, both correction updates and coefficient refresh. Allow roughly another
1–2 TFLOP for COMET inference. The full-cap scenario is **88.61 policy TFLOP per
update**. Actual source lengths vary; these are not `6 × all_parameters × tokens`
estimates, which would miscount encoder/decoder work and the tied embedding/LM
head. One multiply-add counts as two FLOPs. The profiler only accounts for
supported matrix operations; API inference, softmax and CPU work are excluded.

## Current rental comparison

The [larger-model trial report](nllb_scale_trials.md) contains the later GB-unit
market search, updated offers and 1.3B/3.3B memory plans. It supersedes the offer
selection below; this section preserves the original 600M calculation.

The [official CLI snapshot](../diagnostics/offers_20261114.json) searched verified,
single-GPU offers with reliability above 95% and a $0.65/hour cap, pricing 45 GiB
storage. The selections below additionally have >99% reliability, at least
32 GB host RAM, 12 effective CPU cores and 300 Mbps advertised download, outside
China. Real Hugging Face throughput has **not** been tested on these hosts.

| Selection | Offer | Location | VRAM nominal | Dense BF16 TFLOPS | Price/hour | VRAM × TFLOPS / hourly dollar |
|---|---:|---|---:|---:|---:|---:|
| Lowest expected completed-step cost | 60310926 | India | 24 GB RTX 3090 | ~71 | $0.201 | ~8,509 |
| Faster GPU alternative | 45660816 | South Korea | 24 GB RTX 4090 | 165.2 | $0.426 | ~9,311 |
| Larger-memory compute alternative | 24964759 | Japan | 48 GB L40S | 362.05 | $0.475 | ~36,586 |

BF16 figures use **dense FP32 accumulation**, not sparse or FP16-accumulation
marketing numbers. Sources: NVIDIA's
[GA102 specifications](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf),
[Ada specifications](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf)
and [L40S specifications](https://www.nvidia.com/en-in/data-center/l40s/).
No A100 was present under the search cap. The overall search was nonempty; it
was not broadened or retried above that cap.

The multiplied VRAM/TFLOPs index favors L40S, but unused memory and idle GPU time
do not finish updates. The 4090 costs 2.12× as much as the selected 3090 and would
need more than 2.12× **end-to-end** speed to be cheaper. Judge wait makes that
unlikely in the modeled workload. Recommendation: **one RTX 3090, both NLLB and
COMET on GPU, two updates per judged rollout**. Choose 4090/L40S if a measured
pilot demonstrates enough wall-time benefit or shorter completion time is worth
the premium. Offers can disappear before execution.

## Time and total-cost scenarios

For 2,000 SFT + 10,000 RL updates, the following are **planning scenarios**, not
benchmarks of these rentals. They include 3–6 seconds of judge wait per optimizer
update after sharing each response over two updates, GPU work, 10 minutes setup,
SFT at 0.5 seconds/update and twelve 5-minute validations. Long API tails,
download stalls or full-cap batches can exceed these ranges.

| GPU | RL seconds/update | Total time | Rental | API estimate | Combined estimate |
|---|---:|---:|---:|---:|---:|
| RTX 3090 | 5–9.5 | 15.3–27.8 h | $3.08–5.59 | $6.66–13.32 | **$9.74–18.91** |
| RTX 4090 | 4–8 | 12.6–23.7 h | $5.35–10.08 | $6.66–13.32 | $12.01–23.40 |
| L40S | 4–8 | 12.6–23.7 h | $5.96–11.24 | $6.66–13.32 | $12.62–24.56 |

API assumptions: 64 candidates + approximately four reference checks per fresh
rollout, 20% cache/deduplication savings, 200–400 input and 4–8 output tokens per
request. This gives 272,000 training requests plus 30,720 validation requests.
Prices are $0.10/M input and $0.50/M output on the
[published OpenRouter model page](https://openrouter.ai/openai/gpt-6-luna), checked
November 14. Reasoning tokens, retries, gateway markups, tax and transfer fees
are excluded. Storage is included in the selected hourly prices. Actual request
usage can differ materially; API cost is the dominant optimization target.

## Execute and monitor

From the project root, use the existing Conda environment at `../venv` (the
workspace `./venv`). On a prepared PyTorch rental, use its prepared environment
or create a Conda environment at the same relative location. Export the existing
`HUGGING_FACE_API_KEY`, `ANTHROPIC_AUTH_TOKEN` and `WANDB_API_KEY` securely; do not
place credentials in YAML. The runtime downloads foundation weights from Hugging
Face, including the NLLB PyTorch `.bin` checkpoint.

The native-token filtered files already exist locally. To recreate them on a
new machine containing the original `data/full_prepared` inputs:

```bash
python -m scripts.data.filter_token_pairs --format nllb \
  --tokenizer facebook/nllb-200-distilled-600M
python -m scripts.main.train --config configs/runs/nllb_600m_correction.yaml --dry-run
```

Before the full run, benchmark policy memory without API charges:

```bash
python -m scripts.benchmarks.correction --steps 5 --source-length 258 \
  --target-lengths 258 --batch-sizes 4
```

Then verify a short **live** SFT→RL pipeline using an isolated output path. The
command below reaches real API feedback and COMET; no live check was run here:

```bash
python -m scripts.main.train --config configs/runs/nllb_600m_correction.yaml \
  --set output_dir=logs/nllb-correction-pilot \
  --set max_optimizer_steps.sft=2 --set max_optimizer_steps.rl=10 \
  --set evaluation_every_optimizer_steps=2 \
  --set trainer.limit_val_batches=1 --set data_loader.num_workers=4
```

Do not treat model download completion as training success. Confirm optimizer
updates and correction/GRPO metrics before launching the configured schedule.
Run the full profile in the existing tmux launcher:

```bash
bash scripts/rental/start.sh --config configs/runs/nllb_600m_correction.yaml
tmux attach -t mt-train
```

Detach with `Ctrl-b, d`; the run continues. The launcher never rents, stops or
destroys instances. Standalone held-out evaluation uses the correction module's
checkpoint class automatically when this profile is supplied:

```bash
python -m scripts.main.eval --config configs/runs/nllb_600m_correction.yaml \
  --checkpoint logs/nllb-600m-correction/rl/checkpoints/handoff.ckpt \
  --stage rl --split test --trust-checkpoint --save-predictions
```

Track `train/correction/anchor_nll`, `logprob_margin`, `pair_weight`, `pairs`,
`reference_anchors`, `quarantined_groups` and validation metrics in W&B. Inspect
translation errors independently; a rising chosen/rejected margin alone does
not establish better generalization.

Reproduce the resource/cost JSON with:

```bash
python -m scripts.benchmarks.correction_cost
```

Raw artifacts: [GPU smoke](../diagnostics/correction_20261114/gpu.json),
[full-cap memory](../diagnostics/correction_20261114/worst.json),
[matched batching benchmark](../diagnostics/correction_20261114/matched.json),
[SFT memory probe](../diagnostics/correction_20261114/sft.json),
[cost calculation](../diagnostics/correction_20261114/cost.json).

## Noise-aware correction ablation (15 November 2026)

The full official reference test baseline is COMETKiwi **0.808565 EN→VI** and
**0.805733 VI→EN** on 19,151 pairs. Full training scores are **0.799835 EN→VI**
and **0.798563 VI→EN** on 2,977,999 pairs. These supersede the approximately
0.8074 early training-reference snapshot. Reference evaluation traversed corpus
order; the early mean was not a random-sample estimate of the final mean.
These are scores of the existing translations, not a correctness ceiling or
a calibrated probability. The old model comparison used a smaller test subset.
Use matched source IDs before comparing its model and reference scores.

Full descriptive report and charts:
[`report.html`](../runs/comet-reference-20261115/dataset_analysis/report.html),
with machine-readable statistics, source-conditioned strata and a training-only
review worksheet in the same directory. Reproduce locally with
`python scripts/tools/analyze_reference_corpus.py` and
`python scripts/tools/present_reference_analysis.py`; neither uses a GPU or API.

A reference can omit information, and the reverse view can then require an
unsupported addition. Do not certify references just because they are dataset
targets, or fit a global acceptance threshold to the official test means.
Across-source score dispersion also does not estimate judge noise. Subtracting
a reference baseline before ordinary GRPO centering cancels exactly.

The opt-in experiment
[`nllb_robust_correction.yaml`](../configs/experiment/nllb_robust_correction.yaml)
implements the following changes. Historical configurations retain their old
objective when `correction.noise_aware=false`.

| Pathology | Implemented treatment | Limitation |
|---|---|---|
| Tiny QE differences acquire unit-scale advantages | Pairwise reward contrasts use a dead band, fixed scale and bounded magnitude | The 0.02 floor is an ablation value, not measured noise |
| All partially wrong candidates can promote the least-bad output | SAC 0.7 joins the correction route; only eligible Full-match groups receive policy-gradient ranking | A Full-match label can still be wrong |
| Scalar evaluators disagree about a pair | Abstain if the proposed anchor's COMET trails the rejection beyond a tolerance | Disagreement does not tell us which evaluator is right |
| No reliable comparison still triggers self-imitation | A correction needs at least one supported pair; otherwise no auxiliary update | Coverage shifts toward easier/evaluable sources; log and review abstentions |
| Sampled winner becomes its own unconditional teacher | Candidate NLL weight is 0.1, reference weight 0.25, both multiplied by pair support; NLL stops once the anchor is sufficiently likely | These are heuristics, not reference reliability estimates |
| Hard noisy preferences push odds indefinitely | Soft preference labels (0.9/0.1) have a finite preferred margin; limited NLL avoids overriding that saturation indefinitely | Shared parameters and optimizer state preclude per-sequence guarantees |
| Literal phrasing and capitalization pass a semantic screen | Optional independent criterion evidence includes fluency and orthography, checks anchors and references, and abstains on unknowns | Requires separate bilingual calibration; disabled by default |

For rewards `r_i` in one source group, the new policy coefficient is:

```text
d_ij = r_i - r_j
A_i = mean_j clip(sign(d_ij) * max(abs(d_ij) - noise_floor, 0) / scale,
                  -max_advantage, max_advantage)
```

No division by the current group's standard deviation remains in this branch.
The pairwise coefficients sum to zero, and a whole group below the dead band
has zero signal. They apply only to groups passing the routing checks.
This is a project-specific objective variant, not original GRPO or Dr. GRPO.
All weights are applied after comparison/normalization and are not renormalized
by the number of surviving groups. Off-policy references remain outside the
importance ratio. The existing exact microbatch-gradient implementation is reused.

The fixed-score replay of 45,096 archived source groups found 20,663 groups
eligible for this Full-match ranking path. With the pilot's 0.02 floor, 16,039
of those (77.62%) have no surviving contrast. Mean absolute coefficients on
these same groups change from 0.7474 to 0.01448. This is strong attenuation:
it can prevent optimization of irrelevant details but can also discard useful
small improvements and starve learning. It is not evidence that 77.62% of the
old signal was incorrect. Calibrate the floor on reviewed candidate pairs,
and include smaller-floor/fixed-scale sensitivity runs before adoption.
Artifacts: [replay script](../diagnostics/noisy_objective_20261115/replay.py),
[replay results](../diagnostics/noisy_objective_20261115/replay.json).

The preference loss is `-w * ((1-e)*logsigmoid(beta*margin) +
e*logsigmoid(-beta*margin))`, using mean token log probabilities. At `e=0.1`
its isolated optimal margin is `log(9)/beta`. The anchor term uses
`provenance_weight * max(pair_weights) * relu(anchor_NLL - anchor_nll_floor)`.
It supplies missing tokens without continuously sharpening an already likely
self-generated target. Gap weights are surrogate support, not probabilities.

Use the existing Hydra entry point, an existing compatible SFT export, and
training-only native NLLB CSVs:

```bash
python -m src.train --config-name grpo experiment=nllb_robust_correction \
  model.model_name_or_path=/path/to/compatible/sft_export
```

This is a separate 1,000-update pilot, not a production launch request. It uses
the current direct COMET × SAC reward and adds no recurring SFT or KL changes,
so those are not confounded with the noise-handling experiment. It requires
the usual COMET/SAC access and prepared data. Do not point it at the reference
scoring worker or replace any running job. Only tiny synthetic CPU training
checks were run; no new translation training was launched here.

Separate criterion evidence uses the existing `CriterionJudge` schema, now version 2.
Its rubric explicitly assesses contextual word choice, literal calques, fluency,
and target-language capitalization. Source case is not mechanically copied.
With `evidence.enabled=true`, anchors need complete applicable passing evidence,
Full SAC and the QE screen. A supported criterion error can create a contrast
even when SAC/QE tie. Missing categories, low confidence, low scores, ambiguous
context and uncalibrated judgments cannot certify anchors. Explicitly inapplicable
number checks do not fabricate numerical scores. Contradictory scalar/criterion
signals abstain. Major faithfulness errors do not get a compensating fluency bonus.

First annotate a training-only sample with bilingual reviewers, including
literal-but-faithful outputs, wrong word senses, missing facts, names, capitalization,
good paraphrases and defective references in both directions. Measure per-category
false positives and missed errors. A model's reported confidence is not this
calibration. Training rejects enabled evidence without `calibrated=true`; that
flag records the caller's external validation claim, not automatic certification.
Criterion annotations are cached by exact inputs and prompt version. No criterion
API calls were made as part of implementation/testing. The example reuses the
SAC backend with a different rubric; this does not imply statistical independence.
It can share the same model's biases. Calibrate against external bilingual review
and use a distinct adjudicator when investigating correlated failures.

Compare at matched sources/tokens and with separate output directories:

1. Historical correction control.
2. Noise-aware coefficients only (isolate the new coefficients in a code ablation).
3. Full noise-aware scalar correction, with the same cached scalar judgments.
4. The same objective plus independently calibrated criterion evidence.

Use a fixed development set with blinded source-conditioned adequacy and fluency
review. Report both-direction COMET/chrF/BLEU, omissions/additions, wrong senses,
capitalization, anchor provenance, abstention/coverage by length/domain, saturated
NLL fraction, likelihood margins and sampled drift to the rollout policy. Select
checkpoints on that development evidence; keep the official test for final reporting.
If proxy reward rises while these audited errors worsen, reject the checkpoint.
No quality improvement or complete solution to evaluator bias is established by
the software tests. This remains an ablation, not a tuned optimum.

Further options deliberately deferred: verified teacher corrections for quarantined
examples; bounded resampling to maintain hard-source coverage; clean supervised
exposure before RL; document context for genuinely ambiguous senses; and a measured
frozen-reference retention term. Increasing LoRA rank, adding synthetic paraphrases,
or simply training longer does not resolve a mis-specified preference signal.

Research context: [QE reward overoptimization in MT (He et al., NAACL 2024)](https://aclanthology.org/2024.naacl-long.451/)
and [CPO (Xu et al., ICML 2024)](https://proceedings.mlr.press/v235/xu24t.html)
motivate independent evaluation and explicit preference targets. Neither validates
these project-specific thresholds or guarantees gains for Vietnamese–English.

Verification: 54 CPU tests passed with mocked feedback, including exact dense
versus microbatched gradients in both objective modes, tiny NLLB optimizer
updates/checkpoint resume, causal-policy compatibility, minor-error routing,
conflicting evaluators, small/tied rewards and independent surface errors.
Ruff and whitespace checks passed. Four historical profile/resource cases were
outside the focused suite; the historical run-profile YAMLs are absent in this checkout.
Tests ran in an isolated directory/environment on the existing rental without
changing the running COMET reference scorer. No paid judge calls were used.

## Reconciliation after the complete corpus audit

The complete statistics support several mechanisms but do not validate the
earlier numerical settings or establish a downstream translation gain.

| Finding from complete data | Interpretation | Revision to the proposed procedure |
|---|---|---|
| Train medians 0.83670 / 0.83157; below-0.5 shares 2.09% / 1.85% (EN→VI / VI→EN) | A strong high-score center plus a heavier low-score tail; neither tail size is a human error rate | Prioritize stratified review of the tail and known failure types, not wholesale replacement of references |
| Below-0.75 shares 20.67% / 19.36% in training | A 0.75 screen would exclude a substantial part of the reference distribution | Keep this as a scalar-only ablation screen; calibrate anchor acceptance against reviewed candidate pairs, length and style |
| 52.92% of training EN sentences have ≤10 whitespace units; VI mean length 16.91 units versus EN 13.26 | Short utterances and fragments matter; Vietnamese whitespace units often measure syllables | Maintain short-source/context-required strata and avoid a uniform target/source unit ratio |
| 55,014 VI training references start lowercase; 204,606 do not end in .!?… | There are stylistic variation and fragments, as well as possible errors; the terminal flag also includes closing quotes | Judge capitalization only under target-language conventions and actual sentence context; do not automatically capitalize or declare missing punctuation |
| 477 training pairs meet a conservative copy-like flag; examples include untranslated prose, names and equations | The heuristic finds real issues but also valid language-independent content | Queue flags for source-aware language review; do not automatically remove them or turn every flag into a negative reward |
| Train directional correlation 0.8993, but 18.68% of pairs differ by >0.05 | Forward and reverse views are related but not interchangeable | Retain direct directional scoring and separate error/coverage reporting; do not average a bad reverse candidate into a forward candidate's reward |
| 1,439 repeated normalized pair rows; 49,854 normalized EN sources with multiple distinct targets | Full-pair deduplication and multiple valid translations are different cases | Preserve target alternatives and connected source groups; do not erase paraphrases by keeping just one target per source |
| Raw train/test overlap: 18 test pair rows, 891 test EN source rows, 225 test VI source rows | Raw official partitions need overlap controls for strict source-disjoint evaluation | Keep the existing connected-group exclusion. These counts are not proof of leakage in the already prepared project partitions; their manifest/outputs need a separate verification |

The earlier model comparison was joined to exact official text for all 1,600
pairs in both directions. Its matched reference means are **0.810725 EN→VI**
and **0.817351 VI→EN**, versus RL means **0.839934** and **0.846201**. Thus
RL exceeds the reference on this evaluator by **0.029209** and **0.028850**.
This supports the proxy improvement on matched inputs; it does not show that
every model output is better. Comparing these model scores to full-test means
instead would mix populations. Same-checkpoint score comparisons may also have
minor numerical differences from runtime/batching changes.

The noise-control replay uses training rollouts only. At fixed scale 0.1:

| Dead band | Eligible groups with no contrast | Mean absolute policy coefficient |
|---|---:|---:|
| 0 (offline control) | 9.32% | 0.04955 |
| 0.002 | 20.78% | 0.04239 |
| 0.005 | 38.98% | 0.03417 |
| 0.010 | 58.53% | 0.02496 |
| 0.020 | 77.62% | 0.01448 |

Historical standardized coefficients average 0.7474 on these same groups.
Even the zero-dead-band control is much smaller: fixed scaling and thresholding
are separate interventions. Do not attribute the combined attenuation to the
dead band alone or assume coefficient ratios equal parameter-update ratios.
Sweep scale and floor separately, and inspect branch gradient norms before
choosing a learning rate. Zero floor is only an offline control in the current
implementation, which requires a positive training floor.

The recommended order is now:

1. Verify prepared source-group boundaries; retain all legitimate alternative
   translations. Maintain source-length/style coverage and audit rare suspicious
   copying, source-language and alignment cases without broad heuristic deletion.
2. Annotate the generated **training-only 240-item review worksheet**, balanced
   over score bands and directions. Supplement it with sampled training rollout
   candidate contrasts: references alone cannot calibrate a ranking noise floor.
   Record context-required cases instead of inventing missing document information.
3. Calibrate source-aware adequacy, naturalness and orthography checks against
   that review. The scalar-only profile remains a comparison control, not the
   recommended retraining recipe. SAC 0.7 is a repair candidate; it is not an
   independently established error label. A second rubric on the same model can
   share the same failure modes.
4. Compare the bounded-coefficient and softened-correction interventions
   separately on matched training sources, then combine only if independent
   development adequacy/fluency and coverage are retained. Account for skipped
   sources and capped effective updates rather than matching optimizer counts alone.
5. Use a fixed representative development set for selection and the official
   test only for final reporting. Corpus-order reference evaluation is not a
   random sample or a training learning curve. Original model-training loaders
   shuffle batches; this audit does not establish a prefix-biased training stream.

No floor, scale, anchor weight or smoothing probability has been tuned or
validated by the completed reference statistics. No objective/tensor code was
retuned from the official test. The retained soft-label finite margin, reduced
self-imitation and absolute routing safeguards are mechanistic protections,
not guarantees of label truth or translation improvement. Very long pairs were
included in scoring, but exact internal COMET-tokenizer truncation was not
audited; source-length extremes should be interpreted with that limitation.
