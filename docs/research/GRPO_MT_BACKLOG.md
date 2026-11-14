# GRPO MT: diagnosis, backlog and proposed solution

Latest investigation, November 11: [feedback learning and cyclic phase conflicts](feedback_learning_investigation_20261111.md).
The completed cyclic run had active scalar-reward learning, an inactive LLM
judge, and repeated SFT drops offsetting about 90% of summed RL COMET gains.
Its preserved optimizer counters and full learning rates differ from the
older run documented below; do not carry the earlier warmup diagnosis forward.

Date: 8 November 2026. Evidence: scalar/configuration snapshots from A100
instance **59691795**, plus a read-only CPU audit of the configured training data.
Code changes below are local; the running process still uses its original code.
No new rental, LLM generation job or paid feedback requests were started for this review.

## Recommendation

Fix sampling correctness and warmup before increasing training duration, reducing
trainable layers, or purchasing synthetic data. Establish a fixed development
baseline, inspect candidate-level reward conflicts, then compare all-layer LoRA
with last-four-layer LoRA. Use a small, audited LLM augmentation pilot only after
these controls work. Keep the final test set out of model and reward selection.

## What the evidence actually establishes

| Finding | Evidence | Interpretation |
|---|---|---|
| The observed RL run is short | Fresh SFT: 317 optimizer updates. Initial RL snapshot: 42 updates. 1,900 directional RL views / batch 48 = 40 rollouts; two updates per rollout = **80 planned RL updates**. | Thousands of earlier SFT updates are not thousands of GRPO updates. |
| RL never finishes warmup | Actual resolved configuration has 100 warmup updates and LR 5e-6. At update 80, LR is 4e-6; the mean configured LR over 80 updates is 2.025e-6. | A 100-update warmup is unsuitable for this 80-update experiment. |
| Sampling and scoring disagree | Runtime warns that Transformers replaces temperature/top-p/repetition settings with model defaults. A regression test reproduces this merge and checks sampled-token probabilities. | The intended GRPO likelihood ratios are invalid when the sampler and `_logps` describe different policies. This is a correctness bug, not just a tuning choice. |
| Evaluation is exposed to the same merge | Model-specific defaults can override the evaluation config, including intended greedy decoding. | Re-evaluate saved checkpoints using explicitly fixed decoding before interpreting small validation differences. |
| Backbone freezing already exists | Resolved configuration enables rank-8 attention LoRA; the PEFT construction freezes base parameters. | Restricting adapter layers is an optional capacity ablation, not the first fix for accidental full-backbone training. |
| Reward is not an all-ties signal | Snapshot rewards at RL updates 2/22/42: 0.67407 / 0.66769 / 0.67625. Within-source reward std is about 0.0166–0.0184; exact ties range from 0% to 4.17%. | There is a learning signal, but three changing batches cannot establish a plateau or improvement. |
| Prompt truncation is rare here | Seeded augmented-data audit: RL prompts over the 256-token cap **0/1,900**; SFT **1/7,600**. RL references over 128 tokens: **6/1,900**. | Truncation deserves monitoring, but is not a convincing explanation for broad reward stagnation in this selected dataset. |
| Reward conflict is not yet measured | Legacy logs contain batch-average BLEU/COMET, not their candidate-level ranks. | A disagreement percentage cannot be reconstructed honestly from these averages. New instrumentation records the missing evidence. |

These findings identify a definite sampling bug and an inappropriate schedule.
They do **not** prove that either alone explains every reward fluctuation. The
logged KL is only around 3e-4; its beta=0.04 weighted magnitude is about 1e-5.
Existing logs do not support claiming that a large KL penalty is the main cause.

Evidence files:

- `logs/reward_diagnosis/active/rl/resolved_config.yaml`
- `logs/reward_diagnosis/active/rl/progress.jsonl`
- `logs/reward_diagnosis/active/sft/result.json`
- `logs/reward_diagnosis/active/driver_tail.log`
- `logs/reward_diagnosis/source_length_audit.json`

The refreshed snapshot reaches **62/80 RL updates** and contains four recorded
batch means. Its [scalar audit chart](../../logs/reward_diagnosis/reward_diagnosis.png)
separates observed reward points from the LR schedule derived from configuration.
Only the SFT end-of-stage development result was available in that snapshot;
no missing RL evaluation point has been invented. Machine-readable findings:
`logs/reward_diagnosis/diagnosis.json`.

## Prioritized backlog

"Prepared" means implemented locally and tested; it does not mean deployed into
the active Python process. Acceptance requires the controlled validation below.

| ID | Priority / status | Work and proposed solution | Acceptance criteria | Dependencies |
|---|---|---|---|---|
| MT-001 | **P0 · prepared** | Disable Transformers model-default merging for rollout generation and evaluation, including PEFT wrappers. Preserve compatibility with older Transformers signatures. | Generation transition probabilities match `_logps` within 2e-5 on masked tokens; neutral top-p/temperature/repetition settings stay neutral; intended greedy evaluation remains greedy. Check both base and LoRA models. | None |
| MT-002 | **P0 · prepared** | Use stage-specific warmup: SFT 100, current short RL experiment **8**. Log actual LR and optimizer-update count. For larger datasets, derive warmup from the planned update count. | Planned versus actual RL updates are explicit; LR reaches 5e-6 at update 8; SFT and RL charts do not share an ambiguous step counter. | MT-001 |
| MT-003 | **P0 · prepared; baseline still pending** | Log each optimizer update, not only each reused rollout or every twentieth update. Append validation history and define W&B scalar chart axes. Add explicit fixed-set validation at RL step 0 before the next experiment. | Two updates from one rollout produce two records. Resuming preserves counters. Validation charts retain all evaluations, directions and optimizer steps. Step-0 and periodic evaluations use the same complete dev set and decoding. | MT-001 |
| MT-004 | **P1 · prepared** | Record reward component dispersion, weighted within-source variation, BLEU–COMET ranking disagreement, tie/duplicate/empty/capped fractions, gradient norm, LR, KL, clipping and parameter drift. Store scalar candidate scores without source text. | Disagreement excludes ties, reports comparable-pair counts, and is computed within each source. Undefined correlations are omitted. A fixed-checkpoint audit establishes whether BLEU or QE dominates preference decisions. | MT-003 |
| MT-011 | **P0 · implemented; A100 replay benchmark completed** | Length-bucket sampler/collator, actual-length candidate regrouping, padding trimming, and immediate weighted backward per batched graph are implemented for GRPO and both self-play roles. Group advantages remain fixed and clipping/stepping occurs once per update. Opt-in preset: `phomt_rl_throughput`. | Local loss/gradient/metric equivalence tests pass. Exactly **ten disposable GPU updates** compared legacy/chunk-8/chunk-16 paths; adapters restored. Chunk 8 measured **5.13× faster updates**, with 19.43 GiB peak allocation including a 3 GiB feedback reserve. Same-plan old/current likelihoods agree exactly; FP32 four-candidate audit differs by at most 1.92e-5. See scope below; full COMET/data validation remains separate. | MT-001, MT-003 |
| MT-005 | **P1 · prepared ablation** | Compare current all-attention LoRA with LoRA only in Qwen decoder blocks **24–27**. Keep base weights frozen in both arms. | Selected-layer checkpoints resume correctly; only selected adapters change; compare on the same SFT starting checkpoint, source order, candidate count, seed and development set. Report both directions and retention checks. | MT-001–004 |
| MT-006 | **P1 · proposed** | Pilot LLM candidate augmentation on 100 train-only parent pairs, preserving stage membership. Add independently calibrated faithfulness labels. | No validation/test contamination; immutable parent IDs and stage provenance; audit judge decisions against EN/VI annotations; compare downstream dev performance with the fixed baseline. No API calls before a separate API budget/provider is selected. | MT-004 |
| MT-007 | **P2 · proposed** | Ablate the existing 70% QE / 30% BLEU reward against a faithfulness-gated variant. Inspect component scales and candidate preference conflicts before changing weights. | Measure actual within-source preference flips, independently judged omissions/additions, and corpus BLEU/chrF/QE. A rising proxy score with worse faithfulness fails acceptance. | MT-004, MT-006 |
| MT-008 | **P2 · proposed** | Make augmentation token-budget aware and monitor output truncation. Prefer aligned segmentation or reject oversized concatenations over hiding source content and scoring against unseen text. | Caps remain enforced; source/target alignment is retained; no cross-stage partners; dev/test remain unaugmented. | MT-003 |
| MT-009 | **P2 · proposed** | Compare two or three RL passes or smaller rollout batches only after correctness is fixed. Use early stopping on a fixed dev set rather than targeting an arbitrary thousand updates. | Report unique sources, candidates, optimizer updates and GPU cost separately; gains must survive fixed-set evaluation across seeds. | MT-001–004 |
| MT-010 | **P3 · proposed, separate trainer** | If the pilot succeeds, implement a distinct offline multi-pair preference experiment rather than relabeling GRPO as M²PO. | Reproduce the intended objective, handle tied rankings safely, and compare against matched-data GRPO and single-pair baselines. | MT-006–007 |

## Proposed implementation sequence

1. Apply MT-001–003 in the **next** controlled run. Reuse the saved SFT starting
   model, not the already RL-updated model, so comparisons have a common start.
2. Run the established short GPU check, then a single instrumented RL pass.
   Evaluate before training and periodically on the same 500 development views.
   Plot reward components separately; label training reward as sampled-batch data.
3. Use MT-004 to decide what is failing. Low diversity suggests a candidate
   sampling issue; weak/zero gradients suggest optimization; conflicting reward
   ranks suggest supervision. Avoid changing all three at once.
4. Compare the layer-selection arms with the other choices held fixed. Fewer
   adapter layers may regularize scarce data, but can also reduce useful capacity.
   Frozen model weights remain the dominant memory allocation.
5. Purchase only the small augmentation pilot after its judge and cost are
   reviewed. Expand only if a matched-data experiment improves independent dev
   quality and faithfulness. Run the full test set after choosing the method.

Prepared experiment presets:

```bash
# Requires the same saved/reconstructed merged SFT starting model.
python src/train.py --config-name grpo experiment=phomt_rl_diagnostics
python src/train.py --config-name grpo experiment=phomt_rl_top4
```

Both presets use four candidates and batch 16, eight warmup updates, and periodic
development evaluation. The second changes adapter layer selection. They are
proposed ablation settings, not proven optimal hyperparameters.

## M²PO review and relevance

[M²PO](https://aclanthology.org/2026.acl-long.469/) is offline CPO, not GRPO. It
combines teacher/base candidates, discrete semantic-alignment gating of QE,
scheduled model confidence, multiple preference pairs, ranking and NLL losses.
Its reported language pairs are English↔Chinese/German/Japanese, not Vietnamese;
its reward uses COMETKiwi-XXL rather than our wmt22 evaluator. Those distinctions
prevent treating its reported gains as an EN↔VI result or dropping its sampler
unchanged into on-policy GRPO. See the [paper, §§4–5 and Appendix C](https://aclanthology.org/2026.acl-long.469.pdf).

Expected coverage of its feedback on **our** criteria is an inference requiring
an EN↔VI audit, not a measured accuracy percentage:

| Criterion | Expected coverage | Additional control I recommend |
|---|---|---|
| Semantic adequacy | Direct | Independent bilingual calibration |
| Omissions | Direct | Separate omission annotations |
| Hallucinated additions | Direct | Separate addition annotations |
| Fluency | Indirect through QE | Independent fluency judgment |
| Names, numbers, units | Partial | Locale-aware consistency checks |
| Negation and relations | Partial | Targeted contrastive examples |
| Target language and format | Partial | Explicit validation rules |
| Terminology and domain constraints | Partial | Task-specific glossary checks |
| Exact reference overlap | Not its target | Retain BLEU/chrF as diagnostics |

No defensible "covers X%" claim is possible from the current logs. In particular,
a semantic category does not provide per-error counts or guarantee detection of
every number, entity or terminology error.

### Proposed train-only augmentation pilot

Use 100 parent pairs in both directions: 200 source views. Produce eight
candidates per view (four teacher, four local-policy), yielding 1,600 candidates,
800 teacher outputs and 1,600 alignment labels. This is a pilot size I propose,
not a reported PhoMT experiment. Track the actual API tokens and GPU time.

Keep all descendants in their parent's original SFT or RL partition. Deduplicate
and validate language, names, numbers, semantic preservation and length. Include
partial-error candidates as **rejected preferences**, not supervised gold text.
Compare against a no-augmentation baseline with matched source counts.

Aligned bilingual paraphrases and back-translation are further experiments I
propose beyond candidate augmentation. Never paraphrase one side and silently
reuse an incompatible target. Generated sources require a checked aligned
translation and the same provenance/leakage rules.

For GRPO, a possible experiment is `semantic_gate * (0.7 * QE + 0.3 * BLEU)`.
That would be our adaptation, not a reproduction of the paper. Keep rollout
sampling and scoring identical. Offline candidate-generation settings cannot
be copied into GRPO without accounting for the changed sampling distribution.

## Local implementation and validation

- Sampling fix: `src/models/grpo_module.py`, including the self-play generation path.
- Candidate diagnostics: `src/models/components/reward_diagnostics.py`.
- Update/evaluation histories and chart axes: `src/utils/translation_telemetry.py`,
  `scripts/rental/wandb_relay.py`.
- Selected-layer LoRA: `src/models/components/adapters.py` and the two presets above.
- Length sampler/collator: `src/data/components/length_batching.py` and the translation data module.
- Candidate grouping and immediate backward: `src/models/components/candidate_batching.py`,
  the GRPO/self-play modules, and `configs/experiment/phomt_rl_throughput.yaml`.
- Regression suites: `tests/grpo/test_diagnostics.py` and `tests/grpo/test_length_batching.py`.

The full local suite passed **160 tests** with **95.42% coverage**, including sampler/scorer agreement,
per-update records, evaluation history, W&B relay deduplication, and selected-layer
training/checkpoint reload. New checks cover length buckets and epoch shuffling,
augmentation RNG isolation, full-batch/microbatch gradient and metric equivalence,
forced language context, uneven chunks and one optimizer step per full update.
These tests establish implementation behavior, not
improved translation quality. Focused strict type checking and Ruff also passed.
No live-run restart or new augmentation API calls
are part of this backlog delivery.

Transformers default handling is documented in its
[generation API](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/text_generation)
and was verified against the installed library source and regression tests.

## Measured speed and the M²PO comparison

The latest log reports the fresh run **completed**, including final testing.
The complete training-phase trace contains 40 rollout batches and 80 updates:

| Work | Measured mean |
|---|---:|
| Rollout batch: 48 sources, 96 candidates, two updates | 64.912 s |
| Optimizer update including amortized rollout/scoring | 32.456 s |
| Current-policy forward, per update | 8.217 s |
| Backward, per update | 15.715 s |
| Actual Adam/transfer optimizer operation, per update | 0.0457 s |
| COMET feedback, per training rollout batch | 0.872 s |

COMET occupies **1.34%** of rollout-batch time; current-policy forward/backward
occupy about **74%**. Removing COMET alone would therefore save approximately
1.3%, not produce a major speedup. Measured GPU training totals 43.3 minutes,
about **$0.289** at $0.401111/hour, excluding setup, validation, testing and backup.

M²PO reports **8× lower GPU training time** than its group-8 GRPO/GSPO baselines
(Appendix G, Table 14), and approximately two hours for two epochs on H100.
It prepares expensive feedback offline, but does not report per-request feedback
latency. These are training-time results, not evidence that its judge answers
faster. [Source: paper §§5.2 and Appendix G](https://aclanthology.org/2026.acl-long.469.pdf).

For this implementation, the immediate speed experiment is MT-011. An offline
multi-pair method is a separate algorithm/data-preparation experiment (MT-010),
not a justified prediction of dividing our 32.5 seconds by eight. Hardware,
model, group size, sequence lengths, dataset and preprocessing costs differ.

## Policy forward and backward bottlenecks

This analysis uses the completed A100 SXM4 40 GB run: Qwen2.5-7B-Instruct,
BF16, rank-8 attention LoRA in all 28 decoder blocks, source/completion caps
256/128, gradient checkpointing, and `logprob_batch_size=1`. A normal rollout
contains 48 sources and 96 candidates and is reused for two optimizer updates.
The means below include the smaller final rollout. These are synchronized
phase wall times, including Python/dispatch overhead, not kernel-only timings.

| Phase | Calls per rollout | Mean seconds per call | Amortized seconds per optimizer update | Share of rollout time |
|---|---:|---:|---:|---:|
| Current-policy scoring forward | 2 | 8.217 | 8.217 | 25.32% |
| Backward, including checkpoint recomputation | 2 | 15.715 | 15.715 | 48.42% |
| Old-policy scoring, no gradients | 1 | 4.887 | 2.444 | 7.53% |
| Reference scoring, no gradients | 1 | 3.478 | 1.739 | 5.36% |
| Autoregressive generation | 1 | 7.310 | 3.655 | 11.26% |
| COMETKiwi feedback | 1 | 0.872 | 0.436 | 1.34% |
| Optimizer operation | 2 | 0.0457 | 0.0457 | 0.14% |

The whole rollout averages **64.912 seconds**, or **32.456 seconds per update**.
The table leaves about 0.63% for other work; it does not double-count nested
COMET phases. COMET timing here is reconstructed from training events, excluding
development evaluations. Source files are
`logs/reward_diagnosis/active/rl/phase_timing/summary.json` and `events.jsonl`.

### Bottlenecks in the profiled implementation, before the batching change

1. **Scoring is split into rows, but backward is not.** `_logps` builds a list
   of candidate outputs and concatenates them. `_translation_step` computes one
   loss and one backward after the entire list is built. Every candidate's graph
   remains live until that backward. At chunk size 1, a full rollout executes
   96 old-policy + 96 reference + 2 × 96 current-policy model calls: **384
   scoring forwards**, excluding generation and checkpoint recomputation.
   Larger chunks reduce call count but alone do not release the graphs earlier.
2. **Checkpointing repeats decoder work during backward.** All-layer LoRA
   requires gradients through all decoder blocks. There are 96 × 28 = 2,688
   decoder-block checkpoint invocations per full current-policy forward, with
   checkpointed regions revisited during backward. Nonreentrant checkpointing
   can stop recomputation early, so this is not a claim that every operation is
   repeated in full. [PyTorch checkpoint documentation](https://docs.pytorch.org/docs/2.9/checkpoint.html).
3. **Frozen weights still propagate activation gradients.** For
   `y = x W^T + s (x A^T) B^T`, freezing `W` removes its weight gradient, but
   `dL/dx` still includes `(dL/dy) W`. Earlier trainable adapters require this
   dense propagation through later attention and MLP layers. The optimizer is
   cheap because it updates small adapters; backward still traverses a large
   network. Reducing LoRA rank alone does not remove that traversal.
4. **Chunks retain global padding and use a dense FP32 vocabulary softmax.**
   `_logps` slices rows without trimming padded columns. It converts completion
   logits to FP32 and materializes `log_softmax` over all 152,064 vocabulary
   entries before gathering target tokens. As a shape illustration, 96 candidates
   × 128 completion positions × 152,064 entries × 4 bytes is **6.96 GiB** for
   that one tensor across retained graphs. Actual padded lengths are absent
   from the logs, so this is not a measured allocation. Valid source+completion
   tokens average 96.4 per candidate; completion tokens average 38.5. The code
   already uses `logits_to_keep` where supported to avoid projecting source
   positions. [Qwen model configuration](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json).

A tiny random-Qwen CPU graph audit confirmed the structural behavior without
renting a GPU: eight candidates at chunk size 1 caused eight forward and eight
checkpoint recomputation entries per block; chunk size 4 caused two each.
Disabling checkpointing removed recomputation. With adapters only in the final
block and checkpointing off, earlier frozen blocks had no backward calls and
their outputs did not require gradients. This validates graph structure, not
A100 speed or translation quality. Results:
`logs/reward_diagnosis/backprop_graph_audit.json`.

### Optimization order and verification

The first two changes below are now implemented locally, including data-level
length buckets and actual generated-candidate grouping. GRPO and self-play use
`_backward_rollout`, so their training path no longer concatenates all candidate
graphs. `_logps` restores likelihood row order for detached old/reference scoring
and compatibility callers. The opt-in `phomt_rl_throughput` preset separates
rollout size, graph size and padded-token limits. The completed A100 replay
benchmark below measures the new path; the historical timings above describe
the previous code.

1. Implement true training microbatches: calculate group advantages once; for
   each candidate chunk, backward `chunk_loss * chunk_size / total_candidates`
   immediately; accumulate gradients; clip and step once after all chunks.
   Keep two reuse iterations and the original per-completion GRPO reduction.
   Detach logged metrics so they do not keep graphs alive. This first removes
   the memory obstacle to increasing the actual scoring chunk size.
2. Trim source and completion padding per chunk, then consider length buckets.
   Preserve attention/position IDs and deterministic language-token context;
   the last valid scored position, rather than a raw mask count, determines
   completion trimming. Check token likelihood and gradient equivalence.
3. Compare chunk sizes 4/8 and checkpointing off or limited to selected blocks
   only after memory fits. Do not disable checkpointing on the current retained
   full-rollout graphs without a memory check. Record padded tokens and peak
   allocated/reserved memory; VRAM occupancy alone is not throughput.
4. Evaluate the existing top-four-layer LoRA preset as a separate capacity
   ablation. Ensure no embedding-output gradient hook forces the frozen prefix
   into backward. This can remove backward through blocks 0–23 but still
   requires their forward pass and a translation-quality comparison.
5. If vocabulary projection/softmax remains material, benchmark an equivalent
   fused/chunked linear-logprob loss. Preserve masks, temperature, k3 KL and
   per-completion normalization rather than silently changing the objective.

The existing logs cannot divide the 15.715-second backward into exact MLP,
attention, vocabulary, recomputation and autograd-engine costs. A representative
operator-level GPU trace is required for those percentages. No additional
rental, training restart or GPU speedup measurement was performed for that
initial code/log analysis. The follow-up rental benchmark is recorded below.

## Completed A100 batching benchmark — 2026-11-08

The disposable benchmark used pinned **public foundation Qwen2.5-7B-Instruct**,
rank-8 q/k/v/o LoRA in all layers, BF16 autocast, checkpointing, caps 256/128,
16 synthetic EN↔VI sources and four sampled candidates each. All paths replayed
the same 64 candidates and restored adapter parameters and Adam state between
updates. Three configurations had one warmup and two measured updates each;
the tenth update probed half the rollout. All ten updates were discarded and
the original adapter checksum was restored. No training checkpoint was saved.

| Scoring path | Forward s/update | Backward s/update | Whole optimizer update s | Candidates/s | Peak allocated GiB |
|---|---:|---:|---:|---:|---:|
| Legacy: 64 one-row graphs, concatenated before one backward | 5.139 | 9.557 | 14.798 | 4.32 | 25.37 |
| New: cap 8, immediate weighted backward | 0.987 | 1.857 | **2.884** | **22.19** | **19.43** |
| New: cap 16, immediate weighted backward | 1.093 | 1.997 | 3.119 | 20.52 | 21.41 |

All peak allocations include a **3 GiB dummy allocation reserved for feedback**.
COMET was not loaded or executed, and feedback latency is excluded. The GPU was
an A100-SXM4-40GB; the environment was Torch 2.14.1+cu132, Transformers 4.57.6,
PEFT 0.18.1 and Lightning 2.6.6. Cross-run comparisons against older Torch
versions or the earlier 96-candidate training trace are not controlled.

Chunk 8 improved update time **5.13×**. Including measured generation and
old/reference scoring amortized over two updates gives **21.994 → 8.440 seconds
per update**, or **2.61×**, still excluding feedback, validation and setup.
Cap 16 was about 8.1% slower than cap 8: padded scoring tokens increased from
8,152 to 10,316 (26.5% more). Increasing graph size did not improve this replay's
throughput. The half-rollout probe used 32 candidates and measured 1.832 s/update
and 17.47 candidates/s; it was only one measurement, not an optimal-batch search.

The benchmark reports **model-forward calls separately from backward API calls**:
legacy has 64 independent forward branches and one backward invocation; cap 8
has eight vectorized forward graphs and eight immediate backward invocations.
The gain is fewer row-by-row decoder/checkpoint operations and less retained
state, rather than reducing the legacy's single backward API invocation.

### Precision audit before training

Changing BF16 batch/padding shapes produced a maximum token-log-probability
change of 0.608 in the main replay, with mean absolute change 0.00901 in the audit.
That warranted a separate **forward-only** audit, not additional optimizer
updates. An untrimmed batch-8 comparison also showed drift (maximum 0.774),
so padding removal alone does not explain it.

- Repeating the same batched scoring layout: **zero difference**.
- Old-policy no-grad versus gradient-enabled current-policy scoring with the
  same layout and unchanged weights: **zero difference**.
- Four-candidate FP32 batching/padding check: maximum **1.9133e-5**, mean **1.0268e-6**.
- Disabling reduced-precision BF16 GEMM reduction reduced, but did not eliminate,
  cross-shape differences; no global precision flag was changed in training code.

The inference supported by these checks is floating-point shape sensitivity,
with no batching equivalence failure observed in the tested FP32 subset. Keep
old/current scoring plans consistent within each rollout. PyTorch documents
that batched and sliced calculations need not be bitwise identical:
[numerical accuracy documentation](https://docs.pytorch.org/docs/2.9/notes/numerical_accuracy.html).

Reproduction: `python -m scripts.benchmarks.grpo` followed by
`python -m scripts.benchmarks.scoring_precision` on the same GPU/cache. The latter
performs zero optimizer updates. A CUDA smoke test verifies the benchmark's
ten-update limit and restoration of adapter parameters. Scalar artifacts are in
`logs/benchmark_grpo_a100/final-results/benchmark-results/`.

At 1,900 directional RL views, batch 16 and two reuse updates imply 119 rollout
batches and 238 updates. Applying this synthetic replay's 8.440 seconds/update
gives approximately **33.5 minutes**, about **$0.22** at the measured rental rate
of $0.394167/hour. This is an estimate conditional on similar token lengths;
COMET execution, development/final evaluation, SFT, setup and model download
are additional. The disposable benchmark's account debit at cleanup was
**$0.2982**, under its $1 cap; instance **59713818** was destroyed and the
official instance listing verified **zero remaining instances**. Training was
not started.
