# NLLB correction alignment and rental plan

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
