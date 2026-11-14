# NLLB-1.3B and the larger dense-model trial

November 14, 2026. Both trials below used public pinned Hugging Face weights on
the local RTX 4060 Laptop 8 GB, with mocked COMET/SAC. No rental was created and
no paid feedback request was made. This report updates the hardware shortlist in
the [600M execution plan](correction_training.md).

The larger official dense NLLB checkpoint is **3.3B**, used here for the requested
approximately 4B experiment. Model cards:
[distilled 1.3B](https://huggingface.co/facebook/nllb-200-distilled-1.3B) and
[3.3B](https://huggingface.co/facebook/nllb-200-3.3B). Both use the research
CC-BY-NC-4.0 license.

## Prepared profiles

| Setting | 1.3B | 3.3B |
|---|---|---|
| Profile | `nllb_1p3b_correction.yaml` | `nllb_3p3b_correction.yaml` |
| HF model | `facebook/nllb-200-distilled-1.3B` | `facebook/nllb-200-3.3B` |
| Revision | `7be3e24664b38ce1cac29b8aeed6911aa0cf0576` | `1a07f7d195896b2114afcb79b7b57ab512e7b43e` |
| Frozen base parameters | 1,370,638,336 | 3,344,863,232 |
| Rank-8 LoRA parameters | 11,796,480 | 17,301,504 |
| Global SFT/RL batch | 16 / 16 | 16 / 16 |
| SFT graph microbatch | 8 | 4 |
| Maximum scoring graph rows | 32 | 16 |
| Padded scoring-token budget | 4,096 | 2,048 |
| Scoring rows at full 258/258 cap | 7 | 3 |
| Rental precision | BF16 | BF16 |
| Output directory | `logs/nllb-1p3b-correction` | `logs/nllb-3p3b-correction` |

All other experimental controls match: original disjoint 80/20 partitions,
2,000 SFT + 10,000 hybrid optimizer updates, four candidates per directional
source, two updates per judged rollout, rank-8 all-linear adapters, learning
rates, scheduler, source/target content cap 256, native language controls and
fixed bidirectional evaluation batches. SFT microbatch losses are weighted by
valid target-token count, preserving the exact global token-mean objective.

Tokenizer JSON files are byte-identical across 600M, 1.3B and 3.3B, and the
normalized tokenizer-state SHA-256 also matches. The new manifests reuse the
existing CSVs **after verifying every CSV hash and tokenizer state**. No pairs
are re-sampled, copied or moved between partitions. Tests cover compatible
reuse and rejection after CSV tampering.

## Actual local results

| Probe | 1.3B BF16 | 3.3B NF4 |
|---|---:|---:|
| Successful RL optimizer updates | 5 | 5 |
| Directions | EN→VI and VI→EN | EN→VI and VI→EN |
| Peak allocated during the five-step smoke | 2.79 GiB | 4.61 GiB |
| Full-cap policy scoring graph | 4 rows, 5.76 GiB | 1 row, 4.37 GiB |
| Full-cap graph forward + backward | 969 ms | 907 ms |
| Five-step smoke wall time | 34.0 s | 40.9 s |

These timings use tiny two-source/two-candidate smoke batches and a short
generation cap. They are **not production step-time comparisons**: precision,
graph batch size and kernels differ. Mock scores exercise the correction branch;
they do not measure translation quality or API latency.

The 3.3B BF16 loader ran out of memory on the local 8 GB GPU before training.
Direct-to-GPU loading removed CPU-memory pressure but still exceeded 8 GB during
legacy-shard loading. NF4 then passed. Rental profiles retain BF16; its full
training fit must be verified on the intended host. The benchmark now updates
Lightning's device tracker even when HF has already placed a policy on CUDA.

## BF16 memory and work projections

| Production plan | 1.3B | 3.3B |
|---|---:|---:|
| Planned SFT GPU memory | 15.00 GiB | 21.64 GiB |
| Planned RL GPU memory | 14.07 GiB | 19.04 GiB |
| Policy work per update at typical 24/24 padded lengths | 19.06 TFLOP | 47.18 TFLOP |
| Policy work per update at the full 258/258 cap | 210.25 TFLOP | 517.96 TFLOP |
| Additional COMET work | approximately 1–2 TFLOP/update | approximately 1–2 TFLOP/update |

These memory budgets include estimated FP32 COMET weights, 0.75 GiB COMET
activation allowance and 2 GiB headroom. The 1.3B projection uses its measured
BF16 graph. The **3.3B BF16 projection replaces NF4 base storage with logical BF16
weight bytes and retains the conservative NF4 reserved-workspace envelope**.
It is not a measured BF16 peak. NF4 profiler counts are not used to claim BF16
speed; dense matrix dimensions supply its analytical FLOPs.

A 24 GB card is a reasonable 1.3B target. The 3.3B BF16 estimate leaves limited
24 GB headroom, so **32 GB is the preferred trial target**. A100 memory is not
required by the estimate, and no A100 appeared under the $0.65/hour search cap.

## Corrected market search and price/speed plan

The GB-unit search returned 95 offers with 45 GB disk pricing. Repricing with
64 GB disk for the larger weights and SFT export returned 90 offers. Filters:
one verified GPU, reliability >99%, host RAM ≥32 GB, ≥12 effective CPU cores,
advertised download ≥300 Mbps, compute capability ≥8.0, outside China,
total hourly price ≤$0.65. Actual HF throughput was not measured.

| Practical option | Offer | Location | GPU RAM | With 64 GB disk |
|---|---:|---|---:|---:|
| Cheapest validated architecture family | 57832995 | Oregon, US | RTX 3090 24 GB | $0.218/h |
| Recommended 3.3B headroom/value candidate | 44160784 | Belgium | RTX PRO 4500 32 GB | $0.364/h |
| Ada alternative | 60310371 | Vietnam | RTX 4090 24 GB | $0.400/h |
| Faster advertised benchmark, higher price | 45171306 | South Korea | RTX 5090 32 GB | $0.551/h |

The PRO 4500 is Blackwell. Use a PyTorch build with native `sm_120` support and
CUDA 12.8+; the local CUDA 12.6 validation does not validate that stack.
[NVIDIA's compatibility guidance](https://docs.nvidia.com/cuda/archive/12.8.2/blackwell-compatibility-guide/index.html)
and [CUDA 12.8 support announcement](https://developer.nvidia.com/blog/cuda-toolkit-12-8-delivers-nvidia-blackwell-support/)
describe the requirements. Its performance has not been measured here; advertised
DLPerf is a screening proxy, not the framework's throughput.

Recommendation: start **1.3B BF16 on a 3090** for low cost; trial **3.3B BF16 on
the 32 GB PRO 4500** for memory margin after verifying the image. Both keep COMET
on the same GPU and SAC on OpenRouter. Keep the first live pilot short.

For the configured full schedule, conservative time/cost **scenarios** are:

| Plan | Time | GPU rental | API allowance | Combined |
|---|---:|---:|---:|---:|
| 1.3B, RTX 3090 | 24–41 h | $5.17–9.00 | $6.66–13.32 | $11.83–22.32 |
| 3.3B, RTX PRO 4500 | 31–54 h | $11.13–19.67 | $6.66–13.32 | $17.79–32.99 |

GPU-work time is scaled from the explicit 600M scenario by dense forward FLOPs;
it is not a hardware benchmark. Judge latency, caching, retries, long sequences
and model quality can change wall time and request volume substantially. API
prices/assumptions are documented in the [base plan](correction_training.md).
Storage is included; network, taxes, reasoning tokens and retries are excluded.

## Commands

From the project root in the existing Conda environment:

On a rental, verify a BF16 CUDA backward before downloading large weights:

```bash
python -c 'import torch; x=torch.randn(16,16,device="cuda",dtype=torch.bfloat16,requires_grad=True); (x@x).float().sum().backward(); print(torch.cuda.get_device_name(),torch.version.cuda,torch.cuda.get_arch_list())'
```

On Blackwell, confirm the image includes `sm_120` kernels. Then resolve the profiles:

```bash
python -m scripts.main.train --config configs/runs/nllb_1p3b_correction.yaml --dry-run
python -m scripts.main.train --config configs/runs/nllb_3p3b_correction.yaml --dry-run
```

On a fresh machine, create the compatibility manifests after preparing the
original 600M bounded split, using the model IDs and revisions above:

```bash
python -m scripts.data.filter_token_pairs --format nllb \
  --tokenizer facebook/nllb-200-distilled-1.3B \
  --model-name facebook/nllb-200-distilled-1.3B \
  --revision 7be3e24664b38ce1cac29b8aeed6911aa0cf0576 \
  --reuse-manifest data/token256_nllb/manifest.json --output data/token256_nllb_1p3b
python -m scripts.data.filter_token_pairs --format nllb \
  --tokenizer facebook/nllb-200-3.3B --model-name facebook/nllb-200-3.3B \
  --revision 1a07f7d195896b2114afcb79b7b57ab512e7b43e \
  --reuse-manifest data/token256_nllb/manifest.json --output data/token256_nllb_3p3b
```

Run a short real-service pilot on the chosen host before committing to the full
schedule; this uses live COMET/SAC and W&B keys from the environment:

```bash
python -m scripts.main.train --config configs/runs/nllb_3p3b_correction.yaml \
  --set output_dir=logs/nllb-3p3b-pilot \
  --set max_optimizer_steps.sft=2 --set max_optimizer_steps.rl=10 \
  --set evaluation_every_optimizer_steps=2 --set trainer.limit_val_batches=1
```

Replace `3p3b` with `1p3b` for that model's pilot. Full headless execution uses
the existing tmux launcher with the selected profile:

```bash
bash scripts/rental/start.sh --config configs/runs/nllb_3p3b_correction.yaml
tmux attach -t mt-train
```

Use distinct output paths for each model. Detach with `Ctrl-b, d`.
No instance lifecycle actions were performed for this report.

Artifacts:
[1.3B GPU trial](../diagnostics/correction_20261114/nllb_1p3b_gpu.json),
[3.3B NF4 GPU trial](../diagnostics/correction_20261114/nllb_3p3b_nf4_gpu.json),
[3.3B BF16 load limit](../diagnostics/correction_20261114/nllb_3p3b_bf16_load.json),
[1.3B cost](../diagnostics/correction_20261114/nllb_1p3b_cost.json),
[3.3B BF16 projection](../diagnostics/correction_20261114/nllb_3p3b_cost.json),
[latest 64 GB offer snapshot](../diagnostics/offers_wide_64gb_20261114.json).
