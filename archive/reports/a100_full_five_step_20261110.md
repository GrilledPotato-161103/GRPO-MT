# Full-data pipeline verification — 10 November 2026

**Passed:** five SFT optimizer updates and five GRPO optimizer updates on a new
verified A100 PCIe 40 GB rental, instance **60031289**, Japan. Both stages used
`Qwen/Qwen2.5-7B-Instruct` with frozen-backbone rank-8 LoRA and real
`Unbabel/wmt22-cometkiwi-da` on GPU. BF16, gradient checkpointing, 16 data
workers, source/target caps 256/128, RL source batch 16, four candidates, and
logprob batch 8 were enabled. GRPO beta was fixed at **0.01**.

The pipeline stopped naturally at its requested step limits. GPU memory and
utilization returned to zero. The rental remains running under user control at
approximately **$0.43/hour including 60 GB disk**, plus any transfer charges.
No full-corpus training was started; no rental watchdog or automatic deletion
was installed.

## Evidence

- [Complete scalar verification report](../logs/a100_full_test_20261110/five_step_verification.json)
- [Full-data usage and configuration](full_dataset_training.md)
- [W&B SFT](https://wandb.ai/25025069-vietnam-national-university-hanoi/Machine%20Translation/runs/phomt-full-five-20261110-sft)
- [W&B GRPO](https://wandb.ai/25025069-vietnam-national-university-hanoi/Machine%20Translation/runs/phomt-full-five-20261110-rl)

Each stage had five successful timed optimizer calls, finite metrics, a saved
checkpoint, and validation covering 16 EN→VI and 16 VI→EN translations.
Per-update GRPO diagnostics contain updates 1 through 5 exactly. The official
test set was neither uploaded nor evaluated. The 192-pair functionality bundle
used disjoint samples from the newly prepared full-corpus partitions.

| Pilot validation | After five SFT updates | After five GRPO updates |
|---|---:|---:|
| Pooled corpus BLEU | 15.97 | 17.08 |
| Pooled corpus chrF++ | 41.49 | 42.02 |
| Macro COMETKiwi | 0.79394 | 0.79405 |
| Fixed segment reward | 0.60459 | 0.60455 |

These tiny, early checkpoints do **not** establish translation improvement or
convergence. The values verify evaluation plumbing and scale handling.

## Timing scope

Five SFT batches averaged **1.075 seconds/batch**. Three GRPO rollout batches
containing five total updates took approximately **55.64 seconds** in total,
or **11.13 seconds/update amortized**. Gradient-enabled policy forward and
backward phases averaged approximately **1.12** and **1.96 seconds/update**,
respectively. These short, cold-pilot measurements exclude model download,
setup and held-out validation and are not a full-data throughput guarantee.
The JSON report includes individual phase counts and timing summaries.

All **200 local automated tests passed** before the rental run. Ruff and mypy
checks passed. Structured criterion feedback, caching, group-confidence
handling, and adaptive-KL resume/quality guards have local functional tests;
the real-model rental test used the matched fixed-KL COMETKiwi/BLEU reward.
No paid LLM-judge calls were made and no EN–VI confidence calibration is claimed.

## Access

From `GRPO-MT` in PowerShell:

```powershell
.\scripts\monitor_rental.ps1 status
.\scripts\monitor_rental.ps1 train
```

The monitor uses your existing `~/.ssh/id_ed25519`. The completed test console
is retained in tmux session `mt-test`; detach with **Ctrl-b, then d**.

Direct endpoint: `root@202.122.49.242`, port `15786`. Remote output:
`/workspace/grpo-mt/logs/five-step`. Checkpoints:
`sft/checkpoints/last.ckpt` and `rl/checkpoints/last.ckpt` under that directory.
The complete dataset remains local; upload the prepared full-data splits before
using the full-training configuration on this rental.

Setup required correcting a missing package root marker and a concurrent
download progress-bar failure. Both occurred before any optimizer update;
downloaded assets were reused. The successful run used the corrected bundle.
