# Full PhoMT training on A100

**Latest status, November 11:** all 10 cycles and final official-test evaluation completed. Instance 60031289 was destroyed after verified local downloads. See [the final report](final_test_and_cleanup_20261111.md) for checkpoints, predictions and all metric logs.

**Update, November 10:** training was stopped at the user's request at SFT step **24,393**, before GRPO. The resumable checkpoint is `sft/checkpoints/interrupted.ckpt`; a local copy is saved under `logs/a100_full_test_20261110/stopped-checkpoint/`. The `STOP` file remains present to prevent automatic continuation. The `mt-train` pane has exited. Evaluation of this checkpoint runs separately in `mt-eval`; the rental remains available under user control.

Instance: **60031289**, Japan, one A100 PCIe 40 GB, approximately **$0.43/hour** including the allocated disk. The user controls the rental. No watchdog, timed shutdown, or automatic destruction runs.

Handoff verified after **17 successful SFT optimizer updates**. The tmux training pane remained alive, GPU utilization was **100%**, and VRAM use was **27,663 / 40,960 MiB** at that observation. Rich was enabled and the backbone was frozen. The observation connection has closed; training continues independently. GRPO follows after the full SFT epoch finishes.

The detached **`mt-train`** session runs a fresh **SFT → GRPO** pipeline from Hugging Face Qwen2.5-7B-Instruct foundation weights, independently of the five-step pilot. Rich progress remains visible when attached. Both stages use frozen-backbone rank-8 LoRA, BF16, source/target caps of 256/128, and 16 persistent data-loader workers. SFT uses batch 24; RL uses batch 16, four candidates, two updates per rollout, and log-probability microbatches of eight.

Data: 2,373,127 SFT pairs, 593,282 separate RL pairs, and 2,000 validation pairs, with both translation directions used. One epoch runs per stage. The unchanged 19,151-pair official test set is reserved for explicit evaluation. Validation reports COMETKiwi, BLEU and chrF++ every **10,000 optimizer updates**, plus final validation. The training reward is 70% COMETKiwi and 30% sentence BLEU; the uncalibrated LLM feedback harness is disabled.

From Windows PowerShell in `GRPO-MT`:

```powershell
# Attach to the live Rich progress display.
.\scripts\monitor_rental.ps1 train

# Show GPU utilization in a separate session.
.\scripts\monitor_rental.ps1 gpu

# Read status without attaching.
.\scripts\monitor_rental.ps1 status
```

Detach with **Ctrl-b, then d**. Detaching keeps training running. **Ctrl-C interrupts training.**

Direct access using your existing key:

```powershell
ssh -t -i "$env:USERPROFILE/.ssh/id_ed25519" -p 15786 root@202.122.49.242 "tmux attach -t mt-train"
```

W&B: [SFT live charts](https://wandb.ai/25025069-vietnam-national-university-hanoi/Machine%20Translation/runs/phomt-full-20261110-sft). The pipeline creates [the RL run](https://wandb.ai/25025069-vietnam-national-university-hanoi/Machine%20Translation/runs/phomt-full-20261110-rl) after SFT finishes.

Remote files:

- Configuration: `/workspace/full-run.yaml`
- Output: `/workspace/grpo-mt/logs/full-phomt-20261110`
- Console: `/workspace/full_console.log`
- Checkpoints: `sft/checkpoints/` and `rl/checkpoints/` under the output directory; periodic checkpoints every 100 updates.
- Scalar timing evidence: `sft/phase_timing/events.jsonl` and `rl/phase_timing/events.jsonl` under the output directory.

Inside the rental, request a checkpointed stop without stopping the instance:

```bash
touch /workspace/grpo-mt/logs/full-phomt-20261110/STOP
```

The reproducible manual launch command is below. Do not launch another copy while `mt-train` is running. The active process already has its W&B key in memory; a later manual restart needs `WANDB_API_KEY` exported securely again.

```bash
cd /workspace/grpo-mt
source /venv/main/bin/activate
python -u -m scripts.main.train --config /workspace/full-run.yaml
```

Successful handoff verification is saved locally in `logs/a100_full_test_20261110/full-handoff-verification.json` and remotely in the output directory as `handoff_verification.json`. Earlier initialization observations remain in `full-latest-status.json` and are superseded by the handoff report. The initial preparation attempt was interrupted with **zero optimizer updates** to enable Rich; its files were preserved under `logs/full-phomt-20261110-initialization` on the rental.
