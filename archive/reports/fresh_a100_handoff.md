# Fresh single-A100 training

Instance **59691795**, A100 SXM4 **40 GB**, **$0.401111/hour**. Fresh SFT → RL
from the pinned Qwen2.5-7B-Instruct foundation; no prior checkpoint is loaded.
The previous completed run and its results remain preserved separately.

```powershell
cd E:\IAI\Lab\Research\GRPO-MT
.\scripts\monitor_rental.ps1 train
.\scripts\monitor_rental.ps1 gpu
.\scripts\monitor_rental.ps1 status
```

Training is detached in `mt-train`; setup is in `mt-setup`, GPU display in
`mt-monitor`. Detach from a view with **Ctrl-b, then d**.

On the rental, once setup has completed:

```bash
cd /workspace/grpo-mt
bash scripts/start_a100_session.sh train
tail -n 30 /workspace/train_driver.log
```

One SFT epoch (batch **24**) then one RL epoch (batch **48**, group 2, two updates
per rollout). Prompt/source cap **256 tokens**, target cap **128**. BF16 rank-8
LoRA, gradient checkpointing, TransferAdamW; **COMETKiwi on GPU**, scorer batch 4.
Reward: 70% COMETKiwi + 30% BLEU. The prior passed ten-update A100 configuration
check established these batch/length limits; this run initializes fresh adapters
and optimizers from the foundation model and does not load old checkpoints.

**All model weights download directly from Hugging Face on the rental.** Only
code, dataset and configuration are uploaded. For gated COMETKiwi, the explicitly
authorized Hugging Face token is sent over encrypted SSH to temporary process
memory for that download; it is not saved to disk or logs.

Settings: `logs/production/rental_config.json`, resolved configs under
`logs/production/{sft,rl}/resolved_config.yaml`. Preset: `configs/rental_a100.json`.
Setup command: `bash scripts/setup_fresh_a100.sh`.

W&B project: [Machine Translation](https://wandb.ai/25025069-vietnam-national-university-hanoi/Machine%20Translation).
The local relay retains the API key. Checkpoints and logs are backed up under
`logs/training_a100_fresh_us/access/checkpoints_and_logs`; extra full-model downloads
are disabled. **Keep this PC online for W&B, backups, and automatic destruction.**
The original combined **$10 cap** remains in force; about $2 was spent before
this fresh rental. Both directions are evaluated on the full held-out test set.

Local code/data staging is also headless. Reusable staging command:

```powershell
python -X utf8 -m scripts.stage_fresh_a100 --state logs/training_a100_fresh_us/access/state.json --inputs logs/training_a100_fresh
```

Run staging only on an unprepared rental; it starts the `mt-setup` session and
requires the registered key and accepted SSH host key from the saved state.
For the approved gated download on this rental:

```powershell
python -X utf8 -m scripts.download_rental_comet --state logs/training_a100_fresh_us/access/state.json
```

Download status: `logs/training_a100_fresh_us/access/direct_hf.stdout.log`.
