# Prepared Vast.ai GPU verification

LoRA/QLoRA verification was subsequently completed for Qwen2.5-7B-Instruct on an RTX 4090. See the [adapter capacity guide](lora_qwen7b.md) and [measured results, costs, and verified cleanup](vast_lora_results_2026-11-08.md). The adapter runner is `scripts/run_lora_verification.sh`; it uses synthetic data and makes no feedback calls.

State: completed. After the credential was refreshed, the authorized RTX 4090 instance 59558218 passed the coverage/functionality suite, BF16 CLI smoke checks, and a small pretrained-policy self-play pilot. Artifacts were downloaded and the instance was destroyed. See [measured results and cost evidence](vast_results_2026-11-07.md).

The local conda environment is `E:/IAI/Lab/Research/venv`, adjacent to the GRPO-MT checkout. On Windows its executable is `../venv/python.exe`; it is not `venv/Scripts/python.exe`. All framework validation uses that interpreter.

## Package and verify locally

From the GRPO-MT project directory:

```powershell
& ../venv/python.exe -m pytest tests/grpo -q -o addopts=''
& ../venv/python.exe -m scripts.gpu_smoke --output logs/gpu_smoke
& ../venv/python.exe -m scripts.package_gpu_test --output logs/grpo-gpu-test.tar.gz
```

The bundle includes source, configs, tests, dependency metadata, documentation, and synthetic example JSONL only. It excludes the PhoMT CSV files, Git history, credentials, caches, and local environments. Smoke models are created from tiny T5/GPT-2 configs and saved locally; the test does not download model weights.

`scripts.gpu_smoke` requires CUDA, tests both architectures through the Hydra train/eval CLIs, resumes each checkpoint to four optimizer steps, and writes `report.json` plus six logs and checkpoints. Each subprocess has a five-minute timeout. Passing these tests demonstrates training infrastructure, not MT quality or the memory requirements of a real translation checkpoint.

## Procedure for a future authorized verification run

Use the `vastai` binary already on PATH. Do not install another Vast CLI or call the API directly. Account credentials should be configured outside source control. Read the installed skill at `C:/Users/Admin/.codex/skills/vastai/SKILL.md` before executing the rental lifecycle.

1. Confirm authentication/balance and inspect current instances without changing existing resources: `vastai show user --raw` and `vastai show instances-v1 --limit 25 --raw`.
2. Confirm a usable local SSH private key and register its **public key contents** before launching: `vastai create ssh-key '<actual public key contents>' --raw`. Never pass a key-file path as the public-key argument.
3. Search with the approved hourly cap and one verified GPU. Example query structure: `vastai search offers 'num_gpus=1 gpu_ram>=16 verified=true rentable=true direct_port_count>=1 reliability>0.95 dph_total<=APPROVED_CAP' --storage 20 -o dph_total --limit 5 --raw`. Replace APPROVED_CAP with the approved numeric limit; include disk and transfer fees in the total estimate. An empty result must be reported before relaxing filters.
4. Select a current offer; record offer ID, quoted hourly price, storage price, start time, approved budget, and a firm cleanup deadline. Only then use `vastai create instance OFFER_ID --image vastai/pytorch:@vastai-automatic-tag --disk 20 --ssh --direct --cancel-unavail --label grpo-mt-smoke --raw`.
5. Record the returned `new_contract`. Poll `vastai show instance INSTANCE_ID --raw` with a finite provisioning timeout. Inspect actual/intended/current/next state. Destroy on terminal failure, a stopped target, or timeout.
6. Read the actual direct SSH address/port from instance JSON. Keep known-host verification enabled. If SSH fails, retrieve `vastai logs INSTANCE_ID --tail 100 --raw` before any recovery attempt.
7. Upload only `logs/grpo-gpu-test.tar.gz`; extract under `/workspace/grpo-mt`. Use the image's CUDA-enabled Python environment; install `pip install -e . pytest pytest-cov` without replacing its CUDA torch build unnecessarily.
8. Run `python -m pytest tests/grpo -q -o addopts=''` followed by `python -m scripts.gpu_smoke --output /workspace/grpo-results`. Optionally test bf16 on supported hardware with a separate output directory. Abort when the approved deadline/budget is reached.
9. Download report, logs, and any needed checkpoints while the instance is running. Record actual GPU, software versions, exit codes, runtime, and charges separately from price estimates.
10. Destroy **only the instance created for this test**, using `vastai destroy instance INSTANCE_ID -y --raw`. Stopping leaves disk charges; destruction is the cleanup action. Verify removal using the CLI. Preserve cleanup evidence locally.

No launch command is executed by the packaging or smoke scripts. The completed rental has been cleaned up; a future rental is a separate action. Use `tests/grpo` for coverage and functionality and a representative real-feedback pilot to generate `timing.json` before pricing full training. See [self-play configuration](self_play.md).
