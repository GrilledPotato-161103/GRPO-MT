#!/usr/bin/env bash
set -euo pipefail
cd /workspace/grpo-mt
source /venv/main/bin/activate
if [ -e /usr/lib/x86_64-linux-gnu/libcuda.so.1 ]; then
  export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so.1
fi
export HF_HOME=/workspace/hf-cache
export COMET_CHECKPOINT=/workspace/cometkiwi/checkpoints/model.ckpt
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset WANDB_API_KEY HUGGING_FACE_API_KEY CLAUDE_AUTH_KEY
mode="${1:-verify}"
if [ "$mode" = verify ]; then
  checkpoint=$(python -c 'import json; print(json.load(open("logs/production/sft/result.json"))["checkpoint"])')
  python -u -m scripts.verify_a100_ten_steps --checkpoint "$checkpoint"
elif [ "$mode" = train ]; then
  python -u -c '
import json, os, sys, time
from pathlib import Path
from scripts.train_bilingual_rental import main
root = Path("logs/production").resolve()
cfg = json.loads((root / "rental_config.json").read_text())
if not cfg.get("fresh_start", False):
    report = json.loads(Path("logs/a100_verify10/report.json").read_text())
    assert report["status"] == "passed" and report["optimizer_steps"] == 10
assert time.time() < cfg["graceful_deadline_unix"], "Rental deadline has passed"
os.environ.update(TRAIN_CACHE_READY_REVISION=cfg["policy_revision"], TRAIN_RUN_GROUP=cfg["run_group"])
(root / "STOP").unlink(missing_ok=True)
sys.argv = ["train_bilingual_rental", "--output", str(root), "--deadline", str(cfg["graceful_deadline_unix"])]
main()
'
else
  printf 'Usage: bash scripts/run_a100_session.sh verify|train\n' >&2
  exit 2
fi
