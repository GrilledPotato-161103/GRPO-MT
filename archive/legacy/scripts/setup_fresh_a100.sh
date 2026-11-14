#!/usr/bin/env bash
set -euo pipefail
cd /workspace/grpo-mt
source /venv/main/bin/activate
export HF_HOME=/workspace/hf-cache TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export COMET_CHECKPOINT=/workspace/cometkiwi/checkpoints/model.ckpt
unset WANDB_API_KEY HUGGING_FACE_API_KEY CLAUDE_AUTH_KEY
trap 'code=$?; printf "{\"status\":\"failed\",\"stage\":\"setup\",\"exit_code\":%s}\n" "$code" > logs/production/state.json; exit "$code"' ERR
python -m pip install -e '.[lora,comet]' 'transformers==4.57.6' 'peft==0.18.1' 'lightning==2.6.6' 'wandb==0.24.0' > /workspace/install_training.log 2>&1
python -m pip check > /workspace/pip_check_training.log
python -u -c 'import json; from pathlib import Path; from huggingface_hub import snapshot_download; cfg=json.loads(Path("logs/production/rental_config.json").read_text()); snapshot_download("Qwen/Qwen2.5-7B-Instruct", revision=cfg["policy_revision"], allow_patterns=["*.json","*.safetensors","*.txt","*.jinja"])'
while [ ! -f /workspace/COMET_READY ]; do sleep 2; done
python -u -c 'from pathlib import Path; from scripts.train_bilingual_rental import cache_assets; cache_assets(Path("logs/production"))'
# Initialize fresh foundation weights and optimizers, with no old checkpoints.
bash scripts/start_a100_session.sh train
