#!/usr/bin/env bash
set -euo pipefail
cd /workspace/grpo-mt
source /venv/main/bin/activate
export HF_HOME=/workspace/hf-cache TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
unset WANDB_API_KEY HUGGING_FACE_API_KEY CLAUDE_AUTH_KEY
python -m pip install -e '.[lora,comet]' 'transformers==4.57.6' 'peft==0.18.1' 'lightning==2.6.6' 'wandb==0.24.0' > /workspace/install_training.log 2>&1
python -m pip check > /workspace/pip_check_training.log
python -u -m scripts.restore_rental_cache
python -u -m scripts.prepare_a100_resume
while [ ! -f /workspace/COMET_READY ]; do sleep 2; done
bash scripts/start_a100_session.sh verify
printf 'Exactly-ten-update verification started in mt-train.\n'
