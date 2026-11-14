#!/usr/bin/env bash
set -euo pipefail
cd /workspace/grpo-mt
mkdir -p logs/production
trap 'code=$?; printf "{\"status\":\"failed\",\"exit_code\":%s}\n" "$code" > logs/production/state.json; exit "$code"' ERR
export HF_HOME=/workspace/hf-cache
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export WANDB_DISABLE_CODE=true
export WANDB_DISABLE_GIT=true
export COMET_CHECKPOINT=/workspace/cometkiwi/checkpoints/model.ckpt
if [ -e /usr/lib/x86_64-linux-gnu/libcuda.so.1 ]; then
  export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so.1
fi
source /venv/main/bin/activate
python -c 'import torch; print(torch.__version__, torch.cuda.get_device_name()); print(torch.ones(1, device="cuda"))'
python -m pip install -e '.[lora,comet]' 'wandb==0.24.0' > /workspace/install_training.log 2>&1
python -m pip check > /workspace/pip_check_training.log 2>&1
python -m scripts.train_bilingual_rental --output /workspace/grpo-mt/logs/production --deadline "$TRAIN_DEADLINE_UNIX"
