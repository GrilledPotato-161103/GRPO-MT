#!/usr/bin/env bash
set -euo pipefail
cd /workspace/grpo-mt
source /venv/main/bin/activate
if [ -e /usr/lib/x86_64-linux-gnu/libcuda.so.1 ]; then
  export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so.1
fi
export HF_HOME=/workspace/hf-cache
export COMET_CHECKPOINT=/workspace/cometkiwi/checkpoints/model.ckpt
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
trap 'code=$?; printf "{\"status\":\"failed\",\"exit_code\":%s}\n" "$code" > logs/production/state.json; exit "$code"' ERR
mkdir -p logs/production
printf '{"status":"preparing"}\n' > logs/production/state.json
python -m pip install -e '.[lora,comet]' 'wandb==0.24.0'
python -m pip check
python -c 'import torch; print(torch.cuda.get_device_name()); print(torch.ones(1, device="cuda"))'
python -c 'import json; from pathlib import Path; from huggingface_hub import snapshot_download; revision=json.loads(Path("logs/production/rental_config.json").read_text())["policy_revision"]; snapshot_download("Qwen/Qwen2.5-7B-Instruct", revision=revision, allow_patterns=["*.json","*.safetensors","*.txt","*.model","*.jinja"])'
while [ ! -f /workspace/COMET_READY ]; do sleep 2; done
python -c 'from pathlib import Path; from scripts.train_bilingual_rental import cache_assets; cache_assets(Path("logs/production"))'
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
printf '{"status":"tuning"}\n' > logs/production/state.json
PROFILE_CHECKPOINT=logs/production/sft/checkpoints/interrupted.ckpt
if [ -f logs/production/sft/checkpoints/rebatch.ckpt ]; then
  PROFILE_CHECKPOINT=logs/production/sft/checkpoints/rebatch.ckpt
fi
if ! python -c 'import json; from pathlib import Path; data=json.loads(Path("logs/production/batch_sizes.json").read_text()); assert set(data["selected"])=={"sft","rl"}' 2>/dev/null; then
  python -u -m scripts.tune_batch_size --checkpoint "$PROFILE_CHECKPOINT" --output logs/production/batch_sizes.json
fi
if ! python -c 'import json; from pathlib import Path; assert "sft_token_budget" in json.loads(Path("logs/production/batch_sizes.json").read_text())' 2>/dev/null; then
  python -u -m scripts.tune_token_budget --checkpoint "$PROFILE_CHECKPOINT" --report logs/production/batch_sizes.json
fi
python -u -m scripts.resume_bilingual_rental --output logs/production
