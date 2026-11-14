#!/usr/bin/env bash
# Run inside the rented CUDA environment. Uses synthetic test data only.
set -euo pipefail
RESULTS="${1:-/workspace/lora-results}"
mkdir -p "$RESULTS"
export HF_HOME=/workspace/hf-cache
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
python - <<'PY' > "$RESULTS/cuda_preflight.log" 2>&1
import torch
print(torch.__version__, flush=True)
print(torch.cuda.get_device_name(), flush=True)
print(torch.ones(1, device='cuda'), flush=True)
PY
python -m pip install -e '.[qlora]' pytest pytest-cov > "$RESULTS/install.log" 2>&1
python -m pytest tests/grpo -q -o addopts='' --cov --cov-report=json:"$RESULTS/coverage.json" --cov-fail-under=90 > "$RESULTS/pytest.log" 2>&1
python - <<'PY' > "$RESULTS/hardware.json"
import json,platform,torch,transformers,peft,bitsandbytes
print(json.dumps({'python':platform.python_version(),'torch':torch.__version__,
                  'transformers':transformers.__version__,'peft':peft.__version__,
                  'bitsandbytes':bitsandbytes.__version__,'gpu':torch.cuda.get_device_name(),
                  'cuda':torch.version.cuda},indent=2))
PY
for MODE in lora qlora; do
  if python -m scripts.profile_adapter_memory --mode "$MODE" --source-tokens 512 --completion-tokens 64 --output "$RESULTS/$MODE" > "$RESULTS/$MODE.log" 2>&1; then
    printf '0\n' > "$RESULTS/$MODE.exit"
  else
    printf '%s\n' "$?" > "$RESULTS/$MODE.exit"
  fi
done
touch "$RESULTS/COMPLETE"
