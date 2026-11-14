#!/usr/bin/env bash
# Only synthetic examples and a pre-downloaded COMETKiwi artifact are used.
set -euo pipefail
RESULTS="${1:-/workspace/translation-results}"
COMET_CHECKPOINT="${2:-/workspace/cometkiwi/checkpoints/model.ckpt}"
MODEL_READY="${3:-}"
mkdir -p "$RESULTS"
export HF_HOME=/workspace/hf-cache
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
if [ -e /usr/lib/x86_64-linux-gnu/libcuda.so.1 ]; then
  export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so.1
fi
python - <<'PY' > "$RESULTS/cuda_preflight.log" 2>&1
import torch
print(torch.__version__)
print(torch.cuda.get_device_name())
print(torch.ones(1, device='cuda'))
PY
python -m pip install -e '.[qlora,comet]' pytest pytest-cov > "$RESULTS/install.log" 2>&1
python -m pip check > "$RESULTS/pip_check.log" 2>&1
python -m pytest tests/grpo -q -o addopts='' --cov --cov-report=json:"$RESULTS/coverage.json" --cov-fail-under=90 > "$RESULTS/pytest.log" 2>&1
python - <<'PY' > "$RESULTS/hardware.json"
import importlib.metadata as metadata,json,platform,torch
print(json.dumps({'python':platform.python_version(),'torch':torch.__version__,
    'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),
    'gpu_gib':torch.cuda.get_device_properties(0).total_memory/2**30,
    'packages':{name:metadata.version(name) for name in ['lightning','transformers','peft','bitsandbytes','unbabel-comet','numpy','torchmetrics']}},indent=2))
PY
if [ -n "$MODEL_READY" ]; then
  MODEL_DEADLINE=$((SECONDS + 900))
  until [ -f "$MODEL_READY" ]; do
    if [ "$SECONDS" -ge "$MODEL_DEADLINE" ]; then
      printf 'Timed out waiting for the complete model artifact\n' >&2
      exit 1
    fi
    sleep 2
  done
fi
test -f "$COMET_CHECKPOINT"
python -m scripts.verify_cometkiwi --model "$COMET_CHECKPOINT" --device cuda:0 --output "$RESULTS/cometkiwi_real.json" --cache-dir /workspace/stage-work/comet-cache > "$RESULTS/cometkiwi_real.log" 2>&1
for MODE in full lora qlora; do
  python -m scripts.benchmark_translation_stages --mode "$MODE" --steps 3 --comet-model "$COMET_CHECKPOINT" --output "$RESULTS/benchmarks/$MODE" --work "/workspace/stage-work/$MODE" > "$RESULTS/$MODE.log" 2>&1
done
python -m scripts.summarize_stage_profiles --root "$RESULTS/benchmarks" --output "$RESULTS/policy_speed.csv" > "$RESULTS/summary.log" 2>&1
touch "$RESULTS/COMPLETE"
