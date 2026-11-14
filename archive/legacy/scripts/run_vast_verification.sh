#!/usr/bin/env bash
# Run from the extracted source directory. No real feedback credentials are used.
set -euo pipefail
RESULTS="${1:-/workspace/grpo-results}"
mkdir -p "$RESULTS"
export HF_HOME=/workspace/hf-cache
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
python -m pip install -e . pytest pytest-cov > "$RESULTS/install.log" 2>&1
python - <<'PY' > "$RESULTS/hardware.json"
import json,platform,torch,transformers,lightning
print(json.dumps({'python':platform.python_version(),'torch':torch.__version__,
                  'transformers':transformers.__version__,'lightning':lightning.__version__,
                  'gpu':torch.cuda.get_device_name(),'cuda':torch.version.cuda},indent=2))
PY
python -m pytest tests/grpo -q -o addopts='' --cov --cov-report=term-missing --cov-report=json:"$RESULTS/coverage.json" --cov-report=xml:"$RESULTS/coverage.xml" --cov-fail-under=90 > "$RESULTS/pytest.log" 2>&1
python -m scripts.gpu_smoke --output "$RESULTS/gpu_smoke" --precision bf16-mixed > "$RESULTS/gpu_smoke.log" 2>&1
python -m scripts.benchmark_self_play --output "$RESULTS/pretrained_benchmark" --steps 3 --mock-feedback > "$RESULTS/pretrained_benchmark.log" 2>&1
printf 'All Vast.ai verification stages passed\n' > "$RESULTS/SUCCESS"
