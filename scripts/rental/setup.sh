#!/usr/bin/env bash
# Run inside a PyTorch image; keep the image's installed CUDA/Torch packages.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
if [[ -z "${PYTHON_BIN:-}" ]]; then
    for activate in ./venv/bin/activate ../venv/bin/activate /venv/main/bin/activate; do
        if [[ -f "$activate" ]]; then source "$activate"; break; fi
    done
fi
"${PYTHON_BIN:-python}" -m pip install --no-cache-dir -r requirements.txt
"${PYTHON_BIN:-python}" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(torch.cuda.get_device_name(), torch.__version__)'
command -v tmux >/dev/null || { echo 'Install tmux: apt-get update && apt-get install -y tmux'; exit 1; }
"${PYTHON_BIN:-python}" -m scripts.main.train --dry-run >/dev/null
echo 'Environment ready. Set service keys, then run bash scripts/rental/start.sh.'
