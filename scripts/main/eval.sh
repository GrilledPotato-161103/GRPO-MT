#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
if [[ -z "${PYTHON_BIN:-}" ]]; then
    for activate in ./venv/bin/activate ../venv/bin/activate /venv/main/bin/activate; do
        if [[ -f "$activate" ]]; then source "$activate"; break; fi
    done
fi
exec "${PYTHON_BIN:-python}" -u -m scripts.main.eval "$@"
