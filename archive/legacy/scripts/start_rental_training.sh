#!/usr/bin/env bash
set -euo pipefail
cd /workspace/grpo-mt
command -v tmux >/dev/null || { printf 'Install tmux before starting this script.\n'; exit 1; }
if tmux has-session -t mt-train 2>/dev/null; then
  printf 'Session already exists: tmux attach -t mt-train\n'
  exit 0
fi
if pgrep -f '[p]ython.*scripts\.(train_bilingual_rental|resume_bilingual_rental|tune_batch_size)' >/dev/null; then
  printf 'A training/tuning process is already active; refusing a duplicate.\n'
  exit 1
fi
tmux new-session -d -s mt-train -c /workspace/grpo-mt 'bash -o pipefail -c "bash scripts/run_training_session.sh 2>&1 | tee -a /workspace/train_driver.log"'
tmux set-option -t mt-train remain-on-exit on
tmux new-session -d -s mt-monitor 'watch -n 2 nvidia-smi' || true
printf 'Training: tmux attach -t mt-train\nGPU: tmux attach -t mt-monitor\nDetach: Ctrl-b then d\n'
