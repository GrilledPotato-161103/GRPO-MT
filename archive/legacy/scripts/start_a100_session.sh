#!/usr/bin/env bash
set -euo pipefail
cd /workspace/grpo-mt
mode="${1:-verify}"
case "$mode" in verify|train) ;; *) printf 'Usage: %s verify|train\n' "$0"; exit 2;; esac
if tmux has-session -t mt-train 2>/dev/null; then
  if [ "$(tmux display-message -p -t mt-train '#{pane_dead}')" != 1 ]; then
    printf 'Training or verification is active: tmux attach -t mt-train\n'
    exit 1
  fi
  tmux kill-session -t mt-train
fi
tmux new-session -d -s mt-train -c /workspace/grpo-mt "bash -o pipefail -c 'bash scripts/run_a100_session.sh $mode 2>&1 | tee -a /workspace/train_driver.log'"
tmux set-option -t mt-train remain-on-exit on
tmux has-session -t mt-monitor 2>/dev/null || tmux new-session -d -s mt-monitor 'watch -n 2 nvidia-smi'
printf 'Session: tmux attach -t mt-train\nDetach: Ctrl-b, then d\n'
