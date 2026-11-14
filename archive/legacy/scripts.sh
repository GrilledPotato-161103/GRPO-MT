#!/usr/bin/env bash
# Run from an attached tmux session on an A100 rental: bash scripts.sh
# Environment overrides: TRAIN_HOURS=2 SFT_BATCH_SIZE=24 RL_BATCH_SIZE=16
# GROUP_SIZE=4 LOGPROB_BATCH_SIZE=8 EVAL_EVERY=10000 SKIP_INSTALL=1
# WANDB_MODE=online needs WANDB_API_KEY; offline uses the existing local relay.
# HUGGING_FACE_API_KEY is needed only if gated COMET weights are not cached.
# A training deadline saves checkpoints and stops GPU work, not rental billing.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export RUN_DIR="${RUN_DIR:-$PWD/logs/production}"

if [[ "${1:-train}" == status ]]; then
    tmux list-sessions 2>/dev/null || true
    cat "$RUN_DIR/state.json" 2>/dev/null || true
    tail -n 12 /workspace/train_driver.log 2>/dev/null || true
    exit 0
fi

mkdir -p "$RUN_DIR"
exec 9>"$RUN_DIR/manual-training.lock"
flock -n 9 || { echo 'Another scripts.sh process is running.'; exit 1; }
if pgrep -f '[p]ython.*scripts.train_bilingual_rental' >/dev/null; then
    echo 'Training is already running. View it with: tmux attach -t mt-train'
    exit 0
fi
if tmux has-session -t mt-setup 2>/dev/null && [[ "$(tmux display-message -p -t mt-setup '#{pane_dead}')" != 1 ]]; then
    echo 'Automatic setup is active: tmux attach -t mt-setup'
    echo 'To take over preparation, stop that setup session first, then rerun scripts.sh.'
    exit 0
fi
if [[ -f "$RUN_DIR/sft/result.json" || -f "$RUN_DIR/rl/result.json" || -d exports/phomt_sft ]]; then
    echo 'Existing trained outputs detected; refusing to overwrite them. Resume from a saved checkpoint instead.'
    exit 1
fi

for activate in ./venv/bin/activate ../venv/bin/activate /venv/main/bin/activate; do
    if [[ -f "$activate" ]]; then source "$activate"; break; fi
done
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
export COMET_CHECKPOINT="${COMET_CHECKPOINT:-/workspace/cometkiwi/checkpoints/model.ckpt}"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export WANDB_PROJECT='Machine Translation'
export WANDB_DISABLE_CODE=true WANDB_DISABLE_GIT=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE TRAIN_CACHE_READY_REVISION

# Existing managed rentals retain their offline relay; standalone runs log online.
if [[ -z "${WANDB_MODE:-}" ]]; then
    if [[ -f "$RUN_DIR/rental_config.json" ]]; then export WANDB_MODE=offline
    else export WANDB_MODE=online; fi
fi
if [[ "$WANDB_MODE" != offline && -z "${WANDB_API_KEY:-}" ]]; then
    [[ -t 0 ]] || { echo 'Set WANDB_API_KEY or WANDB_MODE=offline before starting headless.'; exit 1; }
    read -r -s -p 'W&B API key (hidden): ' WANDB_API_KEY; printf '\n'
    export WANDB_API_KEY
fi
if [[ ! -s "$COMET_CHECKPOINT" && -z "${HUGGING_FACE_API_KEY:-}" ]]; then
    [[ -t 0 ]] || { echo 'Set HUGGING_FACE_API_KEY before starting headless.'; exit 1; }
    read -r -s -p 'Hugging Face token (hidden): ' HUGGING_FACE_API_KEY; printf '\n'
    export HUGGING_FACE_API_KEY
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if [[ "${SKIP_INSTALL:-0}" != 1 ]]; then
    python -m pip install -e '.[lora,comet]' 'transformers==4.57.6' 'peft==0.18.1' \
        'lightning==2.6.6' 'wandb==0.24.0' 'setuptools>=65,<81'
fi

# Existing prepared 80/20 partitions and held-out files must already be present.
python - <<'PY'
import json, os, time
from pathlib import Path
root = Path(os.environ['RUN_DIR'])
path = root / 'rental_config.json'
cfg = json.loads(path.read_text()) if path.exists() else {}
for name in ('data/phomt_stages/manifest.json', 'data/phomt_stages/sft.csv',
             'data/phomt_stages/rl.csv', 'data/phomt_prepared/validation.csv',
             'data/phomt_test_selected.csv'):
    if not Path(name).is_file(): raise SystemExit(f'Missing prepared data: {name}')
deadline = min(float(cfg.get('graceful_deadline_unix', float('inf'))),
               time.time() + float(os.environ.get('TRAIN_HOURS', '2')) * 3600)
if deadline <= time.time(): raise SystemExit('Existing authorized rental deadline has passed; not extending it.')
cfg.update(fresh_start=True, start_stage='sft',
    policy_revision='a09a35458c702b33eeacc393d103063234e8bc28', source_cap=256, target_cap=128,
    batch_sizes={'sft': int(os.environ.get('SFT_BATCH_SIZE','24')), 'rl': int(os.environ.get('RL_BATCH_SIZE','16'))},
    experiments={'sft':'phomt_sft', 'rl':'phomt_a100_train'},
    comet_device='cuda:0', comet_batch_size=4, warmup_steps={'sft':100,'rl':8},
    length_batching=True, logprob_batch_size=int(os.environ.get('LOGPROB_BATCH_SIZE','8')),
    logprob_token_budget=3072, group_size=int(os.environ.get('GROUP_SIZE','4')), num_iterations=2,
    evaluation_every_optimizer_steps=int(os.environ.get('EVAL_EVERY','10000')),
    log_every_n_steps=1, use_token_budget=False,
    wandb_offline=os.environ['WANDB_MODE']=='offline', graceful_deadline_unix=deadline,
    run_group=cfg.get('run_group', time.strftime('phomt-manual-a100-%Y%m%d-%H%M%S')))
path.write_text(json.dumps(cfg, indent=2))
print(f"SFT→GRPO; evaluation every {cfg['evaluation_every_optimizer_steps']} optimizer updates, plus final validation/test.")
PY

if [[ ! -s "$COMET_CHECKPOINT" ]]; then
    python - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download
token = os.environ.pop('HUGGING_FACE_API_KEY')
snapshot_download('Unbabel/wmt22-cometkiwi-da',
    revision='1ad785194e391eebc6c53e2d0776cada8f83179a', token=token,
    allow_patterns=['checkpoints/model.ckpt','hparams.yaml'],
    local_dir=str(Path(os.environ['COMET_CHECKPOINT']).parent.parent))
del token
PY
fi
unset HUGGING_FACE_API_KEY HF_TOKEN HUGGINGFACEHUB_API_TOKEN
export TRAIN_RUN_GROUP="$(python -c 'import json,os;from pathlib import Path;print(json.loads((Path(os.environ["RUN_DIR"])/"rental_config.json").read_text())["run_group"])')"
deadline="$(python -c 'import json,os;from pathlib import Path;print(json.loads((Path(os.environ["RUN_DIR"])/"rental_config.json").read_text())["graceful_deadline_unix"])')"
python -u -m scripts.train_bilingual_rental --output "$RUN_DIR" --deadline "$deadline" 2>&1 | tee -a "$RUN_DIR/manual-train.log"
