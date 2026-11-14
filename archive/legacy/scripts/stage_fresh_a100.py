"""Stage code, dataset and configuration only; download all models on the rental."""

import argparse
from pathlib import Path

from scripts.watch_vast_training import Supervisor, command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    args = parser.parse_args()
    supervisor = Supervisor(args.state)
    files = [
        Path("logs/fresh-a100-source.tar.gz"),
        Path("logs/training_20261108/training-data.tar.gz"),
        args.inputs / "rental_config.json",
    ]
    command([*supervisor.scp, *(str(path) for path in files), f"{supervisor.state['host']}:/workspace/"], 300)
    supervisor.remote("""
set -e
cd /workspace/grpo-mt
tar xzf /workspace/fresh-a100-source.tar.gz
tar xzf /workspace/training-data.tar.gz
mkdir -p logs/production
cp /workspace/rental_config.json logs/production/rental_config.json
printf '{"status":"preparing","stage":"fresh_sft"}\n' > logs/production/state.json
tmux new-session -d -s mt-setup -c /workspace/grpo-mt 'bash -o pipefail -c "bash scripts/setup_fresh_a100.sh 2>&1 | tee /workspace/fresh_a100_setup.log"'
tmux set-option -t mt-setup remain-on-exit on
tmux new-session -d -s mt-monitor 'watch -n 2 nvidia-smi'
""")
    print("Headless setup launched; models download directly from Hugging Face.", flush=True)


if __name__ == "__main__":
    main()
