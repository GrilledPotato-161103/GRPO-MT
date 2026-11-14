"""Download COMET directly on an authorized rental using an ephemeral HF token."""

import argparse
import os
import shlex
import subprocess
from pathlib import Path

from scripts.watch_vast_training import Supervisor
from src.utils.credentials import environment_secret

REMOTE_PROGRAM = r'''
import json, sys, time
from pathlib import Path
token = sys.stdin.readline().strip()
if not token:
    raise RuntimeError("Hugging Face access token is required")
try:
    while not Path("/workspace/pip_check_training.log").exists():
        state = Path("/workspace/grpo-mt/logs/production/state.json")
        if state.exists() and json.loads(state.read_text()).get("status") == "failed":
            raise RuntimeError("Dependency setup failed")
        time.sleep(2)
    from huggingface_hub import snapshot_download
    snapshot_download(
        "Unbabel/wmt22-cometkiwi-da",
        revision="1ad785194e391eebc6c53e2d0776cada8f83179a",
        token=token,
        allow_patterns=["checkpoints/model.ckpt", "hparams.yaml"],
        local_dir="/workspace/cometkiwi",
    )
    del token
    assert Path("/workspace/cometkiwi/checkpoints/model.ckpt").stat().st_size > 0
    Path("/workspace/COMET_READY").touch()
    print(json.dumps({"status":"downloaded_from_huggingface","model":"Unbabel/wmt22-cometkiwi-da"}), flush=True)
except Exception as error:
    print(json.dumps({"status":"download_failed","error_type":type(error).__name__}), flush=True)
    raise SystemExit(1)
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    args = parser.parse_args()
    supervisor = Supervisor(args.state)
    token = environment_secret("HUGGING_FACE_API_KEY")
    if not token:
        raise RuntimeError("HUGGING_FACE_API_KEY is required in the local user environment")
    environment = os.environ.copy()
    for name in ("HUGGING_FACE_API_KEY", "WANDB_API_KEY", "CLAUDE_AUTH_KEY"):
        environment.pop(name, None)
    script = (
        "HF_HOME=/workspace/hf-cache HF_ENDPOINT=https://huggingface.co "
        "/venv/main/bin/python -u -c " + shlex.quote(REMOTE_PROGRAM)
    )
    result = subprocess.run(
        [*supervisor.ssh, script], input=token + "\n", text=True,
        encoding="utf-8", errors="replace", capture_output=True,
        env=environment, timeout=7200, check=False,
    )
    del token
    if result.returncode:
        raise RuntimeError(f"Remote Hugging Face download failed (exit {result.returncode})")
    print(result.stdout, flush=True)
    (supervisor.root / "DIRECT_HF_DOWNLOAD_COMPLETE").touch()


if __name__ == "__main__":
    main()
