"""Publish rental telemetry through the explicitly authorized local W&B key."""

import json
import os
import shlex
import subprocess
import time
import winreg

import wandb

from scripts.rental.wandb_relay import WandbRelay
from scripts.tools.manage_nllb_10k_rental import OUT, ROOT, connection


def main():
    with winreg.OpenKey(
        winreg.HKEY_LOCAL_MACHINE,
        r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
    ) as registry:
        key, _ = winreg.QueryValueEx(registry, "WANDB_API_KEY")
    os.environ["WANDB_API_KEY"] = str(key).strip()
    config = json.loads(
        (ROOT / "configs/rental/nllb_10k_fp16_20261116.json").read_text()
    )
    directory = OUT / "live_wandb"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "run_config.json").write_text(json.dumps(config))
    relay = WandbRelay(directory, config["wandb"]["group"], config["wandb"]["entity"])
    ssh, *_ = connection(60824538)
    code = """import json
from pathlib import Path
r=Path('/workspace/nllb-10k-fp16/logs/nllb-1p3b-10k-fp16-20261116')
names=['worker_status.json']
for stage in ['sft','rl']:
 names += [stage+'/'+n for n in ['progress.jsonl','translation_metrics/evaluation_history.jsonl','phase_timing/summary.json','result.json']]
print(json.dumps({n:(r/n).read_text() for n in names if (r/n).is_file()}))
"""
    command = [*ssh, "/venv/main/bin/python -c " + shlex.quote(code)]
    try:
        while True:
            try:
                response = subprocess.run(
                    command, capture_output=True, text=True, timeout=50, check=True
                )
                files = json.loads(response.stdout)
                for name, content in files.items():
                    path = directory / "checkpoints_and_logs" / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(content, encoding="utf-8")
                relay.sync()
                urls = {
                    s: (directory / f"wandb_{s}_url.txt").read_text().strip()
                    for s in ["sft", "rl"]
                    if (directory / f"wandb_{s}_url.txt").exists()
                }
                status = {
                    "alive": True,
                    "updated_unix": time.time(),
                    "urls": urls,
                    "last_steps": relay.steps,
                    "completed": sorted(relay.completed),
                    "transport": "native offline WandbLogger + live local telemetry relay",
                }
                (OUT / "live_wandb_status.json").write_text(
                    json.dumps(status, indent=2)
                )
                print(json.dumps(status), flush=True)
                remote = json.loads(files.get("worker_status.json", "{}"))
                if remote.get("phase") == "complete" and {"sft", "rl"}.issubset(
                    relay.completed
                ):
                    break
            except (
                OSError,
                ValueError,
                subprocess.SubprocessError,
                wandb.Error,
            ) as error:
                print(
                    json.dumps(
                        {"relay_retry": True, "error_type": type(error).__name__}
                    ),
                    flush=True,
                )
            time.sleep(20)
    finally:
        relay.finish()


if __name__ == "__main__":
    main()
