"""Exercise both GRPO architectures on one GPU, including CLI eval and resume."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from scripts.verify.assets import create_assets


def run_smoke(output: Path, precision: str = "32-true") -> None:
    """Run synthetic-only GPU jobs; save logs, checkpoints, and runtime metadata."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required; refusing a CPU fallback")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    torch.set_num_threads(1)
    torch.manual_seed(42)
    metadata: dict[str, Any] = {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "precision": precision,
        "status": "running",
        "runs": [],
    }
    report = output / "report.json"
    report.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    try:
        for architecture in ("seq2seq", "causal"):
            asset = create_assets(output / f"{architecture}_model", architecture)
            common = [
                f"model.model_name_or_path={asset.as_posix()}",
                "model.group_size=2",
                "model.num_iterations=2",
                "model.max_new_tokens=4",
                "model.max_source_length=32",
                "trainer.accelerator=gpu",
                "trainer.devices=1",
                f"+trainer.precision={precision}",
                "extras.print_config=false",
            ]
            train_dir = output / f"{architecture}_train"
            commands = [
                (
                    "train",
                    [
                        "-m",
                        "src.train",
                        "--config-name",
                        "grpo",
                        *common,
                        "trainer.max_epochs=1",
                        "+trainer.limit_train_batches=1",
                        "+trainer.limit_val_batches=1",
                        "+trainer.limit_test_batches=1",
                        f"hydra.run.dir={train_dir.as_posix()}",
                    ],
                ),
                (
                    "eval",
                    [
                        "-m",
                        "src.eval",
                        "--config-name",
                        "grpo_eval",
                        *common,
                        "+trainer.limit_test_batches=1",
                        f"ckpt_path={(train_dir / 'checkpoints/last.ckpt').as_posix()}",
                        f"hydra.run.dir={(output / f'{architecture}_eval').as_posix()}",
                    ],
                ),
                (
                    "resume",
                    [
                        "-m",
                        "src.train",
                        "--config-name",
                        "grpo",
                        *common,
                        "trainer.max_epochs=2",
                        "+trainer.limit_train_batches=1",
                        "+trainer.limit_val_batches=1",
                        "+trainer.limit_test_batches=1",
                        f"ckpt_path={(train_dir / 'checkpoints/last.ckpt').as_posix()}",
                        f"hydra.run.dir={(output / f'{architecture}_resume').as_posix()}",
                    ],
                ),
            ]
            for stage, args in commands:
                log = output / f"{architecture}_{stage}.log"
                with log.open("w", encoding="utf-8") as stream:
                    result = subprocess.run(
                        [sys.executable, *args],
                        cwd=root,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        timeout=300,
                        check=False,
                    )
                metadata["runs"].append(
                    {
                        "architecture": architecture,
                        "stage": stage,
                        "exit_code": result.returncode,
                        "log": str(log),
                    }
                )
                if result.returncode:
                    raise RuntimeError(f"{architecture} {stage} failed; see {log}")
                print(f"PASS {architecture} {stage}", flush=True)
            checkpoint = torch.load(
                output / f"{architecture}_resume/checkpoints/last.ckpt",
                map_location="cpu",
                weights_only=False,
            )
            if checkpoint["global_step"] != 4:
                raise RuntimeError("Resume did not restore optimizer/global step")
        metadata["status"] = "passed"
    except Exception:
        metadata["status"] = "failed"
        raise
    finally:
        report.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    """Parse the GPU smoke test command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("logs/gpu_smoke"))
    parser.add_argument(
        "--precision", choices=["32-true", "bf16-mixed", "16-mixed"], default="32-true"
    )
    args = parser.parse_args()
    run_smoke(args.output, args.precision)


if __name__ == "__main__":
    main()
