"""Resume the interrupted first SFT epoch with measured batch sizes and unchanged Adam state."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from scripts.rebatch_checkpoint import migrate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("logs/production"))
    parser.add_argument("--deadline", type=float)
    parser.add_argument("--group")
    args = parser.parse_args()
    root = args.output.resolve()
    if (root / "rental_config.json").exists():
        configuration = json.loads((root / "rental_config.json").read_text())
        args.deadline = args.deadline or configuration["graceful_deadline_unix"]
        args.group = args.group or configuration["run_group"]
    if args.deadline is None or args.group is None:
        raise ValueError("Supply deadline/group or a rental_config.json")
    if time.time() >= args.deadline:
        raise RuntimeError("The authorized rental deadline has passed")
    batches = json.loads((root / "batch_sizes.json").read_text())
    if set(batches["selected"]) != {"sft", "rl"}:
        raise RuntimeError("Wait for both SFT and RL batch probes to finish")
    source = root / "sft/checkpoints/interrupted.ckpt"
    target = root / "sft/checkpoints/rebatch.ckpt"
    metadata_file = root / "rebatch_resume.json"
    if not target.exists():
        consumed = migrate(source, target)
        metadata_file.write_text(
            json.dumps(
                {"checkpoint": str(target), "consumed_examples": consumed}, indent=2
            )
        )
    if not metadata_file.exists():
        raise RuntimeError("Existing rebatch checkpoint has no resume metadata")
    cache = json.loads((root / "cache_manifest.json").read_text())
    os.environ.update(
        HF_HOME="/workspace/hf-cache",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TRAIN_CACHE_READY_REVISION=cache["policy_revision"],
        TRAIN_RUN_GROUP=args.group,
    )
    for key in ("WANDB_API_KEY", "HUGGING_FACE_API_KEY", "CLAUDE_AUTH_KEY"):
        os.environ.pop(key, None)
    (root / "STOP").unlink(missing_ok=True)
    (root / "state.json").write_text(json.dumps({"status": "resuming", "stage": "sft"}))
    os.execv(
        sys.executable,
        [
            sys.executable,
            "-m",
            "scripts.train_bilingual_rental",
            "--output",
            str(root),
            "--deadline",
            str(args.deadline),
        ],
    )


if __name__ == "__main__":
    main()
