"""Change batch accounting without changing adapter weights, Adam state, or global step."""

import argparse
import json
from pathlib import Path

import torch


def migrate(
    source: Path,
    output: Path,
    original_batch_size: int = 1,
    previous_examples: int = 0,
    total_examples: int | None = None,
) -> int:
    if output.exists():
        raise FileExistsError(output)
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    loop = checkpoint["loops"]["fit_loop"]
    processed = int(loop["epoch_loop.batch_progress"]["current"]["processed"])
    if original_batch_size < 1 or previous_examples < 0:
        raise ValueError("Invalid original batch size or previous example offset")
    if (
        original_batch_size == 1
        and previous_examples == 0
        and processed != checkpoint["global_step"]
    ):
        raise ValueError(
            "This migration requires the first SFT epoch at original batch size one"
        )
    consumed = previous_examples + processed * original_batch_size
    if total_examples is not None and (
        total_examples < 1 or consumed >= total_examples
    ):
        raise ValueError("Cannot rebatch a completed epoch")
    progress = loop["epoch_loop.batch_progress"]
    for section in ("total", "current"):
        progress[section] = {key: 0 for key in progress[section]}
    progress["is_last_batch"] = False
    checkpoint["rebatch_resume"] = {
        "original_checkpoint": str(source),
        "consumed_examples": consumed,
        "optimizer_and_weights_unchanged": True,
    }
    torch.save(checkpoint, output)
    return consumed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(
        json.dumps({"consumed_examples": migrate(arguments.source, arguments.output)})
    )
