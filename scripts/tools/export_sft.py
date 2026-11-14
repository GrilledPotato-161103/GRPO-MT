"""Export a trusted SFT Lightning checkpoint as the base for the RL stage."""

import argparse
from pathlib import Path

from src.models.sft_module import export_sft


def main() -> None:
    """Export dense weights, merging adapters after unquantized CPU loading."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--load-dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16"
    )
    args = parser.parse_args()
    export_sft(args.checkpoint, args.output, args.load_dtype)


if __name__ == "__main__":
    main()
