"""Prepare full PhoMT splits and optional memory-mapped record indexes."""

import argparse
import json
from pathlib import Path

from scripts.data.prepare_full import prepare_full


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-file", type=Path, default=Path("data/full/PhoMT_training.csv")
    )
    parser.add_argument(
        "--test-file", type=Path, default=Path("data/full/PhoMT_test.csv")
    )
    parser.add_argument("--output", type=Path, default=Path("data/full_prepared"))
    parser.add_argument("--validation-pairs", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="Reuse existing prepared splits without rewriting them",
    )
    args = parser.parse_args()
    if args.index_only:
        manifest = json.loads(
            (args.output / "manifest.json").read_text(encoding="utf-8")
        )
    else:
        manifest = prepare_full(
            args.train_file,
            args.test_file,
            args.output,
            args.validation_pairs,
            seed=args.seed,
        )
    from src.data.components.indexed_translation import IndexedTranslationRows

    indexed = {}
    for name, path in [
        (name, args.output / f"{name}.csv") for name in ("sft", "rl", "validation")
    ] + [("test", args.test_file)]:
        rows = IndexedTranslationRows(path)
        indexed[name] = len(rows)
        print(json.dumps({"indexed": name, "pairs": len(rows)}), flush=True)
    expected = {
        name: manifest["counts"]["test_pairs" if name == "test" else name]
        for name in indexed
    }
    if indexed != expected:
        raise ValueError("Prepared data counts differ from the dataset manifest")


if __name__ == "__main__":
    main()
