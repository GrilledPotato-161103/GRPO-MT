"""Split the existing training pool into disjoint SFT/RL pairs before mirroring directions."""

import argparse
import csv
import hashlib
import json
import unicodedata
from pathlib import Path
from typing import Any, cast


def _normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    if not fields or not {"en", "vi"}.issubset(fields) or not rows:
        raise ValueError("Each stage input must contain nonempty en/vi CSV pairs")
    if any(
        not isinstance(row.get(key), str) or not row[key].strip()
        for row in rows
        for key in ("en", "vi")
    ):
        raise ValueError("Empty or non-text translation pair")
    return list(fields), cast(list[dict[str, str]], rows)


def prepare(
    train_file: Path,
    val_file: Path,
    test_file: Path,
    output: Path,
    sft_fraction: float = 0.8,
    seed: int = 42,
) -> dict[str, Any]:
    """Validate both languages, preserve held-out data, and write stage manifests."""
    if not 0 < sft_fraction < 1:
        raise ValueError("sft_fraction must be between zero and one")
    fields, train = _read(train_file)
    _, validation = _read(val_file)
    _, test = _read(test_file)
    if len(train) < 2:
        raise ValueError("Need at least two training pairs for SFT/RL splitting")
    for key in ("en", "vi"):
        pool = [_normalize(row[key]) for row in train]
        dev = {_normalize(row[key]) for row in validation}
        held_out = {_normalize(row[key]) for row in test}
        if len(pool) != len(set(pool)):
            raise ValueError(f"Duplicate {key} training texts need a grouped split")
        if set(pool) & (dev | held_out) or dev & held_out:
            raise ValueError(
                f"Normalized {key} overlap between train, validation, or test"
            )
    paths = {stage: output / f"{stage}.csv" for stage in ("sft", "rl")}
    manifest_path = output / "manifest.json"
    if any(path.exists() for path in (*paths.values(), manifest_path)):
        raise FileExistsError(
            "Stage outputs already exist; choose a new output directory"
        )
    if output.resolve() in {
        path.resolve() for path in (train_file, val_file, test_file)
    }:
        raise ValueError("Stage output must be a directory separate from input files")
    ordered = sorted(
        train,
        key=lambda row: hashlib.sha256(
            f"{seed}:{_normalize(row['en'])}\0{_normalize(row['vi'])}".encode()
        ).digest(),
    )
    sft_count = max(1, min(len(train) - 1, round(len(train) * sft_fraction)))
    partitions = {"sft": ordered[:sft_count], "rl": ordered[sft_count:]}
    output.mkdir(parents=True, exist_ok=True)
    for stage, rows in partitions.items():
        with paths[stage].open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    manifest: dict[str, Any] = {
        "seed": seed,
        "sft_fraction": sft_fraction,
        "counts": {
            "training_pool": len(train),
            "sft": sft_count,
            "rl": len(train) - sft_count,
            "validation": len(validation),
            "test_pairs": len(test),
            "test_directional_examples": 2 * len(test),
        },
        "inputs": {
            name: {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for name, path in (
                ("train", train_file),
                ("validation", val_file),
                ("test", test_file),
            )
        },
        "outputs": {
            name: {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for name, path in paths.items()
        },
        "normalized_en_vi_overlap": 0,
        "split_before_bidirectional_expansion_and_augmentation": True,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    """Prepare the SFT/RL files without modifying prepared development/test data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-file", type=Path, default=Path("data/phomt_prepared/train.csv")
    )
    parser.add_argument(
        "--val-file", type=Path, default=Path("data/phomt_prepared/validation.csv")
    )
    parser.add_argument(
        "--test-file", type=Path, default=Path("data/phomt_test_selected.csv")
    )
    parser.add_argument("--output", type=Path, default=Path("data/phomt_stages"))
    parser.add_argument("--sft-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.train_file,
                args.val_file,
                args.test_file,
                args.output,
                args.sft_fraction,
                args.seed,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
