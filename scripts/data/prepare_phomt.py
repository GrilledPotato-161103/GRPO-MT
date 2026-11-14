"""Prepare disjoint, deterministic PhoMT train/development files without overwriting originals."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


def prepare(
    train_file: Path,
    test_file: Path,
    output: Path,
    val_fraction: float = 0.05,
    seed: int = 42,
) -> dict[str, int]:
    """Split unique source rows; reject overlap rather than silently discard data."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between zero and one")
    with train_file.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        train = list(reader)
    with test_file.open(encoding="utf-8-sig", newline="") as stream:
        test = list(csv.DictReader(stream))
    if fields is None or len(train) < 2 or not test:
        raise ValueError("Need training and held-out data with a CSV header")

    def normalize(row: dict[str, str]) -> str:
        return " ".join(row["en"].casefold().split())

    train_sources = [normalize(row) for row in train]
    test_sources = {normalize(row) for row in test}
    if len(set(train_sources)) != len(train_sources):
        raise ValueError("Duplicate training sources require a grouped split")
    if set(train_sources) & test_sources:
        raise ValueError("Train/test source overlap detected; inspect before splitting")
    targets = [
        output / "train.csv",
        output / "validation.csv",
        output / "manifest.json",
    ]
    if any(path.exists() for path in targets):
        raise FileExistsError(
            "Prepared outputs already exist; choose a new output directory"
        )
    ordered = sorted(
        train,
        key=lambda row: hashlib.sha256(f"{seed}:{normalize(row)}".encode()).digest(),
    )
    n_val = max(1, min(len(train) - 1, round(len(train) * val_fraction)))
    output.mkdir(parents=True, exist_ok=True)
    for path, rows in ((targets[0], ordered[n_val:]), (targets[1], ordered[:n_val])):
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    counts = {
        "train": len(train) - n_val,
        "validation": n_val,
        "test": len(test),
        "overlap": 0,
    }
    manifest = {
        "counts": counts,
        "seed": seed,
        "input_train": str(train_file),
        "input_test": str(test_file),
        "train_sha256": hashlib.sha256(train_file.read_bytes()).hexdigest(),
        "test_sha256": hashlib.sha256(test_file.read_bytes()).hexdigest(),
    }
    targets[2].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return counts


def main() -> None:
    """Create prepared CSV files and a provenance manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-file", type=Path, default=Path("data/phomt_train_selected.csv")
    )
    parser.add_argument(
        "--test-file", type=Path, default=Path("data/phomt_test_selected.csv")
    )
    parser.add_argument("--output", type=Path, default=Path("data/phomt_prepared"))
    args = parser.parse_args()
    print(json.dumps(prepare(args.train_file, args.test_file, args.output), indent=2))


if __name__ == "__main__":
    main()
