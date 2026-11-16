"""Select a fixed, locally prepared 1,000-pair official-test panel."""

import csv
import hashlib
import json
from pathlib import Path

from scripts.data.prepare_stages import _normalize


def sample(source: Path, output: Path, seed: int = 20261116, count: int = 1000):
    with source.open(encoding="utf-8", newline="") as handle:
        original = list(csv.DictReader(handle))
    unique = {}
    for row in original:
        unique.setdefault((_normalize(row["en"]), _normalize(row["vi"])), row)
    rows = sorted(
        unique.values(),
        key=lambda r: hashlib.sha256(
            (f"{seed}:test:" + r["en"] + "\0" + r["vi"]).encode()
        ).digest(),
    )
    if len(rows) < count:
        raise ValueError("Insufficient distinct complete held-out pairs")
    with output.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["en", "vi"])
        writer.writeheader()
        writer.writerows({k: r[k] for k in ("en", "vi")} for r in rows[:count])
    return {
        "path": str(output.resolve()),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "pairs": count,
        "directional_records": 2 * count,
        "seed": seed,
        "method": "Seeded SHA256 permutation, matching the SFT sampling algorithm with a test namespace",
        "source": str(source.resolve()),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_pairs": len(original),
        "distinct_normalized_pairs": len(rows),
        "quality_based_selection": False,
        "complete_token_bound_preserved": True,
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    data = root / "data/nllb_10k_fp16_20261116/bounded"
    record = sample(data / "test_bounded.csv", data / "test_sample_1000.csv")
    for path in (
        data / "manifest.json",
        root / "runs/nllb-10k-fp16-20261116/dataset_manifest.json",
    ):
        manifest = json.loads(path.read_text())
        manifest["outputs"]["test_sample"] = record
        manifest["counts"]["test_sample"] = record["pairs"]
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (data / "test_sample_1000_manifest.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    print(json.dumps(record, indent=2))
