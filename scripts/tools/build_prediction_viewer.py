"""Build a dependency-free, offline MT prediction review page from a CSV."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


DEFAULT_CSV = (
    Path(__file__).resolve().parents[2]
    / "runs/nllb-1p3b-60596557-20261115/run/evaluation/test_comparison/rl/translation_metrics/test_predictions.csv"
)


def build(csv_path: Path, output: Path) -> None:
    raw = csv_path.read_bytes()
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"example_index", "direction", "source", "reference", "prediction"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV must contain columns: {', '.join(sorted(required))}")
        rows = list(reader)
    if not rows:
        raise ValueError("CSV contains no predictions")
    ids = [row["example_index"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("example_index must be unique for reliable review storage")
    payload = {
        "rows": rows,
        "filename": csv_path.name,
        "path": csv_path.as_posix(),
        "fingerprint": hashlib.sha256(raw).hexdigest(),
    }
    # Prevent CSV text from closing the embedded script, including literal </script>.
    data = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    template = Path(__file__).with_name("prediction_viewer.html").read_text(encoding="utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template.replace("__PREDICTION_DATA__", data), encoding="utf-8")
    print(f"Built {output.resolve()} ({len(rows):,} predictions)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output", type=Path, help="Defaults to prediction_review.html beside the CSV")
    args = parser.parse_args()
    build(args.csv.resolve(), (args.output or args.csv.with_name("prediction_review.html")).resolve())
