"""Flatten all benchmark policy/feedback phases into a reusable speed table."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def summarize(root: Path, output: Path) -> list[dict[str, Any]]:
    """Combine per-mode, per-stage synchronized latency and throughput measurements."""
    rows = []
    for path in sorted(root.glob("*/report.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["status"] != "passed":
            raise ValueError(f"Incomplete benchmark: {path.parent.name}")
        for stage, result in report["stages"].items():
            for phase, metrics in result["phases"]["phases"].items():
                if phase.startswith("feedback/"):
                    component = "synthetic_mock_feedback"
                elif phase.startswith("reward/cometkiwi"):
                    component = "cometkiwi"
                elif phase.startswith("reward/"):
                    component = phase.split("/", 1)[1]
                elif phase.startswith("data/"):
                    component = "online_augmentation"
                else:
                    component = "policy"
                rows.append(
                    {
                        "policy_model": report["model"],
                        "mode": report["mode"],
                        "stage": stage,
                        "component": component,
                        "phase": phase,
                        **{
                            key: metrics.get(key, "")
                            for key in (
                                "calls",
                                "total_seconds",
                                "mean_seconds",
                                "median_seconds",
                                "p95_seconds",
                                "min_seconds",
                                "max_seconds",
                                "input_tokens",
                                "input_tokens_per_second",
                                "output_tokens",
                                "output_tokens_per_second",
                                "samples",
                                "samples_per_second",
                            )
                        },
                    }
                )
    if not rows:
        raise ValueError("No completed benchmark reports")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    """Write one row per model/mode/stage/phase without collapsing unlike workloads."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(f"Recorded {len(summarize(args.root, args.output))} phase rows")


if __name__ == "__main__":
    main()
