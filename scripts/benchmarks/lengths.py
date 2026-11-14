"""Measure uncapped Qwen token lengths on deterministic samples of each local split."""

import argparse
import json
import random
from pathlib import Path
from typing import Any, cast

from scripts.main.common import load_config
from src.data.components.indexed_translation import IndexedTranslationRows


def audit(
    config: Path, output: Path, samples: int = 10000, local_only: bool = False
) -> dict[str, Any]:
    import numpy as np
    from transformers import AutoTokenizer

    if samples < 1:
        raise ValueError("samples must be positive")
    cfg = load_config(config, [])
    tokenizer = cast(Any, AutoTokenizer).from_pretrained(
        cfg["model_name"],
        revision=cfg["policy_revision"],
        cache_dir="logs/tokenizer_audit_cache",
        token=False,
        local_files_only=local_only,
    )
    report: dict[str, Any] = {
        "model": cfg["model_name"],
        "revision": cfg["policy_revision"],
        "seed": cfg["seed"],
        "source_cap": cfg["source_cap"],
        "target_cap": cfg["target_cap"],
        "sampling": "uniform without replacement per split; clean unaugmented texts",
        "splits": {},
    }
    template = "Translate {source_lang} to {target_lang}. Return only the translation.\nSource: {source}\nTranslation:"
    for stage, path in cfg["data_files"].items():
        rows = IndexedTranslationRows(path)
        indices = random.Random(cfg["seed"]).sample(
            range(len(rows)), min(samples, len(rows))
        )
        lengths: dict[str, list[int]] = {
            "en_vi/source": [],
            "en_vi/target": [],
            "vi_en/source": [],
            "vi_en/target": [],
        }
        for start in range(0, len(indices), 256):
            selected = [rows[i] for i in indices[start : start + 256]]
            for direction, source, target, sl, tl in (
                ("en_vi", "source", "reference", "English", "Vietnamese"),
                ("vi_en", "reference", "source", "Vietnamese", "English"),
            ):
                prompts = [
                    template.format(source=row[source], source_lang=sl, target_lang=tl)
                    for row in selected
                ]
                for kind, texts in (
                    ("source", prompts),
                    ("target", [" " + row[target] for row in selected]),
                ):
                    values = tokenizer(
                        texts, add_special_tokens=kind == "source", truncation=False
                    )["input_ids"]
                    lengths[direction + "/" + kind].extend(
                        len(value) + (kind == "target") for value in values
                    )
        statistics = {}
        for key, values in lengths.items():
            cap = cfg["source_cap"] if key.endswith("source") else cfg["target_cap"]
            statistics[key] = {
                "mean": float(np.mean(values)),
                "p95": float(np.percentile(values, 95)),
                "p99": float(np.percentile(values, 99)),
                "max_sampled": max(values),
                "exceeds_cap_fraction": sum(v > cap for v in values) / len(values),
            }
        report["splits"][stage] = {
            "total_pairs": len(rows),
            "sampled_pairs": len(indices),
            "statistics": statistics,
        }
        rows.close()
        print(json.dumps({"audited": stage, "sampled_pairs": len(indices)}), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/runs/full.yaml"))
    parser.add_argument(
        "--output", type=Path, default=Path("logs/full_data_length_audit.json")
    )
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--local-only", action="store_true")
    args = parser.parse_args()
    audit(args.config, args.output, args.samples, args.local_only)


if __name__ == "__main__":
    main()
