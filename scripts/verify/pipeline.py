"""Run a bounded SFT/GRPO functionality test and verify actual optimizer updates."""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from scripts.main.train import main as train


def synthetic_config(config: Path, output: Path, steps: int) -> Path:
    """Create a bounded probe with isolated synthetic data and fresh outputs."""
    from omegaconf import OmegaConf

    from scripts.data.prepare_stages import prepare
    from scripts.main.common import load_config

    if steps < 1:
        raise ValueError("steps must be positive")
    cfg = load_config(config, [])
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    data = output / "data"
    data.mkdir()
    for name, indices in (
        ("train", range(1, 201)),
        ("validation", range(201, 205)),
        ("test", range(205, 209)),
    ):
        with (data / f"{name}.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["en", "vi"])
            for index in indices:
                writer.writerow(
                    [
                        f"The library has {index} books. The books are on the second floor.",
                        f"Thư viện có {index} cuốn sách. Những cuốn sách ở tầng hai.",
                    ]
                )
    prepare(
        data / "train.csv", data / "validation.csv", data / "test.csv", data / "stages"
    )
    cfg.pop("max_optimizer_steps", None)
    cfg.update(
        output_dir=str(output / "run"),
        sft_export_dir=str(output / "run/sft_export"),
        dataset_manifest=str(data / "stages/manifest.json"),
        data_files={
            "sft": str(data / "stages/sft.csv"),
            "rl": str(data / "stages/rl.csv"),
            "validation": str(data / "validation.csv"),
            "test": str(data / "test.csv"),
        },
        budget_control=False,
        max_epochs={"sft": -1, "rl": -1},
        test_after_training=False,
        run_group=output.name,
        evaluation_every_optimizer_steps=max(steps * cfg["num_iterations"], 2),
        length_cache_dir=str(output / "length_cache"),
    )
    cfg["trainer"]["max_steps"] = steps
    if cfg.get("sac_reward", {}).get("fusion", {}).get("enabled"):
        cfg["sac_reward"]["fusion"]["total_steps"] = steps
        cfg["sac_reward"]["cache_dir"] = str(output / "sac_cache")
    destination = output / "probe.yaml"
    OmegaConf.save(OmegaConf.create(cfg), destination)
    return destination


def verify(config: Path, steps: int = 5) -> dict[str, Any]:
    from scripts.main.common import load_config

    if steps < 1:
        raise ValueError("steps must be positive")
    cfg = load_config(
        config, [f"trainer.max_steps={steps}", "test_after_training=false"]
    )
    if cfg.get("max_optimizer_steps"):
        raise ValueError(
            "Use --synthetic or remove max_optimizer_steps from a separate probe configuration"
        )
    code = train(
        [
            "--config",
            str(config),
            "--set",
            f"trainer.max_steps={steps}",
            "--set",
            "test_after_training=false",
        ]
    )
    if code:
        raise RuntimeError(f"Training test exited {code}")
    root = Path(cfg["output_dir"])
    report: dict[str, Any] = {
        "status": "passed",
        "steps_per_stage": steps,
        "scope": "Functionality only; small samples from the full prepared splits, not translation convergence",
        "stages": {},
    }
    for stage, phase in (("sft", "sft/optimizer"), ("rl", "translation/optimizer")):
        result = json.loads((root / stage / "result.json").read_text(encoding="utf-8"))
        events = [
            json.loads(line)
            for line in (root / stage / "phase_timing/events.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        updates = [
            event
            for event in events
            if event["phase"] == phase and event.get("succeeded")
        ]
        audit = json.loads(
            (root / stage / "diagnostics/trainability.json").read_text(encoding="utf-8")
        )
        if (
            result["global_step"] != steps
            or len(updates) != steps
            or not audit["backbone_frozen"]
        ):
            raise RuntimeError(f"{stage} failed optimizer-count/frozen-backbone checks")
        if any(not math.isfinite(float(value)) for value in result["metrics"].values()):
            raise RuntimeError(f"Non-finite {stage} metrics")
        metrics = json.loads(
            (root / stage / "translation_metrics/val.json").read_text(encoding="utf-8")
        )
        if set(metrics["directions"]) != {"en_vi", "vi_en"}:
            raise RuntimeError("Both evaluation directions are required")
        for direction in metrics["directions"].values():
            if not {"bleu", "chrf", "comet"}.issubset(direction):
                raise RuntimeError("Missing BLEU, chrF++ or real COMETKiwi metrics")
        report["stages"][stage] = {
            "global_step": result["global_step"],
            "optimizer_calls": len(updates),
            "backbone_frozen": audit["backbone_frozen"],
            "checkpoint": result["checkpoint"],
            "validation": metrics,
            "timing_summary": json.loads(
                (root / stage / "phase_timing/summary.json").read_text(encoding="utf-8")
            ),
        }
    updates = [
        json.loads(line)
        for line in (root / "rl/diagnostics/updates.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    if [row["global_step"] for row in updates] != list(range(1, steps + 1)):
        raise RuntimeError("Missing per-update GRPO diagnostics")
    report["rl_diagnostics"] = updates
    (root / "five_step_verification.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "sft_updates": steps,
                "rl_updates": steps,
                "report": str(root / "five_step_verification.json"),
            }
        ),
        flush=True,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    from scripts.main.common import DEFAULT_CONFIG

    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate isolated synthetic probe data",
    )
    parser.add_argument(
        "--output", type=Path, help="Fresh directory required with --synthetic"
    )
    args = parser.parse_args()
    if args.synthetic and args.output is None:
        parser.error("--synthetic requires --output pointing to a fresh directory")
    if args.output and not args.synthetic:
        parser.error("--output requires --synthetic")
    config = (
        synthetic_config(args.config, args.output, args.steps)
        if args.synthetic
        else args.config
    )
    verify(config, args.steps)


if __name__ == "__main__":
    main()
