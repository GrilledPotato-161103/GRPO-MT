"""Run sampled SFT/GRPO cycles with paired validation and durable phase handoffs."""

import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast

from scripts.main.common import ROOT, absolute, load_config


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_settings(
    path: Path, overrides: list[str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    from omegaconf import OmegaConf

    raw = OmegaConf.merge(OmegaConf.load(path), OmegaConf.from_dotlist(overrides))
    value = OmegaConf.to_container(raw, resolve=True)
    if not isinstance(value, dict):
        raise TypeError("Cycle configuration must be a mapping")
    settings = cast(dict[str, Any], value)
    for key in ("cycles", "sft_batches", "rl_updates", "validation_pairs", "seed"):
        value = settings[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < (0 if key == "seed" else 1)
        ):
            raise ValueError(f"{key} must be a positive integer (seed may be zero)")
    if not isinstance(settings["evaluate_after_sft"], bool):
        raise TypeError("evaluate_after_sft must be a boolean")
    base = Path(settings["base_config"])
    if not base.is_absolute():
        base = path.resolve().parent / base
    # Reuse all credential validation, input paths, caps, rewards and LoRA settings.
    cfg = load_config(base, [], settings.get("training", {}))
    for key in ("output_dir", "initial_checkpoint"):
        settings[key] = absolute(settings[key]).as_posix()
    for key in ("hf_home", "comet_checkpoint", "dataset_manifest", "sft_export_dir"):
        cfg[key] = absolute(cfg[key]).as_posix()
    cfg["data_files"] = {
        key: absolute(value).as_posix() for key, value in cfg["data_files"].items()
    }
    cfg["output_dir"] = settings["output_dir"]
    if not cfg.get("lora", {}).get("enabled"):
        raise ValueError("Sampled cyclic training requires LoRA")
    if cfg.get("budget_control", True):
        raise ValueError(
            "Use budget_control=false; this runner never controls rental billing"
        )
    if settings["rl_updates"] % cfg["num_iterations"]:
        raise ValueError("rl_updates must be divisible by num_iterations")
    for stage, batches in (
        ("sft", settings["sft_batches"]),
        ("rl", settings["rl_updates"] // cfg["num_iterations"]),
    ):
        if batches * cfg["batch_sizes"][stage] % 2:
            raise ValueError(
                "Each phase must contain an even number of directional examples"
            )
    return settings, cfg


def sample_pairs(
    source: Path, target: Path, count: int, seed: int, offset: int | None = None
) -> dict[str, Any]:
    """Uniform sampling without replacement within a phase, before bidirectional views."""
    from src.data.components.indexed_translation import IndexedTranslationRows

    rows = IndexedTranslationRows(source)
    try:
        if count > len(rows):
            raise ValueError(f"Requested {count} pairs from a {len(rows)}-pair split")
        if offset is None:
            indices = random.Random(seed).sample(range(len(rows)), count)
        else:
            import numpy as np

            indices = []
            position = offset
            while len(indices) < count:
                sweep, start = divmod(position, len(rows))
                take = min(count - len(indices), len(rows) - start)
                permutation = np.random.default_rng(seed + sweep).permutation(len(rows))
                indices.extend(permutation[start : start + take].tolist())
                position += take
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.with_suffix(".sampling.json").exists():
            raise ValueError("Refusing to overwrite a sampled dataset")
        temporary = target.with_suffix(".csv.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["en", "vi"])
            for index in indices:
                row = rows[index]
                writer.writerow([row["source"], row["reference"]])
        temporary.replace(target)
        record = {
            "source": str(source),
            "source_pairs": len(rows),
            "pairs": count,
            "seed": seed,
            "stream_offset": offset,
            "indices": indices,
            "sha256": sha256(target),
        }
        atomic_json(target.with_suffix(".sampling.json"), record)
        return record
    finally:
        rows.close()


def phase_plan(
    settings: dict[str, Any],
    cfg: dict[str, Any],
    cycle: int,
    stage: str,
    policy: str,
    optimizer: str | None,
    stage_offset: int,
    total_offset: int,
) -> dict[str, Any]:
    root = Path(settings["output_dir"])
    batches = (
        settings["sft_batches"]
        if stage == "sft"
        else settings["rl_updates"] // cfg["num_iterations"]
    )
    phase = root / f"cycle_{cycle + 1:03d}" / stage
    return {
        "cycle": cycle + 1,
        "stage": stage,
        "output": str(phase),
        "training": cfg,
        "reference_checkpoint": settings["initial_checkpoint"],
        "policy_checkpoint": policy,
        "optimizer_checkpoint": optimizer,
        "stage_update_offset": stage_offset,
        "total_update_offset": total_offset,
        "updates": batches if stage == "sft" else settings["rl_updates"],
        "batches": batches,
        "train_pairs": batches * cfg["batch_sizes"][stage] // 2,
        "train_file": str(phase / "sample.csv"),
        "validation_file": str(root / "validation.csv"),
        "validation_pairs": settings["validation_pairs"],
        "evaluate": stage == "rl" or settings["evaluate_after_sft"],
        "seed": settings["seed"] + 1009 * (cycle + 1) + (1 if stage == "rl" else 0),
        "stop_file": str(root / "STOP"),
        "run_id": f"{cfg['run_group']}-c{cycle + 1:03d}-{stage}",
    }


def log_evaluation(run: Any, record: dict[str, Any]) -> None:
    if run is None or not record.get("evaluation"):
        return
    phase_index = (record["cycle"] - 1) * 2 + (1 if record["stage"] == "sft" else 2)
    scores = {
        key.removeprefix("val/"): value
        for key, value in record["evaluation"]["metrics"].items()
    }
    run.log(
        {
            "cycle": record["cycle"],
            "phase_index": phase_index,
            "updates/total": record["total_updates"],
            **{f"evaluation/{key}": value for key, value in scores.items()},
            **{f"{record['stage']}_eval/{key}": value for key, value in scores.items()},
        }
    )


def execute(settings: dict[str, Any], cfg: dict[str, Any], resume: bool) -> int:
    import torch
    from huggingface_hub import snapshot_download

    from scripts.main.common import (
        check_feedback,
        check_logging,
        ensure_comet,
        environment,
    )

    root = Path(settings["output_dir"])
    state_path = root / "cycle_state.json"
    fingerprint = hashlib.sha256(
        json.dumps({"settings": settings, "training": cfg}, sort_keys=True).encode()
    ).hexdigest()
    if state_path.exists() and not resume:
        raise ValueError(
            "Cycle output exists; use --resume or another output directory"
        )
    state: dict[str, Any] = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else {"fingerprint": fingerprint, "completed": []}
    )
    if state["fingerprint"] != fingerprint:
        raise ValueError("Resume configuration changed; use the original configuration")
    if (root / "STOP").exists():
        print(
            f"STOP remains present: {root / 'STOP'}; remove it explicitly before resuming"
        )
        return 75
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".cycle.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(str(os.getpid()))
    run = None
    try:
        initial = Path(settings["initial_checkpoint"])
        if not initial.is_file():
            raise ValueError(f"Initial SFT checkpoint is missing: {initial}")
        saved = torch.load(initial, map_location="cpu", weights_only=False)
        if not saved.get("adapter_checkpoint"):
            raise ValueError("Initial checkpoint must be a trusted LoRA SFT checkpoint")
        initial_step = int(saved["global_step"])
        del saved
        environment(cfg)
        check_logging(cfg)
        check_feedback(cfg)
        ensure_comet(cfg)
        provenance_path = root / "provenance.json"
        if not provenance_path.exists():
            manifest = json.loads(Path(cfg["dataset_manifest"]).read_text())
            hashes = {}
            for name, path in cfg["data_files"].items():
                expected = manifest.get("outputs", {}).get(name) or manifest.get(
                    "inputs", {}
                ).get(name)
                if not expected or sha256(Path(path)) != expected["sha256"]:
                    raise ValueError(f"Full dataset hash mismatch: {name}")
                hashes[name] = expected["sha256"]
            atomic_json(
                provenance_path,
                {"datasets": hashes, "reference_sha256": sha256(initial)},
            )
        elif json.loads(provenance_path.read_text())["reference_sha256"] != sha256(
            initial
        ):
            raise ValueError("Starting SFT checkpoint changed")
        if not Path(cfg["model_name"]).is_dir():
            snapshot_download(
                cfg["model_name"],
                revision=cfg["policy_revision"],
                allow_patterns=[
                    "*.json",
                    "*.safetensors",
                    "*.txt",
                    "*.model",
                    "*.jinja",
                ],
            )
        if not (root / "validation.csv").exists():
            sample_pairs(
                Path(cfg["data_files"]["validation"]),
                root / "validation.csv",
                settings["validation_pairs"],
                settings["seed"],
            )
        if (
            sha256(root / "validation.csv")
            != json.loads((root / "validation.sampling.json").read_text())["sha256"]
        ):
            raise ValueError("Fixed validation subset changed")
        if cfg["wandb"]["enabled"]:
            import wandb

            run = wandb.init(
                project=cfg["wandb"]["project"],
                entity=cfg["wandb"].get("entity"),
                id=f"{cfg['run_group']}-overview",
                group=cfg["run_group"],
                name="SFT-GRPO cycles",
                resume="allow",
                dir=str(root),
                mode=cfg["wandb"]["mode"],
                config=settings,
            )
            run.define_metric("phase_index")
            run.define_metric("evaluation/*", step_metric="phase_index")
            run.define_metric("cycle")
            run.define_metric("sft_eval/*", step_metric="cycle")
            run.define_metric("rl_eval/*", step_metric="cycle")
        for record in state["completed"]:
            log_evaluation(run, record)
        policy = settings["initial_checkpoint"]
        optimizers: dict[str, str | None] = {"sft": policy, "rl": None}
        offsets = {"sft": initial_step, "rl": 0}
        total = initial_step
        for cycle in range(settings["cycles"]):
            for stage in ("sft", "rl"):
                previous = next(
                    (
                        r
                        for r in state["completed"]
                        if r["cycle"] == cycle + 1 and r["stage"] == stage
                    ),
                    None,
                )
                if previous:
                    policy = previous["checkpoint"]
                    optimizers[stage] = policy
                    offsets[stage], total = (
                        previous["stage_updates"],
                        previous["total_updates"],
                    )
                    continue
                if (root / "STOP").exists():
                    state["status"] = "stopped"
                    atomic_json(state_path, state)
                    return 75
                plan = phase_plan(
                    settings,
                    cfg,
                    cycle,
                    stage,
                    policy,
                    optimizers[stage],
                    offsets[stage],
                    total,
                )
                phase = Path(plan["output"])
                phase.mkdir(parents=True, exist_ok=True)
                plan_path = phase / "plan.json"
                if not plan_path.exists():
                    sample_pairs(
                        Path(cfg["data_files"][stage]),
                        Path(plan["train_file"]),
                        plan["train_pairs"],
                        settings["seed"] + (1 if stage == "rl" else 0),
                        offset=cycle * plan["train_pairs"],
                    )
                    atomic_json(plan_path, plan)
                elif json.loads(plan_path.read_text()) != plan:
                    raise ValueError(
                        "Saved phase plan differs from the expected continuation"
                    )
                sample_record = json.loads(
                    Path(plan["train_file"]).with_suffix(".sampling.json").read_text()
                )
                if sha256(Path(plan["train_file"])) != sample_record["sha256"]:
                    raise ValueError("Saved phase sample changed")
                state.update(
                    status="training",
                    cycle=cycle + 1,
                    stage=stage,
                    unix_time=time.time(),
                )
                atomic_json(state_path, state)
                print(
                    f"Cycle {cycle + 1}/{settings['cycles']}: {stage}, {plan['batches']} batches, {plan['updates']} updates",
                    flush=True,
                )
                result_path = phase / "result.json"
                if not result_path.exists():
                    child = os.environ.copy()
                    child.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
                    for name in (
                        "HUGGING_FACE_API_KEY",
                        "HF_TOKEN",
                        "HUGGINGFACEHUB_API_TOKEN",
                    ):
                        child.pop(name, None)
                    result = subprocess.run(
                        [
                            sys.executable,
                            "-u",
                            "-m",
                            "scripts.main.cycle_phase",
                            str(plan_path),
                        ],
                        cwd=ROOT,
                        env=child,
                        check=False,
                    )
                    if result.returncode:
                        state["status"] = (
                            "stopped" if result.returncode == 75 else "failed"
                        )
                        atomic_json(state_path, state)
                        return result.returncode
                record = json.loads(result_path.read_text())
                state["completed"].append(record)
                atomic_json(state_path, state)
                policy = record["checkpoint"]
                optimizers[stage] = policy
                offsets[stage], total = record["stage_updates"], record["total_updates"]
                log_evaluation(run, record)
                atomic_json(root / "latest.json", record)
        state.update(status="completed", unix_time=time.time())
        atomic_json(state_path, state)
        return 0
    finally:
        lock.unlink(missing_ok=True)
        if run is not None:
            run.finish()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/runs/cycles.yaml"
    )
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--trust-checkpoint", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        settings, cfg = load_settings(args.config, args.set)
        cfg["run_group"] = f"{Path(settings['output_dir']).name}"
        if args.dry_run:
            plans = [
                phase_plan(
                    settings, cfg, 0, stage, settings["initial_checkpoint"], None, 0, 0
                )
                for stage in ("sft", "rl")
            ]
            print(json.dumps({"settings": settings, "first_cycle": plans}, indent=2))
            return 0
        if not args.trust_checkpoint:
            raise ValueError(
                "Use --trust-checkpoint for your trusted SFT/resume checkpoint"
            )
        return execute(settings, cfg, args.resume)
    except (ValueError, TypeError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
