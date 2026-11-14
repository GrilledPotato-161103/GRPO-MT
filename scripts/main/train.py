"""Train SFT, GRPO, or the complete pipeline with a reusable YAML configuration."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from scripts.main import __version__
from scripts.main.common import DEFAULT_CONFIG, ROOT


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--version", action="version", version=__version__)
    result.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="Launcher YAML file"
    )
    result.add_argument(
        "--stage", choices=("pipeline", "sft", "rl"), default="pipeline"
    )
    result.add_argument(
        "--resume",
        type=Path,
        help="Resume one explicitly selected stage from its checkpoint",
    )
    result.add_argument(
        "--trust-checkpoint",
        action="store_true",
        help="Allow loading your trusted checkpoint's Python metadata",
    )
    result.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Repeatable YAML configuration override",
    )
    result.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved stage configs without loading models or writing files",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    from omegaconf import OmegaConf

    from scripts.main.common import (
        check_feedback,
        check_logging,
        compose_stage,
        deadline,
        ensure_comet,
        environment,
        load_config,
    )

    lock: Path | None = None
    try:
        cfg = load_config(args.config, args.set)
        if args.resume and (
            args.stage == "pipeline" or (not args.trust_checkpoint and not args.dry_run)
        ):
            raise ValueError("--resume requires --stage sft|rl and --trust-checkpoint")
        output = Path(cfg["output_dir"])
        cutoff = deadline(cfg, output)
        stages = ("sft", "rl") if args.stage == "pipeline" else (args.stage,)
        cfg.update(
            start_stage=stages[0],
            end_stage=stages[-1],
            graceful_deadline_unix=cutoff,
            fresh_start=not bool(args.resume),
            run_group=cfg.get("run_group") or time.strftime("phomt-main-%Y%m%d-%H%M%S"),
        )
        if args.resume:
            cfg["resume_checkpoints"] = {args.stage: args.resume.resolve().as_posix()}
        if args.dry_run:
            for stage in stages:
                print(
                    f"# {stage}: {cfg['evaluation_every_optimizer_steps']} optimizer updates per evaluation"
                )
                print(
                    OmegaConf.to_yaml(compose_stage(cfg, stage, cutoff), resolve=True)
                )
            return 0
        for name in ("sft", "rl", "validation", "test"):
            if not Path(cfg["data_files"][name]).is_file():
                raise ValueError(
                    f"Missing {name} data: {cfg['data_files'][name]}; prepare the 80/20 split first"
                )
        if not Path(cfg["dataset_manifest"]).is_file():
            raise ValueError(
                "Dataset manifest is missing; run scripts.data.prepare_stages first"
            )
        if args.resume and not args.resume.is_file():
            raise ValueError(f"Checkpoint not found: {args.resume}")
        if args.stage == "rl" and not Path(cfg["sft_export_dir"]).is_dir():
            raise ValueError(
                "RL requires the merged SFT export; run --stage sft or --stage pipeline first"
            )
        if not args.resume:
            for stage in stages:
                if (output / stage / "result.json").exists() or any(
                    (output / stage / "checkpoints").glob("*.ckpt")
                ):
                    raise ValueError(
                        f"Existing {stage} outputs; choose another output_dir or use --resume"
                    )
            if "sft" in stages and Path(cfg["sft_export_dir"]).exists():
                raise ValueError(
                    "SFT export already exists; choose another sft_export_dir"
                )
        # Compose before any model download, catching Hydra/input mistakes early.
        for stage in stages:
            compose_stage(cfg, stage, cutoff)
        output.mkdir(parents=True, exist_ok=True)
        candidate_lock = output / ".launcher.lock"
        try:
            descriptor = os.open(candidate_lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            raise ValueError(
                f"Launcher lock exists: {candidate_lock}; check the running process before removing a stale lock"
            ) from error
        lock = candidate_lock
        with os.fdopen(descriptor, "w") as stream:
            stream.write(str(os.getpid()))
        environment(cfg)
        check_logging(cfg)
        check_feedback(cfg)
        ensure_comet(cfg)
        (output / "rental_config.json").write_text(
            json.dumps(cfg, indent=2), encoding="utf-8"
        )
        (output / "launcher_config.yaml").write_text(
            OmegaConf.to_yaml(OmegaConf.create(cfg)), encoding="utf-8"
        )
        child = os.environ.copy()
        child["TRAIN_RUN_GROUP"] = cfg["run_group"]
        # Public foundation download and a local COMET checkpoint need no HF secret.
        for name in (
            "HUGGING_FACE_API_KEY",
            "HF_TOKEN",
            "HUGGINGFACEHUB_API_TOKEN",
            "CLAUDE_AUTH_KEY",
            "TRAIN_CACHE_READY_REVISION",
            "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE",
        ):
            if name == "CLAUDE_AUTH_KEY" and any(
                consumer.get("enabled") and not consumer.get("cache_only")
                for consumer in (
                    cfg.get("reward_harness", {}),
                    cfg.get("sac_reward", {}),
                )
            ):
                continue
            child.pop(name, None)
        result = subprocess.run(
            [
                sys.executable,
                "-u",
                "-m",
                "scripts.main.pipeline",
                "--output",
                str(output),
                *(["--deadline", str(cutoff)] if cutoff is not None else []),
            ],
            cwd=ROOT,
            env=child,
            check=False,
        )
        state_path = output / "state.json"
        if (
            result.returncode == 0
            and state_path.exists()
            and json.loads(state_path.read_text()).get("status") == "quota_stopped"
        ):
            print(
                "Training stopped at its deadline/STOP file; inspect checkpoints/interrupted.ckpt.",
                file=sys.stderr,
            )
            return 75
        return result.returncode
    except KeyboardInterrupt:
        print(
            "Interrupted. Inspect checkpoints/interrupted.ckpt before resuming.",
            file=sys.stderr,
        )
        return 130
    except (ValueError, TypeError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    finally:
        if lock is not None:
            lock.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
