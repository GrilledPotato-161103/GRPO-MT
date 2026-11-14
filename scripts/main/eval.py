"""Evaluate a trusted SFT/GRPO checkpoint on clean bidirectional held-out data."""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from scripts.main import __version__
from scripts.main.common import DEFAULT_CONFIG, ROOT


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--version", action="version", version=__version__)
    result.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument(
        "--stage",
        choices=("sft", "rl", "cyclic"),
        default="rl",
        help="Class that created the checkpoint",
    )
    result.add_argument("--split", choices=("val", "test"), default="test")
    result.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save ordered UTF-8 CSV and per-batch JSONL predictions with references",
    )
    result.add_argument(
        "--reference-checkpoint",
        type=Path,
        help="Override the frozen reference path when moving a cyclic checkpoint",
    )
    result.add_argument(
        "--output", type=Path, help="Directory for this evaluation's metrics"
    )
    result.add_argument(
        "--trust-checkpoint",
        action="store_true",
        help="Allow loading your trusted checkpoint's Python metadata",
    )
    result.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    result.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the evaluation plan without opening the checkpoint",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    from scripts.main.common import (
        check_feedback,
        check_logging,
        ensure_comet,
        environment,
        load_config,
    )

    try:
        cfg = load_config(args.config, args.set)
        if args.reference_checkpoint and args.stage != "cyclic":
            raise ValueError("--reference-checkpoint requires --stage cyclic")
        checkpoint = args.checkpoint.resolve()
        output = (
            args.output.resolve()
            if args.output
            else Path(cfg["evaluation"]["output_dir"]) / args.split
        )
        data_file = Path(
            cfg["data_files"]["validation" if args.split == "val" else "test"]
        )
        plan = {
            "checkpoint": str(checkpoint),
            "stage": args.stage,
            "split": args.split,
            "data_file": str(data_file),
            "bidirectional": True,
            "output_dir": str(output),
            "metrics": ["BLEU", "chrF++"]
            + (["COMETKiwi"] if cfg["reward"]["comet_weight"] > 0 else [])
            + (
                [
                    "SAC",
                    "symmetric endpoint SAC"
                    if cfg.get("sac_reward", {}).get("consistency_mode")
                    == "symmetric_endpoints"
                    else "round-trip SAC",
                ]
                if cfg.get("sac_reward", {}).get("enabled")
                else []
            ),
            "batch_size": cfg["evaluation"]["batch_size"],
            "source_cap": cfg["source_cap"],
            "target_cap": cfg["target_cap"],
        }
        if args.dry_run:
            print(json.dumps(plan, indent=2))
            return 0
        if not args.trust_checkpoint:
            raise ValueError(
                "Evaluation requires --trust-checkpoint for your own trusted training checkpoint"
            )
        if not checkpoint.is_file() or not data_file.is_file():
            raise ValueError("Checkpoint or requested held-out split does not exist")
        if (output / "metrics.json").exists():
            raise ValueError(
                "Evaluation output already exists; select a new --output directory"
            )
        os.chdir(ROOT)
        environment(cfg)
        check_logging(cfg)
        check_feedback(cfg)
        ensure_comet(cfg)
        import lightning as L
        from lightning.pytorch.loggers import CSVLogger, WandbLogger

        from src.data.translation_datamodule import TranslationDataModule
        from src.models.anchored_grpo_module import AnchoredGRPOModule
        from src.models.components.translation_reward import TranslationReward
        from src.models.cyclic_module import CyclicTranslationModule
        from src.models.grpo_module import GRPOModule
        from src.models.sft_module import TranslationSFTModule

        output.mkdir(parents=True, exist_ok=True)
        reward = TranslationReward(
            **cfg["reward"],
            comet_model_name=cfg["comet_checkpoint"],
            comet_device=cfg["comet_device"],
            comet_batch_size=cfg["comet_batch_size"],
            sac=cfg.get("sac_reward"),
        )
        model_classes: dict[str, type[GRPOModule]] = {
            "sft": TranslationSFTModule,
            "rl": AnchoredGRPOModule
            if cfg.get("correction") is not None
            else GRPOModule,
            "cyclic": CyclicTranslationModule,
        }
        model_class = model_classes[args.stage]
        checkpoint_options: dict[str, Any] = (
            {"reference_checkpoint": str(args.reference_checkpoint.resolve())}
            if args.reference_checkpoint
            else {}
        )
        model = model_class.load_from_checkpoint(
            str(checkpoint),
            map_location="cpu",
            weights_only=False,
            reward=reward,
            load_dtype=cfg["load_dtype"],
            max_source_length=cfg["source_cap"],
            max_new_tokens=cfg["target_cap"],
            logprob_batch_size=cfg["logprob_batch_size"],
            logprob_token_budget=cfg["logprob_token_budget"],
            save_predictions=args.save_predictions,
            **checkpoint_options,
        )
        data = TranslationDataModule(
            train_file=str(data_file),
            val_file=str(data_file),
            test_file=str(data_file),
            bidirectional=True,
            paired_batches=cfg.get("paired_batches", False),
            batch_size=cfg["evaluation"]["batch_size"],
            augmentation={"enabled": False},
            **cfg.get("data_loader", {}),
        )
        logger = (
            WandbLogger(
                project=cfg["wandb"]["project"],
                entity=cfg["wandb"].get("entity"),
                offline=cfg["wandb_offline"],
                log_model=False,
                save_dir=str(output),
                group=cfg.get("run_group"),
                name=time.strftime("evaluation-%Y%m%d-%H%M%S"),
                job_type="evaluation",
            )
            if cfg["wandb"]["enabled"]
            else CSVLogger(save_dir=str(output), name="csv")
        )
        trainer_options = {
            **cfg["trainer"],
            "logger": logger,
            "default_root_dir": str(output),
            "enable_checkpointing": False,
            "enable_progress_bar": sys.stderr.isatty(),
            "enable_model_summary": False,
        }
        trainer = L.Trainer(**trainer_options)
        try:
            results = (
                trainer.validate(model, datamodule=data)
                if args.split == "val"
                else trainer.test(model, datamodule=data)
            )
            if not results:
                raise ValueError(
                    "No evaluation batches ran; check limit_val_batches/limit_test_batches"
                )
            report = {**plan, "metrics": results[0]}
            (output / "metrics.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8"
            )
            print(json.dumps(report, indent=2))
        finally:
            if cfg["wandb"]["enabled"]:
                import wandb

                wandb.finish()
        return 0
    except KeyboardInterrupt:
        return 130
    except (ValueError, TypeError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
