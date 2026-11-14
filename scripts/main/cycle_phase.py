"""Execute one bounded cycle phase in a fresh process, releasing GPU memory on exit."""

import argparse
import json
from pathlib import Path
from typing import Any, cast


def run_phase(plan: dict[str, Any]) -> int:
    import hydra
    import lightning as L
    import torch
    from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
    from omegaconf import OmegaConf

    from scripts.main.common import compose_stage, environment
    from src.models.cyclic_module import CyclicTranslationModule
    from src.utils.rental_guard import RentalQuotaReached, TrainingProgress

    cfg, stage = plan["training"], plan["stage"]
    environment(cfg)
    torch.set_num_threads(8)
    L.seed_everything(plan["seed"], workers=True)
    phase = Path(plan["output"])
    phase.mkdir(parents=True, exist_ok=True)
    composed = compose_stage(cfg, stage, None)
    composed.data.train_file = plan["train_file"]
    composed.data.val_file = plan["validation_file"]
    composed.data.seed = plan["seed"]
    composed.model.model_name_or_path = cfg["model_name"]
    composed.model.base_revision = cfg["policy_revision"]
    composed.data.length_batching.tokenizer_name_or_path = cfg["model_name"]
    composed.data.length_batching.tokenizer_revision = cfg["policy_revision"]
    model_options = cast(
        dict[str, Any], OmegaConf.to_container(composed.model, resolve=True)
    )
    model_options.pop("_target_")
    model_options.pop("_convert_", None)
    model_options["reward"] = hydra.utils.instantiate(composed.model.reward)
    model_options.update(
        reference_checkpoint=plan["reference_checkpoint"],
        cycle_stage=stage,
        stage_update_offset=plan["stage_update_offset"],
        total_update_offset=plan["total_update_offset"],
        beta=cfg["beta"],
        kl_control=cfg["kl_control"],
    )
    trained = phase / "checkpoints/trained.ckpt"
    interrupted = phase / "checkpoints/interrupted.ckpt"
    source = (
        trained
        if trained.exists()
        else interrupted
        if interrupted.exists()
        else Path(plan["policy_checkpoint"])
    )
    model = CyclicTranslationModule.load_from_checkpoint(
        source, weights_only=False, map_location="cpu", **model_options
    )
    data = hydra.utils.instantiate(composed.data)
    for item in composed.logger.values():
        if item.get("_target_", "").endswith("WandbLogger"):
            item.id = plan["run_id"]
            item.group = cfg["run_group"]
            item.name = plan["run_id"]
            item.save_dir = str(phase)
    loggers = [
        hydra.utils.instantiate(item) for item in composed.logger.values() if item
    ]

    class RestoreOptimizer(L.Callback):
        def on_fit_start(self, trainer: Any, pl_module: Any) -> None:
            path = plan.get("optimizer_checkpoint")
            if path and not interrupted.exists():
                saved = torch.load(path, map_location="cpu", weights_only=False)
                if len(saved.get("optimizer_states", [])) != 1:
                    raise ValueError(
                        "Expected one saved optimizer for the selected objective"
                    )
                trainer.optimizers[0].load_state_dict(saved["optimizer_states"][0])
            for group in trainer.optimizers[0].param_groups:
                group["initial_lr"] = cfg["learning_rate"][stage]
                group["lr"] = cfg["learning_rate"][stage]

    callbacks = [
        RichProgressBar(
            refresh_rate=1, leave=True, console_kwargs={"force_terminal": True}
        ),
        TrainingProgress(stop_file=plan["stop_file"]),
        RestoreOptimizer(),
        ModelCheckpoint(
            dirpath=phase / "checkpoints/periodic",
            filename="step_{step}",
            every_n_train_steps=100,
            save_top_k=1,
            save_last=True,
            save_on_train_epoch_end=False,
        ),
    ]
    if cfg["trainer"].get("enable_progress_bar") is False:
        callbacks = [
            callback
            for callback in callbacks
            if not isinstance(callback, RichProgressBar)
        ]
    trainer = L.Trainer(
        **{
            **cfg["trainer"],
            "max_epochs": 1,
            "min_epochs": 0,
            "max_steps": plan["updates"],
            "limit_train_batches": plan["batches"],
            "limit_val_batches": 1.0,
            "num_sanity_val_steps": 0,
            "check_val_every_n_epoch": 2,
            "val_check_interval": 1.0,
            "log_every_n_steps": cfg["log_every_n_steps"],
            "default_root_dir": str(phase),
            "enable_model_summary": False,
        },
        callbacks=callbacks,
        logger=loggers,
    )
    try:
        if not trained.exists():
            trainer.fit(
                model,
                datamodule=data,
                ckpt_path=str(interrupted) if interrupted.exists() else None,
                weights_only=False,
            )
            if trainer.global_step != plan["updates"]:
                raise RuntimeError(
                    "Phase ended without its requested optimizer updates"
                )
            trainer.save_checkpoint(trained)
        saved = torch.load(trained, map_location="cpu", weights_only=False)
        if saved["global_step"] != plan["updates"]:
            raise ValueError("Completed phase checkpoint has an incorrect update count")
        if plan["evaluate"]:
            data.hparams["batch_size"] = cfg["evaluation"]["batch_size"]
            trainer.validate(model, datamodule=data)
        metrics_path = phase / "translation_metrics/val.json"
        metrics = json.loads(metrics_path.read_text()) if plan["evaluate"] else None
        if metrics and metrics["counts"] != {
            "en_vi": plan["validation_pairs"],
            "vi_en": plan["validation_pairs"],
        }:
            raise ValueError(
                "Evaluation did not cover the fixed balanced validation subset"
            )
        result = {
            "stage": stage,
            "cycle": plan["cycle"],
            "checkpoint": str(trained),
            "updates": saved["global_step"],
            **saved["cycle_state"],
            "evaluation": metrics,
        }
        (phase / "result.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        return 0
    except RentalQuotaReached:
        return 75
    finally:
        if cfg["wandb"]["enabled"]:
            import wandb

            wandb.finish()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    return run_phase(json.loads(args.plan.read_text(encoding="utf-8")))


if __name__ == "__main__":
    raise SystemExit(main())
