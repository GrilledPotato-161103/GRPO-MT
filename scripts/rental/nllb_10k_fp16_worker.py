"""Detached driver for the specifically requested 10k-pair FP16 NLLB run.

Run on the rental after extracting the verified source/data payload. Credentials
arrive through SSH stdin/environment, never this config or a command argument.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "logs/nllb-1p3b-10k-fp16-20261116"
CONFIG = ROOT / "configs/rental/nllb_10k_fp16_20261116.json"


def state(phase, **values):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    record = {"phase": phase, "unix_time": time.time(), **values}
    temporary = OUTPUT / "worker_status.tmp"
    temporary.write_text(json.dumps(record, indent=2))
    temporary.replace(OUTPUT / "worker_status.json")
    print(json.dumps(record), flush=True)


def child(*arguments):
    subprocess.run(
        [sys.executable, "-u", "-m", "scripts.rental.nllb_10k_fp16_worker", *arguments],
        cwd=ROOT,
        check=True,
    )


def bootstrap(config):
    if not os.environ.get("OPENROUTER_API_KEY") or not os.environ.get("HF_TOKEN"):
        raise RuntimeError(
            "Both explicitly designated HF and OpenRouter credentials are required"
        )
    if (OUTPUT / "sft/result.json").exists() or (OUTPUT / "sft_export").exists():
        raise FileExistsError("Fresh training refuses existing trained outputs")
    state("installing_dependencies")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "-e",
            ".[lora,comet,logging,dev]",
            "transformers==4.57.6",
            "peft==0.18.1",
            "lightning==2.6.6",
            "wandb==0.24.0",
            "setuptools<81",
        ],
        cwd=ROOT,
        check=True,
    )
    state("verifying_payload")
    manifest = json.loads((ROOT / "data/run/manifest.json").read_text())
    test_name = "test_sample" if config.get("test_pairs") == 1000 else "test"
    for name in ("sft", "rl", "validation_eval", test_name):
        path = (
            ROOT
            / "data/run"
            / (
                config.get("test_file", "test_bounded.csv")
                if name == test_name
                else name + ".csv"
            )
        )
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != manifest["outputs"][name]["sha256"]:
            raise RuntimeError("Dataset payload checksum mismatch: " + name)
    if manifest["counts"]["sft"] != config["sft_pairs"]:
        raise RuntimeError("SFT population differs from the requested 10,000 pairs")
    state("testing_objective_and_fp16")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/grpo/test_comet_selected_anchor.py",
            "tests/grpo/test_correction_objective.py",
            "tests/grpo/test_correction_evidence.py",
            "-q",
            "-k",
            "not profile and not resource_estimates",
        ],
        cwd=ROOT,
        check=True,
    )
    state("downloading_pinned_huggingface_assets")
    from huggingface_hub import hf_hub_download, snapshot_download

    policy = snapshot_download(
        config["model"],
        revision=config["revision"],
        token=os.environ["HF_TOKEN"],
        allow_patterns=[
            "*.json",
            "pytorch_model*.bin",
            "*.safetensors",
            "*.model",
            "*.txt",
        ],
    )
    comet = config["comet"]
    checkpoint = hf_hub_download(
        comet["model"],
        "checkpoints/model.ckpt",
        revision=comet["revision"],
        token=os.environ["HF_TOKEN"],
    )
    hf_hub_download(
        comet["model"],
        "hparams.yaml",
        revision=comet["revision"],
        token=os.environ["HF_TOKEN"],
    )
    assets = {
        "policy_snapshot": policy,
        "policy_revision": config["revision"],
        "comet_checkpoint": checkpoint,
    }
    (OUTPUT / "assets.json").write_text(json.dumps(assets, indent=2))
    (OUTPUT / "run_config.json").write_text(json.dumps(config, indent=2))
    pipeline(config)


def pipeline(config, resume_checkpoint=None, resume_phase=None):
    (OUTPUT / "run_config.json").write_text(json.dumps(config, indent=2))
    for stage in ("sft", "rl"):
        if not (OUTPUT / stage / "result.json").exists():
            state(
                stage,
                requested_pairs=config["sft_pairs"] if stage == "sft" else None,
                target_optimizer_steps=1250
                if stage == "sft"
                else config["rl_optimizer_steps"],
            )
            arguments = ["--stage", stage]
            if resume_checkpoint and resume_phase == stage:
                arguments += ["--resume", str(resume_checkpoint)]
            child(*arguments)
        if not (OUTPUT / (stage + "_export")).exists():
            state("exporting_best_comet_" + stage)
            child("--stage", "export_" + stage)
    inventory = []
    for folder in (OUTPUT / "sft_export", OUTPUT / "rl_export"):
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                with path.open("rb") as stream:
                    checksum = hashlib.file_digest(stream, "sha256").hexdigest()
                inventory.append(
                    {
                        "path": str(path.relative_to(ROOT)),
                        "bytes": path.stat().st_size,
                        "sha256": checksum,
                    }
                )
    (OUTPUT / "export_manifest.json").write_text(json.dumps(inventory, indent=2))
    state(
        "complete",
        rl_optimizer_steps=config["rl_optimizer_steps"],
        exports="sft_export and rl_export",
    )


def reward(config, stage):
    from src.models.components.translation_reward import TranslationReward

    assets = json.loads((OUTPUT / "assets.json").read_text())
    sac = {"enabled": False}
    if stage == "rl":
        judge = config["sac"]
        sac = {
            "enabled": True,
            "consistency_weight": 0,
            "fusion": {"enabled": False},
            "cache_dir": str(OUTPUT / "sac_cache"),
            "backend": {
                "kind": "api",
                "model": judge["model"],
                "base_url": judge["base_url"],
                "api_key_env": judge["key_variable"],
                "max_tokens": judge["max_tokens"],
                "concurrency": judge["concurrency"],
                "temperature": None,
                "json_mode": False,
            },
        }
    return TranslationReward(
        chrf_weight=0,
        bleu_weight=0,
        comet_weight=1,
        comet_model_name=assets["comet_checkpoint"],
        comet_device=config["comet"]["device"],
        comet_batch_size=config["comet"]["batch_size"],
        sac=sac,
    )


def train(config, stage, resume_checkpoint=None):
    import lightning as L
    import torch
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger, WandbLogger
    from lightning.pytorch.plugins.precision import MixedPrecision

    from src.data.translation_datamodule import TranslationDataModule
    from src.models.anchored_grpo_module import AnchoredGRPOModule
    from src.models.sft_module import TranslationSFTModule
    from src.utils.rental_guard import TrainingProgress

    L.seed_everything(config["seed"], workers=True)
    torch.set_num_threads(8)
    if not torch.cuda.is_available() or "4090" not in torch.cuda.get_device_name():
        raise RuntimeError("This run requires the requested RTX 4090 CUDA device")
    assets = json.loads((OUTPUT / "assets.json").read_text())
    folder = OUTPUT / stage
    model_name = (
        assets["policy_snapshot"] if stage == "sft" else str(OUTPUT / "sft_export")
    )
    completed_batches, resume_epoch = 0, 0
    if resume_checkpoint:
        saved = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
        resume_epoch = int(saved.get("epoch", 0))
        completed_batches = int(
            saved["loops"]["fit_loop"]["epoch_loop.batch_progress"]["current"][
                "completed"
            ]
        )
        nominal_batches = (
            2 * config["sft_pairs"] + config["batch_directional_records"] - 1
        ) // config["batch_directional_records"]
        if stage == "sft" and completed_batches >= nominal_batches:
            completed_batches = 0  # max_steps already reached; no task update will run.
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "resume.json").write_text(
            json.dumps(
                {
                    "checkpoint": str(resume_checkpoint),
                    "saved_global_step": saved["global_step"],
                    "completed_batches": completed_batches,
                    "epoch": resume_epoch,
                    "loop_counters_rebased": False,
                },
                indent=2,
            )
        )
        del saved
    data = TranslationDataModule(
        train_file=str(ROOT / "data/run" / (stage + ".csv")),
        val_file=str(ROOT / "data/run/validation_eval.csv"),
        test_file=str(ROOT / "data/run" / config.get("test_file", "test_bounded.csv")),
        batch_size=config["batch_directional_records"],
        evaluation_batch_size=config["evaluation_batch_size"],
        bidirectional=True,
        paired_batches=True,
        num_workers=4,
        persistent_workers=True,
        disk_backed=True,
        seed=config["seed"],
        augmentation={"enabled": False},
        resume_completed_batches=completed_batches,
        resume_epoch=resume_epoch,
    )
    values = {
        "model_name_or_path": model_name,
        "revision": config["revision"],
        "language_codes": {"English": "eng_Latn", "Vietnamese": "vie_Latn"},
        "prompt_template": "{source}",
        "source_content_limit": 256,
        "strict_prompt_length": True,
        "max_source_length": 258,
        "max_new_tokens": 257,
        "load_dtype": "float16",
        "lora": config["lora"],
        "gradient_checkpointing": True,
        "group_size": config["candidates_per_source"],
        "num_iterations": config["updates_per_rollout"],
        "learning_rate": config[stage + "_learning_rate"],
        "weight_decay": 0,
        "warmup_steps": config["warmup_updates"],
        "temperature": 1,
        "beta": 0,
        "logprob_batch_size": config["logprob_batch_size"],
        "logprob_token_budget": config["logprob_token_budget"],
        "reward": reward(config, stage),
        "profiling": {"enabled": True, "synchronize_cuda": False},
        "save_predictions": True,
        "diagnostics": {"enabled": True, "every_n_updates": 1, "record_rollouts": True},
    }
    if stage == "sft":
        model = TranslationSFTModule(
            **values,
            max_target_length=258,
            microbatch_size=config["sft_microbatch_size"],
        )
    else:
        model = AnchoredGRPOModule(
            **values,
            correction=config["correction"],
            lr_scheduler={
                "kind": "reduce_on_plateau",
                "monitor": "val/comet_macro",
                "mode": "max",
                "factor": 0.5,
                "patience": 1,
                "threshold": 0.001,
                "threshold_mode": "abs",
                "min_lr": 1e-6,
            },
        )
    checkpoint = ModelCheckpoint(
        dirpath=folder / "checkpoints",
        filename="best-step-{step}",
        monitor=config["checkpoint_monitor"],
        mode="max",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )
    periodic = ModelCheckpoint(
        dirpath=folder / "checkpoints/periodic",
        filename="step-{step}",
        every_n_train_steps=100,
        save_top_k=1,
        save_last=True,
        save_on_train_epoch_end=False,
    )
    scaler = torch.amp.GradScaler("cuda", init_scale=config["initial_loss_scale"])
    loggers = [CSVLogger(str(folder / "csv"), name=stage)]
    wb = config.get("wandb", {})
    if wb.get("enabled"):
        os.environ.update(WANDB_DISABLE_CODE="true", WANDB_DISABLE_GIT="true")
        loggers.append(
            WandbLogger(
                project=wb["project"],
                entity=wb["entity"],
                group=wb["group"],
                id=wb["group"] + "-" + stage,
                name="NLLB-1.3B-LoRA-" + stage,
                job_type=stage,
                offline=wb["mode"] == "offline",
                log_model=False,
                save_dir=str(folder),
                config=config,
            )
        )
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        plugins=[MixedPrecision("16-mixed", "cuda", scaler=scaler)],
        max_epochs=config["sft_epochs"] if stage == "sft" else -1,
        max_steps=(2 * config["sft_pairs"] + config["batch_directional_records"] - 1)
        // config["batch_directional_records"]
        if stage == "sft"
        else config["rl_optimizer_steps"],
        default_root_dir=folder,
        logger=loggers,
        callbacks=[
            checkpoint,
            periodic,
            TrainingProgress(str(OUTPUT / "STOP"), progress_every=10),
        ],
        num_sanity_val_steps=0,
        check_val_every_n_epoch=None,
        val_check_interval=config["sft_validation_every_updates"]
        if stage == "sft"
        else config["validation_every_updates"],
        log_every_n_steps=1,
    )
    (folder / "precision.json").parent.mkdir(parents=True, exist_ok=True)
    (folder / "precision.json").write_text(
        json.dumps(
            {
                "precision_plugin": trainer.precision_plugin.precision,
                "loaded_base_dtype": "float16",
                "trainable_parameter_dtypes": sorted(
                    {str(p.dtype) for p in model.policy.parameters() if p.requires_grad}
                ),
                "initial_loss_scale": config["initial_loss_scale"],
                "gpu": torch.cuda.get_device_name(),
            },
            indent=2,
        )
    )
    trainer.fit(
        model,
        datamodule=data,
        ckpt_path=str(resume_checkpoint) if resume_checkpoint else None,
    )
    completed_steps = trainer.global_step
    trainer.save_checkpoint(folder / "checkpoints/handoff.ckpt")
    trainer.validate(model, datamodule=data)
    if stage == "rl" and completed_steps != config["rl_optimizer_steps"]:
        raise RuntimeError("RL did not finish the requested 10,000 optimizer steps")
    if stage == "sft" and completed_steps != 1250:
        raise RuntimeError("SFT did not finish the requested complete epoch")
    test_metrics = []
    if config.get("test_after_training"):
        test_metrics = trainer.test(
            model, datamodule=data, ckpt_path=checkpoint.best_model_path
        )
    result = {
        "completed_optimizer_steps": completed_steps,
        "best_comet_checkpoint": checkpoint.best_model_path,
        "best_comet": float(checkpoint.best_model_score)
        if checkpoint.best_model_score is not None
        else None,
        "handoff_checkpoint": str(folder / "checkpoints/handoff.ckpt"),
        "checkpoint_selection": "best source-conditioned COMET macro on fixed validation panel",
        "test_pairs": config.get("test_pairs"),
        "test_checkpoint": checkpoint.best_model_path,
        "test_metrics": test_metrics,
    }
    (folder / "result.json").write_text(json.dumps(result, indent=2))
    for logger in trainer.loggers:
        if isinstance(logger, WandbLogger):
            logger.experiment.finish()


def export(config, stage):
    import torch

    from src.models.anchored_grpo_module import AnchoredGRPOModule
    from src.models.sft_module import TranslationSFTModule

    result = json.loads((OUTPUT / stage / "result.json").read_text())
    path = result["best_comet_checkpoint"]
    if not path:
        raise RuntimeError("No evaluated best-COMET checkpoint exists")
    klass = TranslationSFTModule if stage == "sft" else AnchoredGRPOModule
    model = klass.load_from_checkpoint(
        path,
        map_location="cpu",
        weights_only=False,
        load_dtype="float16",
        reward=reward(config, stage),
    )
    destination = OUTPUT / (stage + "_export")
    destination.mkdir(exist_ok=False)
    policy = (
        model.policy.merge_and_unload(safe_merge=True)
        if model.uses_lora
        else model.policy
    )
    policy.save_pretrained(destination)
    model.tokenizer.save_pretrained(destination)
    if any(not torch.isfinite(p).all() for p in policy.parameters()):
        raise FloatingPointError("Nonfinite exported weights")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=[
            "bootstrap",
            "resume_pipeline",
            "sft",
            "rl",
            "export_sft",
            "export_rl",
        ],
        default="bootstrap",
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-phase", choices=["sft", "rl"], default="sft")
    args = parser.parse_args()
    os.environ.update(
        TOKENIZERS_PARALLELISM="false",
        HF_HUB_DISABLE_TELEMETRY="1",
        TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1",
        PYTORCH_ALLOC_CONF="expandable_segments:True",
    )
    config = json.loads(CONFIG.read_text())
    try:
        if args.stage == "bootstrap":
            bootstrap(config)
        elif args.stage == "resume_pipeline":
            pipeline(config, args.resume, args.resume_phase)
        elif args.stage.startswith("export_"):
            export(config, args.stage.removeprefix("export_"))
        else:
            train(config, args.stage, args.resume)
    except Exception as error:
        # Do not include request objects, headers or credentials in status files.
        state("failed", stage=args.stage, error_type=type(error).__name__)
        raise


if __name__ == "__main__":
    main()
