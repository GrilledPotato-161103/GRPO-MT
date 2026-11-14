"""Run real PhoMT SFT then GRPO, with online W&B and an external quota guard."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def write_state(root: Path, status: str, **details: Any) -> None:
    record = {"status": status, "unix_time": time.time(), **details}
    temporary = root / "state.json.tmp"
    temporary.write_text(json.dumps(record, indent=2), encoding="utf-8")
    temporary.replace(root / "state.json")


def file_record(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        return {
            "size": path.stat().st_size,
            "sha256": hashlib.file_digest(stream, "sha256").hexdigest(),
        }


def cache_assets(root: Path) -> str:
    """Cache public model assets and validate private-data hashes before GPU work."""
    from huggingface_hub import snapshot_download

    from src.models.components.comet_reward import CometScorer

    runtime = (
        json.loads((root / "rental_config.json").read_text())
        if (root / "rental_config.json").exists()
        else {}
    )
    manifest_path = Path(
        runtime.get("dataset_manifest", "data/phomt_stages/manifest.json")
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = {
        "sft": Path("data/phomt_stages/sft.csv"),
        "rl": Path("data/phomt_stages/rl.csv"),
        "validation": Path("data/phomt_prepared/validation.csv"),
        "test": Path("data/phomt_test_selected.csv"),
    }
    paths.update(
        {name: Path(path) for name, path in runtime.get("data_files", {}).items()}
    )
    for name, path in paths.items():
        record = manifest.get("outputs", {}).get(name) or manifest.get(
            "inputs", {}
        ).get(name)
        if record is None:
            raise ValueError(f"Dataset manifest is missing {name}")
        expected = record["sha256"]
        if file_record(path)["sha256"] != expected:
            raise ValueError(f"Dataset hash mismatch: {name}")
    model_name = runtime.get("model_name", "Qwen/Qwen2.5-7B-Instruct")
    snapshot = (
        str(Path(model_name).resolve())
        if Path(model_name).is_dir()
        else snapshot_download(
            model_name,
            revision=runtime.get("policy_revision", "main"),
            allow_patterns=[
                "*.json",
                "*.safetensors",
                "pytorch_model*.bin",
                "*.txt",
                "*.model",
                "*.jinja",
            ],
        )
    )
    scores = []
    if runtime.get("reward", {}).get("comet_weight", 0.7) > 0:
        scorer = CometScorer(os.environ["COMET_CHECKPOINT"], device="cpu")
        scores = scorer(["The book has ten pages."], ["Cuốn sách có mười trang."])
    revision = (
        runtime.get("policy_revision", "main")
        if Path(model_name).is_dir()
        else Path(snapshot).name
    )
    cached_snapshots = []
    for directory in (Path(os.environ["HF_HOME"]) / "hub").glob(
        "models--*/snapshots/*"
    ):
        files = {}
        for path in directory.iterdir():
            if path.is_file():
                with path.open("rb") as stream:
                    files[path.name] = {
                        "size": path.stat().st_size,
                        "sha256": hashlib.file_digest(stream, "sha256").hexdigest(),
                    }
        cached_snapshots.append(
            {
                "remote_path": str(directory),
                "cache_name": directory.parents[1].name,
                "revision": directory.name,
                "files": files,
            }
        )
    (root / "cache_manifest.json").write_text(
        json.dumps(
            {
                "policy": model_name,
                "policy_revision": revision,
                "comet_checkpoint": os.environ["COMET_CHECKPOINT"],
                "comet_probe": scores,
                "dataset": manifest,
                "cached_before_training": True,
                "hf_snapshots": cached_snapshots,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return revision


def hydra_value(value: Any) -> str:
    """Serialize nested mappings with Hydra keys and JSON-quoted string values."""
    if isinstance(value, dict):
        return (
            "{"
            + ",".join(f"{key}:{hydra_value(item)}" for key, item in value.items())
            + "}"
        )
    if isinstance(value, list):
        return "[" + ",".join(hydra_value(item) for item in value) + "]"
    return json.dumps(value)


def stage_overrides(
    stage: str,
    root: Path,
    deadline: float | None,
    revision: str,
    runtime: dict[str, Any] | None = None,
) -> list[str]:
    """Keep credentials in the process environment, never in Hydra overrides."""
    runtime = (
        runtime
        if runtime is not None
        else (
            json.loads((root / "rental_config.json").read_text())
            if (root / "rental_config.json").exists()
            else {}
        )
    )
    options = [
        f"experiment={runtime.get('experiments', {}).get(stage, f'phomt_{stage}')}",
        f"seed={runtime.get('seed', 42)}",
        "logger=wandb",
        f"test={str(stage == 'rl' and runtime.get('test_after_training', True)).lower()}",
        f"hydra.run.dir={root / stage}",
        f"trainer.max_epochs={runtime.get('max_epochs', {}).get(stage, 1)}",
        "trainer.min_epochs=0",
        f"++trainer.log_every_n_steps={runtime.get('log_every_n_steps', 10)}",
        "++trainer.val_check_interval=1.0",
        "+trainer.limit_val_batches=1.0",
        f"model.max_new_tokens={runtime.get('target_cap', 128)}",
        f"model.max_source_length={runtime.get('source_cap', 512)}",
        f"model.warmup_steps={runtime.get('warmup_steps', {}).get(stage, 100 if stage == 'sft' else 8)}",
        f"model.reward.comet_device={runtime.get('comet_device', 'cpu')}",
        f"model.reward.comet_batch_size={runtime.get('comet_batch_size', 8)}",
        f"model.reward.comet_model_name={os.environ['COMET_CHECKPOINT']}",
        f"logger.wandb.project={json.dumps(runtime.get('wandb', {}).get('project', 'Machine Translation'))}",
        f"logger.wandb.offline={str(runtime.get('wandb_offline', True)).lower()}",
        "logger.wandb.log_model=false",
        f"logger.wandb.id={os.environ['TRAIN_RUN_GROUP']}-{stage}",
        f"logger.wandb.group={os.environ['TRAIN_RUN_GROUP']}",
        f"+logger.wandb.name={json.dumps(Path(runtime.get('model_name', 'Qwen/Qwen2.5-7B')).name + '-LoRA-' + stage)}",
        "+logger.wandb.resume=allow",
        f"logger.wandb.job_type={stage}",
        "+callbacks.periodic._target_=lightning.pytorch.callbacks.ModelCheckpoint",
        f"+callbacks.periodic.dirpath={root / stage / 'checkpoints/periodic'}",
        "+callbacks.periodic.filename='step_{step}'",
        "+callbacks.periodic.every_n_train_steps=100",
        "+callbacks.periodic.save_top_k=1",
        "+callbacks.periodic.save_last=true",
        "+callbacks.periodic.save_on_train_epoch_end=false",
        "+callbacks.rental_guard._target_=src.utils.rental_guard.RentalGuard",
        f"+callbacks.rental_guard.stop_file={root / 'STOP'}",
        f"+callbacks.rental_guard.deadline_unix={deadline}",
    ]
    if deadline is None:
        options = [
            option for option in options if "callbacks.rental_guard." not in option
        ]
        options += [
            "+callbacks.training_progress._target_=src.utils.rental_guard.TrainingProgress",
            f"+callbacks.training_progress.stop_file={root / 'STOP'}",
        ]
    if not runtime.get("early_stopping", True):
        options += ["callbacks.early_stopping=null"]
    if "optimizer" in runtime:
        options += [f"optimizer={runtime['optimizer']}"]
    if runtime.get("lr_scheduler", {}).get(stage):
        options += [
            f"++model.lr_scheduler={hydra_value(runtime['lr_scheduler'][stage])}"
        ]
    if stage in runtime.get("max_optimizer_steps", {}):
        options += [f"++trainer.max_steps={runtime['max_optimizer_steps'][stage]}"]
    if not runtime.get("wandb", {}).get("enabled", True):
        options = [
            option for option in options if not option.lstrip("+").startswith("logger")
        ]
        options.insert(1, "logger=csv")
    elif runtime.get("wandb", {}).get("entity"):
        options += [f"+logger.wandb.entity={json.dumps(runtime['wandb']['entity'])}"]
    for name, value in runtime.get("trainer", {}).items():
        options += [f"++trainer.{name}={json.dumps(value)}"]
    if runtime.get("trainer", {}).get("enable_progress_bar") is False:
        options += ["callbacks.rich_progress_bar=null"]
    for name, value in runtime.get("lora", {}).items():
        options += [f"++model.lora.{name}={json.dumps(value)}"]
    if "prompt_template" in runtime:
        # Hydra quoted strings retain literal \\n; pass native line breaks instead.
        prompt = json.dumps(runtime["prompt_template"]).replace("\\n", "\n")
        options += [f"model.prompt_template={prompt}"]
    for name in (
        "use_chat_template",
        "chat_template_kwargs",
        "source_content_limit",
        "strict_prompt_length",
        "language_codes",
    ):
        if name in runtime:
            options += [f"++model.{name}={hydra_value(runtime[name])}"]
    for name in ("load_dtype", "gradient_checkpointing", "quantization"):
        if name in runtime:
            options += [f"model.{name}={json.dumps(runtime[name])}"]
    for name, value in runtime.get("reward", {}).items():
        options += [f"model.reward.{name}={value}"]
    if runtime.get("sac_reward"):
        options += [f"++model.reward.sac={hydra_value(runtime['sac_reward'])}"]
    if stage == "rl" and runtime.get("correction") is not None:
        options += [
            "model._target_=src.models.anchored_grpo_module.AnchoredGRPOModule",
            f"++model.correction={hydra_value(runtime['correction'])}",
        ]
    if stage == "rl":
        if "beta" in runtime:
            options += [f"model.beta={runtime['beta']}"]
        for name, value in runtime.get("kl_control", {}).items():
            options += [f"++model.kl_control.{name}={hydra_value(value)}"]
        if runtime.get("reward_harness"):
            options += [
                f"++model.reward.harness={hydra_value(runtime['reward_harness'])}"
            ]
    for name, value in runtime.get("augmentation", {}).items():
        options += [f"data.augmentation.{name}={json.dumps(value)}"]
    for name, value in runtime.get("data_loader", {}).items():
        options += [f"++data.{name}={json.dumps(value)}"]
    if runtime.get("evaluation", {}).get("batch_size"):
        options += [
            f"++data.evaluation_batch_size={runtime['evaluation']['batch_size']}"
        ]
    if "paired_batches" in runtime:
        options += [f"++data.paired_batches={json.dumps(runtime['paired_batches'])}"]
    for name, field in (
        (stage, "train_file"),
        ("validation", "val_file"),
        ("test", "test_file"),
    ):
        if name in runtime.get("data_files", {}):
            options += [f"data.{field}={json.dumps(runtime['data_files'][name])}"]
    interval = runtime.get("evaluation_every_optimizer_steps")
    if interval is not None:
        updates_per_batch = 1 if stage == "sft" else runtime.get("num_iterations", 2)
        if (
            isinstance(interval, bool)
            or not isinstance(interval, int)
            or interval < 1
            or interval % updates_per_batch
        ):
            raise ValueError(
                "Evaluation interval must be positive and divisible by optimizer updates per batch"
            )
        options = [
            option
            for option in options
            if not option.startswith("++trainer.val_check_interval=")
        ]
        options += [
            f"++trainer.val_check_interval={interval // updates_per_batch}",
            "++trainer.check_val_every_n_epoch=null",
            "++trainer.num_sanity_val_steps=0",
            "++final_validate=true",
        ]
    if runtime.get("length_batching", False):
        target_reserve = runtime.get("target_cap", 128) + int(
            bool(runtime.get("language_codes"))
        )
        options += [
            "data.length_batching.enabled=true",
            "data.length_batching.tokenizer_name_or_path=${model.model_name_or_path}",
            f"data.length_batching.tokenizer_revision={revision if stage == 'sft' else 'null'}",
            f"data.length_batching.reserve_target_tokens={target_reserve}",
            f"data.length_batching.target_cap={target_reserve}",
            f"data.length_batching.candidate_multiplier={1 if stage == 'sft' else runtime.get('group_size', 4)}",
        ]
        if runtime.get("length_cache_dir"):
            options += [
                f"++data.length_batching.cache_dir={json.dumps(runtime['length_cache_dir'])}"
            ]
    elif "length_batching" in runtime:
        options += ["data.length_batching.enabled=false"]
    if stage == "sft":
        if runtime.get("sft_microbatch_size") is not None:
            options += [f"++model.microbatch_size={runtime['sft_microbatch_size']}"]
        if runtime.get("language_codes"):
            options += [f"model.max_target_length={runtime.get('target_cap', 128) + 1}"]
        options += [
            f"model.base_revision={revision}",
            f"model.learning_rate={runtime.get('learning_rate', {}).get('sft', 1e-4)}",
        ]
        if "model_name" in runtime:
            options += [f"model.model_name_or_path={json.dumps(runtime['model_name'])}"]
    else:
        options += [
            f"model.learning_rate={runtime.get('learning_rate', {}).get('rl', 5e-6)}"
        ]
        if "sft_export_dir" in runtime:
            options += [
                f"model.model_name_or_path={json.dumps(runtime['sft_export_dir'])}"
            ]
        if "logprob_batch_size" in runtime:
            options += [
                f"model.logprob_batch_size={runtime['logprob_batch_size']}",
                f"model.logprob_token_budget={runtime.get('logprob_token_budget', 3072)}",
                f"model.group_size={runtime.get('group_size', 4)}",
                f"model.num_iterations={runtime.get('num_iterations', 2)}",
            ]
    tuning = root / "batch_sizes.json"
    selected = dict(runtime.get("batch_sizes", {}))
    if tuning.is_file():
        selected = json.loads(tuning.read_text(encoding="utf-8"))["selected"]
        selected.update(runtime.get("batch_sizes", {}))
    if stage in selected:
        options += [f"data.batch_size={selected[stage]}"]
    resume = root / ("rebatch_resume.json" if stage == "sft" else "rl_resume.json")
    if runtime.get("resume_checkpoints", {}).get(stage):
        options += [f"ckpt_path={json.dumps(runtime['resume_checkpoints'][stage])}"]
    elif resume.is_file():
        metadata = json.loads(resume.read_text(encoding="utf-8"))
        options += [
            f"ckpt_path={metadata['checkpoint']}",
            f"+data.resume_skip_examples={metadata['consumed_examples']}",
        ]
    return options


def run(root: Path, deadline: float | None) -> None:
    """Finish stages when quota permits; interrupted checkpoints remain resumable."""
    import gc

    import torch
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from src.models.sft_module import export_sft
    from src.train import train
    from src.utils.rental_guard import RentalQuotaReached

    torch.set_num_threads(8)
    root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WANDB_DISABLE_CODE", "true")
    os.environ.setdefault("WANDB_DISABLE_GIT", "true")
    revision = os.environ.get("TRAIN_CACHE_READY_REVISION")
    if not revision:
        write_state(root, "caching")
        revision = cache_assets(root)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["TRAIN_CACHE_READY_REVISION"] = revision
        # Hub/Transformers read offline mode at import time. Restart with the
        # complete cache and no live model tensors from the preflight process.
        subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.main.pipeline",
                "--output",
                str(root),
                *(["--deadline", str(deadline)] if deadline is not None else []),
            ],
            check=True,
        )
        return
    try:
        runtime_file = root / "rental_config.json"
        runtime = json.loads(runtime_file.read_text()) if runtime_file.exists() else {}
        start_stage = runtime.get("start_stage", "sft")
        if start_stage not in {"sft", "rl"}:
            raise ValueError("start_stage must be sft or rl")
        end_stage = runtime.get("end_stage", "rl")
        if end_stage not in {"sft", "rl"} or (
            start_stage == "rl" and end_stage == "sft"
        ):
            raise ValueError("Invalid start/end stage combination")
        stages = (
            ("sft",)
            if end_stage == "sft"
            else ("rl",)
            if start_stage == "rl"
            else ("sft", "rl")
        )
        for stage in stages:
            if (root / "STOP").exists() or (
                deadline is not None and time.time() >= deadline
            ):
                write_state(root, "quota_stopped", stage=stage)
                return
            stage_dir = root / stage
            stage_dir.mkdir(parents=True, exist_ok=True)
            with initialize_config_dir(
                version_base="1.3", config_dir=str(Path("configs").resolve())
            ):
                cfg = compose(
                    config_name="sft" if stage == "sft" else "grpo",
                    overrides=stage_overrides(stage, root, deadline, revision),
                )
            # Programmatic composition has no Hydra runtime singleton.
            cfg.paths.output_dir = str(stage_dir)
            cfg.paths.work_dir = str(Path.cwd())
            OmegaConf.update(cfg, "trusted_checkpoint", True, force_add=True)
            if (
                stage == "sft"
                and (root / "batch_sizes.json").exists()
                and (
                    (root / "rental_config.json").exists()
                    and json.loads((root / "rental_config.json").read_text()).get(
                        "use_token_budget", False
                    )
                )
            ):
                selected = json.loads((root / "batch_sizes.json").read_text())[
                    "selected"
                ]["sft"]
                for key, value in {
                    "token_budget": json.loads(
                        (root / "batch_sizes.json").read_text()
                    ).get(
                        "sft_token_budget",
                        selected
                        * (cfg.model.max_source_length + cfg.model.max_target_length),
                    ),
                    "budget_tokenizer_name_or_path": cfg.model.model_name_or_path,
                    "budget_tokenizer_revision": revision,
                    "token_budget_source_cap": cfg.model.max_source_length,
                    "token_budget_target_cap": cfg.model.max_target_length,
                    "token_budget_prompt_template": cfg.model.prompt_template,
                }.items():
                    OmegaConf.update(cfg, f"data.{key}", value, force_add=True)
                OmegaConf.update(
                    cfg, "trainer.reload_dataloaders_every_n_epochs", 1, force_add=True
                )
            OmegaConf.save(cfg, stage_dir / "resolved_config.yaml", resolve=True)
            write_state(root, "training", stage=stage, policy_revision=revision)
            metrics, objects = train(cfg)
            trainer = objects["trainer"]
            checkpoint = (
                trainer.checkpoint_callback.best_model_path
                or trainer.checkpoint_callback.last_model_path
            )
            if runtime.get("handoff_checkpoint") == "last":
                final_checkpoint = stage_dir / "checkpoints/handoff.ckpt"
                final_checkpoint.parent.mkdir(parents=True, exist_ok=True)
                trainer.save_checkpoint(final_checkpoint)
                checkpoint = str(final_checkpoint)
            if not checkpoint:
                raise RuntimeError("A completed stage must have a resumable checkpoint")
            validation_path = stage_dir / "translation_metrics/val.json"
            evaluated_count = (
                sum(
                    json.loads(validation_path.read_text(encoding="utf-8"))[
                        "counts"
                    ].values()
                )
                if validation_path.exists()
                else 0
            )
            (stage_dir / "result.json").write_text(
                json.dumps(
                    {
                        "checkpoint": checkpoint,
                        "global_step": trainer.global_step,
                        "metrics": {
                            key: float(value) for key, value in metrics.items()
                        },
                        "status": "completed",
                        "validation_directional_examples": evaluated_count,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            del objects, trainer, metrics
            gc.collect()
            torch.cuda.empty_cache()
            if stage == "sft":
                write_state(root, "exporting_sft", checkpoint=checkpoint)
                export_dir = Path(runtime.get("sft_export_dir", "exports/phomt_sft"))
                export_sft(Path(checkpoint), export_dir)
                (root / "sft_export_manifest.json").write_text(
                    json.dumps(
                        {
                            path.name: {
                                "size": path.stat().st_size,
                                "sha256": file_record(path)["sha256"],
                            }
                            for path in export_dir.iterdir()
                            if path.is_file()
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                (root / "SFT_EXPORT_READY").touch()
        write_state(root, "completed", policy_revision=revision)
    except RentalQuotaReached:
        write_state(root, "quota_stopped", stage=stage)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline", type=float)
    args = parser.parse_args()
    root = args.output.resolve()
    try:
        run(root, args.deadline)
    except Exception as error:
        root.mkdir(parents=True, exist_ok=True)
        write_state(root, "failed", error_type=type(error).__name__)
        raise


if __name__ == "__main__":
    main()
