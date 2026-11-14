"""Actual interrupted-training checkpoint recovery and quota configuration."""

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import lightning as L
import pytest
import torch
from hydra import compose, initialize_config_dir

from scripts.main.pipeline import stage_overrides
from scripts.verify.assets import create_assets
from src.data.translation_datamodule import TranslationDataModule
from src.models.sft_module import TranslationSFTModule
from src.utils.rental_guard import RentalGuard, RentalQuotaReached, TrainingProgress


def test_relay_sends_metrics_without_checkpoint_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.rental.wandb_relay import WandbRelay

    for name in (
        "WANDB_API_KEY",
        "WANDB_DISABLE_CODE",
        "WANDB_DISABLE_GIT",
        "WANDB_CACHE_DIR",
        "WANDB_DATA_DIR",
    ):
        monkeypatch.setenv(name, "unit-local-key")
    records = []
    calls = []
    run = SimpleNamespace(
        url="https://wandb.ai/test/project/runs/test",
        summary={},
        define_metric=lambda *a, **k: None,
        log=records.append,
        finish=lambda: None,
    )

    def initialize(**kwargs: object) -> object:
        calls.append(kwargs)
        return run

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=initialize))
    stage = tmp_path / "checkpoints_and_logs/sft"
    stage.mkdir(parents=True)
    (stage / "resolved_config.yaml").write_text("model: {learning_rate: 0.0001}")
    (stage / "progress.jsonl").write_text(
        json.dumps({"global_step": 20, "metrics": {"train/loss": 0.8}}) + "\n"
    )
    (stage / "private.ckpt").write_bytes(b"checkpoint contents must never be uploaded")
    relay = WandbRelay(tmp_path, "unit-group", "unit-entity")
    relay.sync()
    relay.sync()
    assert records == [{"trainer/global_step": 20, "train/loss": 0.8}]
    assert calls[0]["project"] == "Machine Translation"
    assert "unit-local-key" not in json.dumps(calls[0])
    relay.finish()


def test_ssh_subprocess_environment_excludes_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import subprocess

    from scripts.rental.supervisor import command

    for name in ("WANDB_API_KEY", "HUGGING_FACE_API_KEY", "CLAUDE_AUTH_KEY"):
        monkeypatch.setenv(name, "local-only")

    def execute(*args: object, **kwargs: object) -> object:
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        assert not {
            "WANDB_API_KEY",
            "HUGGING_FACE_API_KEY",
            "CLAUDE_AUTH_KEY",
        }.intersection(environment)
        return SimpleNamespace(returncode=0, stdout="ok")

    monkeypatch.setattr(subprocess, "run", execute)
    assert command(["ssh", "test"]) == "ok"


def test_deadline_and_stop_file(tmp_path: Path) -> None:
    guard = RentalGuard(str(tmp_path / "STOP"), time.time() + 60)
    guard._check()
    guard.deadline_unix = time.time() - 1
    with pytest.raises(RentalQuotaReached):
        guard._check()
    guard.deadline_unix = time.time() + 60
    (tmp_path / "STOP").touch()
    with pytest.raises(RentalQuotaReached):
        guard._check()


def test_user_managed_progress_has_only_a_manual_stop(tmp_path: Path) -> None:
    stop = tmp_path / "manual-stop"
    progress = TrainingProgress(str(stop))
    assert progress.deadline_unix is None
    progress._check()
    stop.touch()
    with pytest.raises(RentalQuotaReached):
        progress._check()


def test_quota_checkpoint_preserves_real_optimizer_and_resume(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    assets = create_assets(tmp_path / "assets", "causal")
    stop = tmp_path / "STOP"

    class StopAfterUpdate(L.Callback):
        def on_train_batch_end(self, *args: object) -> None:
            stop.touch()

    model = TranslationSFTModule(
        model_name_or_path=str(assets),
        max_target_length=4,
        max_new_tokens=3,
        max_source_length=32,
        profiling={"enabled": True},
    )
    data = TranslationDataModule(
        train_file="data/example_train.jsonl",
        val_file="data/example_val.jsonl",
        batch_size=1,
    )
    options = {
        "accelerator": "cpu",
        "devices": 1,
        "logger": False,
        "enable_checkpointing": False,
        "enable_model_summary": False,
        "enable_progress_bar": False,
        "num_sanity_val_steps": 0,
        "limit_val_batches": 0,
        "max_epochs": 1,
    }
    trainer = L.Trainer(
        **options,
        default_root_dir=tmp_path / "run",
        callbacks=[RentalGuard(str(stop), time.time() + 60), StopAfterUpdate()],
    )
    with pytest.raises(RentalQuotaReached):
        trainer.fit(model, datamodule=data)
    checkpoint = tmp_path / "run/checkpoints/interrupted.ckpt"
    saved = torch.load(checkpoint, weights_only=True)
    assert saved["global_step"] == 1 and saved["optimizer_states"][0]["state"]
    restored = TranslationSFTModule.load_from_checkpoint(checkpoint, weights_only=True)
    resumed = L.Trainer(**options, default_root_dir=tmp_path / "resumed")
    resumed.fit(restored, datamodule=data, ckpt_path=str(checkpoint))
    assert resumed.global_step > 1
    # A larger batch must process only the remaining shuffled examples and keep
    # the original optimizer-step counter/Adam state.
    from scripts.tools.rebatch_checkpoint import migrate

    migrated = tmp_path / "rebatch.ckpt"
    consumed = migrate(checkpoint, migrated)
    assert consumed == 1
    rebatched = TranslationDataModule(
        train_file="data/example_train.jsonl",
        val_file="data/example_val.jsonl",
        batch_size=2,
        resume_skip_examples=consumed,
    )
    rebatched.setup("fit")
    expected = len(rebatched.train_dataloader())
    restored = TranslationSFTModule.load_from_checkpoint(migrated, weights_only=False)
    rebatch_trainer = L.Trainer(**options, default_root_dir=tmp_path / "rebatch")
    rebatch_trainer.fit(restored, datamodule=rebatched, ckpt_path=str(migrated))
    assert rebatch_trainer.global_step == consumed + expected
    dynamic = TranslationDataModule(
        train_file="data/example_train.jsonl",
        val_file="data/example_val.jsonl",
        batch_size=2,
        resume_skip_examples=consumed,
        token_budget=80,
        budget_tokenizer_name_or_path=str(assets),
        token_budget_source_cap=32,
        token_budget_target_cap=4,
    )
    dynamic.setup("fit")
    dynamic_batches = len(dynamic.train_dataloader())
    dynamic_model = TranslationSFTModule.load_from_checkpoint(
        migrated, weights_only=False
    )
    dynamic_trainer = L.Trainer(**options, default_root_dir=tmp_path / "dynamic")
    dynamic_trainer.fit(dynamic_model, datamodule=dynamic, ckpt_path=str(migrated))
    assert dynamic_trainer.global_step == consumed + dynamic_batches


def test_rebatch_continues_original_shuffle_and_online_augmentation() -> None:
    options = {
        "train_file": "data/example_train.jsonl",
        "val_file": "data/example_val.jsonl",
        "bidirectional": True,
        "augmentation": {
            "enabled": True,
            "whitespace_probability": 1.0,
            "unicode_probability": 1.0,
            "concat_probability": 1.0,
        },
    }
    first = TranslationDataModule(batch_size=1, **options)
    first.setup("fit")
    original = list(first.train_dataloader())
    resumed = TranslationDataModule(batch_size=2, resume_skip_examples=1, **options)
    resumed.setup("fit")
    remaining = [
        {
            key: [values[index]]
            for key, values in batch.items()
            if key != "augmentation_seconds"
        }
        for batch in resumed.train_dataloader()
        for index in range(len(batch["source"]))
    ]
    expected = [
        {key: values for key, values in batch.items() if key != "augmentation_seconds"}
        for batch in original[1:]
    ]
    assert remaining == expected


@pytest.mark.parametrize("stage", ["sft", "rl"])
def test_rental_configs_have_budget_checkpoint_and_online_logging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("PROJECT_ROOT", str(root))
    monkeypatch.setenv(
        "COMET_CHECKPOINT", "/workspace/cometkiwi/checkpoints/model.ckpt"
    )
    monkeypatch.setenv("TRAIN_RUN_GROUP", "unit-test")
    with initialize_config_dir(version_base="1.3", config_dir=str(root / "configs")):
        cfg = compose(
            config_name="sft" if stage == "sft" else "grpo",
            overrides=stage_overrides(stage, tmp_path, time.time() + 3600, "abc"),
        )
    assert cfg.logger.wandb.project == "Machine Translation"
    assert cfg.logger.wandb.offline and cfg.logger.wandb.log_model is False
    assert cfg.callbacks.periodic.every_n_train_steps == 100
    assert cfg.callbacks.rental_guard.deadline_unix > time.time()
    assert cfg.model.max_new_tokens == 128 and cfg.data.bidirectional
    assert cfg.test == (stage == "rl")
