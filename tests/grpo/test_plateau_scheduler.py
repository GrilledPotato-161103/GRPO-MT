"""Native plateau reductions, migration and manual-Lightning checkpoint resume."""

from copy import deepcopy
from pathlib import Path

import lightning as L
import pytest
import torch

from scripts.main.common import DEFAULT_CONFIG, compose_stage, load_config
from scripts.verify.assets import create_assets
from src.data.translation_datamodule import TranslationDataModule
from src.models.components.optim.plateau import ValidationPlateau, plateau_options
from src.models.sft_module import TranslationSFTModule


def test_reduces_only_completed_unique_validations_and_restores_state() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=5e-6)
    parameter.square().backward()
    optimizer.step()
    moment = deepcopy(optimizer.state_dict()["state"])
    options = {"patience": 1, "initial_best": 0.8}
    control = ValidationPlateau(optimizer, 2e-5, options)
    assert optimizer.param_groups[0]["lr"] == 2e-5
    assert optimizer.param_groups[0]["initial_lr"] == 2e-5
    assert torch.equal(
        optimizer.state_dict()["state"][0]["exp_avg"], moment[0]["exp_avg"]
    )
    control.step({"val/comet_macro": 0.8}, 100)
    assert control.step({"val/comet_macro": 0.8}, 100) == {}
    assert optimizer.param_groups[0]["lr"] == 2e-5
    control.step({"val/comet_macro": 0.8}, 200)
    assert optimizer.param_groups[0]["lr"] == 1e-5
    state = deepcopy(control.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    resumed_optimizer = torch.optim.AdamW(
        [torch.nn.Parameter(torch.tensor(1.0))], lr=2e-5
    )
    resumed_optimizer.load_state_dict(optimizer_state)
    resumed = ValidationPlateau(resumed_optimizer, 2e-5, options, state)
    assert resumed.resumed and resumed.last_validation_step == 200
    assert resumed_optimizer.param_groups[0]["lr"] == 1e-5
    resumed.step({"val/comet_macro": 0.805}, 300)
    assert resumed.scheduler.best == 0.805
    for step in range(400, 2000, 100):
        resumed.step({"val/comet_macro": 0.805}, step)
    assert resumed_optimizer.param_groups[0]["lr"] == 1e-6


@pytest.mark.parametrize(
    "config",
    [
        {"factor": 1.0},
        {"patience": -1},
        {"monitor": "train/loss"},
        {"threshold": float("nan")},
        {"unknown": 2},
        {"initial_best": float("inf")},
    ],
)
def test_rejects_invalid_scheduler_configuration(config: dict) -> None:
    with pytest.raises(ValueError):
        plateau_options(config)


def test_missing_or_nonfinite_validation_fails() -> None:
    control = ValidationPlateau(
        torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))]), 2e-5, {}
    )
    for metrics in [{}, {"val/comet_macro": float("nan")}]:
        with pytest.raises(ValueError, match="scheduler metric"):
            control.step(metrics, 10)


def test_profile_composes_only_rl_scheduler() -> None:
    cfg = load_config(
        DEFAULT_CONFIG.with_name("qwen3_1p7b_4090_plateau.yaml"),
        ["wandb.enabled=false"],
    )
    sft, rl = compose_stage(cfg, "sft", None), compose_stage(cfg, "rl", None)
    assert "lr_scheduler" not in sft.model
    assert rl.model.learning_rate == 2e-5 and rl.model.lr_scheduler.mode == "max"
    assert rl.model.lr_scheduler.monitor == "val/comet_macro"
    assert rl.trainer.limit_val_batches == rl.trainer.limit_test_batches == 200


def test_lightning_manual_resume_overrides_old_lr_then_keeps_reduction(
    tmp_path: Path,
) -> None:
    torch.set_num_threads(1)
    asset = create_assets(tmp_path / "base", "causal")
    module = TranslationSFTModule(
        model_name_or_path=str(asset),
        max_source_length=24,
        max_new_tokens=3,
        max_target_length=4,
        group_size=2,
        num_iterations=1,
        learning_rate=5e-6,
        warmup_steps=1,
    )
    data = TranslationDataModule(
        train_file=str(
            Path(__file__).resolve().parents[2] / "data/example_train.jsonl"
        ),
        val_file=str(Path(__file__).resolve().parents[2] / "data/example_val.jsonl"),
        bidirectional=True,
        batch_size=2,
    )
    options = {
        "accelerator": "cpu",
        "devices": 1,
        "logger": False,
        "enable_checkpointing": False,
        "enable_model_summary": False,
        "enable_progress_bar": False,
        "num_sanity_val_steps": 0,
        "limit_train_batches": 2,
        "limit_val_batches": 0,
        "max_epochs": -1,
    }
    trainer = L.Trainer(max_steps=2, default_root_dir=tmp_path / "old", **options)
    trainer.fit(module, datamodule=data)
    old = tmp_path / "old.ckpt"
    trainer.save_checkpoint(old)
    before = torch.load(old, weights_only=False)["optimizer_states"][0]["state"]
    migrated = TranslationSFTModule.load_from_checkpoint(
        old,
        weights_only=False,
        learning_rate=2e-5,
        warmup_steps=100,
        lr_scheduler={"monitor": "val/bleu", "patience": 0, "initial_best": 99.0},
    )
    trainer = L.Trainer(
        max_steps=4,
        default_root_dir=tmp_path / "new",
        **{**options, "limit_val_batches": 1},
    )
    trainer.fit(migrated, datamodule=data, ckpt_path=str(old))
    assert migrated._plateau is not None
    # Real validation hooks reduced the rate, and duplicate final eval cannot do so again.
    assert trainer.optimizers[0].param_groups[0]["lr"] == 1e-5
    assert migrated._plateau.step({"val/bleu": 0.0}, trainer.global_step) == {}
    checkpoint = tmp_path / "scheduled.ckpt"
    trainer.save_checkpoint(checkpoint)
    saved = torch.load(checkpoint, weights_only=False)
    assert saved["validation_plateau"]["last_validation_step"] == 4
    assert before.keys() == saved["optimizer_states"][0]["state"].keys()
    resumed = TranslationSFTModule.load_from_checkpoint(checkpoint, weights_only=False)
    trainer = L.Trainer(max_steps=6, default_root_dir=tmp_path / "again", **options)
    trainer.fit(resumed, datamodule=data, ckpt_path=str(checkpoint))
    assert resumed._plateau is not None and resumed._plateau.resumed
    assert trainer.optimizers[0].param_groups[0]["lr"] == 1e-5
