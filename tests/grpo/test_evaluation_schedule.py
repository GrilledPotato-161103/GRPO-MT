"""Translate optimizer-update cadence into Lightning batches and retain final eval."""

import json
import time
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from scripts.main.pipeline import stage_overrides
from scripts.verify.assets import create_assets
from src.train import train


@pytest.mark.parametrize("stage,expected_batches", [("sft", 10000), ("rl", 5000)])
def test_optimizer_update_evaluation_schedule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str, expected_batches: int
) -> None:
    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("PROJECT_ROOT", str(root))
    monkeypatch.setenv(
        "COMET_CHECKPOINT", "/workspace/cometkiwi/checkpoints/model.ckpt"
    )
    monkeypatch.setenv("TRAIN_RUN_GROUP", "unit-scheduled")
    (tmp_path / "rental_config.json").write_text(
        json.dumps(
            {
                "experiments": {"rl": "phomt_rl_throughput"},
                "evaluation_every_optimizer_steps": 10000,
                "length_batching": True,
                "group_size": 4,
                "num_iterations": 2,
                "logprob_batch_size": 8,
                "logprob_token_budget": 3072,
                "batch_sizes": {"sft": 24, "rl": 16},
            }
        )
    )
    with initialize_config_dir(version_base="1.3", config_dir=str(root / "configs")):
        cfg = compose(
            config_name="sft" if stage == "sft" else "grpo",
            overrides=stage_overrides(stage, tmp_path, time.time() + 3600, "pinned"),
        )
    assert cfg.trainer.val_check_interval == expected_batches
    assert cfg.trainer.check_val_every_n_epoch is None and cfg.final_validate
    assert cfg.data.length_batching.enabled
    assert cfg.data.length_batching.candidate_multiplier == (1 if stage == "sft" else 4)
    if stage == "rl":
        assert cfg.model.logprob_batch_size == 8 and cfg.data.batch_size == 16


def test_sparse_interval_saves_and_evaluates_at_completion(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    asset = create_assets(tmp_path / "model", "causal")
    cfg = OmegaConf.create(
        {
            "seed": 42,
            "train": True,
            "test": True,
            "final_validate": True,
            "data": {
                "_target_": "src.data.translation_datamodule.TranslationDataModule",
                "train_file": "data/example_train.jsonl",
                "val_file": "data/example_val.jsonl",
                "test_file": "data/example_test.jsonl",
                "bidirectional": True,
                "batch_size": 2,
            },
            "model": {
                "_target_": "src.models.grpo_module.GRPOModule",
                "model_name_or_path": str(asset),
                "max_new_tokens": 3,
                "group_size": 2,
                "num_iterations": 2,
                "reward": {
                    "_target_": "src.models.components.translation_reward.TranslationReward",
                    "chrf_weight": 1.0,
                    "bleu_weight": 0.3,
                },
            },
            "callbacks": {
                "model_checkpoint": {
                    "_target_": "lightning.pytorch.callbacks.ModelCheckpoint",
                    "dirpath": str(tmp_path / "checkpoints"),
                    "save_last": True,
                    "save_top_k": 1,
                    "monitor": "val/chrf_macro",
                    "save_on_train_epoch_end": False,
                }
            },
            "logger": None,
            "trainer": {
                "_target_": "lightning.pytorch.Trainer",
                "accelerator": "cpu",
                "devices": 1,
                "max_epochs": 1,
                "limit_train_batches": 1,
                "num_sanity_val_steps": 0,
                "val_check_interval": 5000,
                "check_val_every_n_epoch": None,
                "enable_progress_bar": False,
                "enable_model_summary": False,
                "default_root_dir": str(tmp_path / "run"),
            },
            "paths": {"output_dir": str(tmp_path / "run")},
        }
    )
    metrics, objects = train(cfg)
    assert objects["trainer"].global_step == 2
    assert Path(objects["trainer"].checkpoint_callback.last_model_path).is_file()
    assert {
        "val/chrf_macro",
        "val/bleu_macro",
        "test/chrf_macro",
        "test/bleu_macro",
    } <= metrics.keys()
    history = [
        json.loads(line)
        for line in (tmp_path / "run/translation_metrics/evaluation_history.jsonl")
        .read_text()
        .splitlines()
    ]
    assert [row["stage"] for row in history] == ["val", "test"]
    assert all(row["global_step"] == 2 for row in history)
