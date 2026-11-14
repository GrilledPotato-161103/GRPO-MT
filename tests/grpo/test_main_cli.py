"""Manual CLI configuration, isolation and real tiny-model train/eval roundtrip."""

import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from transformers import Qwen2Config, Qwen2ForCausalLM

from scripts.data.prepare_full import prepare_full
from scripts.data.prepare_stages import prepare
from scripts.main.common import DEFAULT_CONFIG, compose_stage, deadline, load_config
from scripts.main.eval import main as evaluate
from scripts.main.train import main as train
from scripts.verify.assets import create_assets


def test_default_configuration_and_cadence() -> None:
    cfg = load_config(DEFAULT_CONFIG, ["wandb.enabled=false"])
    cfg["run_group"] = "unit-main"
    sft = compose_stage(cfg, "sft", 9999999999)
    rl = compose_stage(cfg, "rl", 9999999999)
    assert (
        sft.trainer.val_check_interval == 1000 and rl.trainer.val_check_interval == 500
    )
    assert rl.data.batch_size == 16 and rl.model.logprob_batch_size == 8
    assert sft.data.num_workers == rl.data.num_workers == 12
    assert rl.data.persistent_workers and rl.data.prefetch_factor == 2
    assert sft.data.length_batching.tokenizer_revision == cfg["policy_revision"]
    assert rl.data.length_batching.tokenizer_revision is None
    assert rl.model.model_name_or_path == cfg["sft_export_dir"]
    assert rl.data.train_file == cfg["data_files"]["rl"]
    assert rl.logger.csv and rl.model.lora.r == 8


def test_explicitly_disabled_length_buckets_override_experiment_defaults() -> None:
    cfg = load_config(
        DEFAULT_CONFIG.with_name("nllb_1p3b_correction.yaml"), ["length_batching=false"]
    )
    for stage in ("sft", "rl"):
        assert not compose_stage(cfg, stage, None).data.length_batching.enabled


def test_qwen3_4090_profile_keeps_non_thinking_prompt_and_200_batch_eval_cap() -> None:
    cfg = load_config(
        DEFAULT_CONFIG.with_name("qwen3_1p7b_4090.yaml"), ["wandb.enabled=false"]
    )
    assert cfg["model_name"] == "Qwen/Qwen3-1.7B"
    assert cfg["sac_reward"]["backend"]["concurrency"] == 16
    for stage in ["sft", "rl"]:
        composed = compose_stage(cfg, stage, None)
        assert composed.model.prompt_template == cfg["prompt_template"]
        assert composed.trainer.limit_val_batches == 200
        assert composed.trainer.limit_test_batches == 200
        assert "<think>\n\n</think>" in composed.model.prompt_template


def test_dry_runs_need_no_credentials_data_or_checkpoint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "must-not-exist"
    assert (
        train(["--stage", "pipeline", "--dry-run", "--set", f"output_dir={output}"])
        == 0
    )
    assert not output.exists()
    assert "val_check_interval: 500" in capsys.readouterr().out
    assert (
        evaluate(
            [
                "--checkpoint",
                str(tmp_path / "not-a-checkpoint"),
                "--dry-run",
                "--set",
                f"output_dir={output}",
            ]
        )
        == 0
    )
    assert not output.exists()
    assert json.loads(capsys.readouterr().out)["bidirectional"]


def test_user_managed_mode_has_no_deadline_or_budget_callback(tmp_path: Path) -> None:
    cfg = load_config(
        DEFAULT_CONFIG,
        [
            "budget_control=false",
            "max_hours=null",
            "early_stopping=false",
            "max_epochs.rl=-1",
        ],
    )
    assert deadline(cfg, tmp_path) is None
    cfg["run_group"] = "user-managed"
    composed = compose_stage(cfg, "rl", None)
    assert "rental_guard" not in composed.callbacks
    assert composed.callbacks.training_progress._target_.endswith("TrainingProgress")
    assert composed.callbacks.early_stopping is None
    assert composed.trainer.max_epochs == -1


@pytest.mark.parametrize(
    "override",
    [
        "group_size=1",
        "num_iterations=3",
        "max_hours=-1",
        "reward.comet_weight=-0.1",
        "trainer.devices=2",
        "data_loader.num_workers=-1",
        "data_loader.prefetch_factor=0",
        "WANDB_API_KEY=secret",
    ],
)
def test_bad_config_rejected_before_gpu_work(override: str) -> None:
    with pytest.raises((ValueError, TypeError)):
        load_config(DEFAULT_CONFIG.with_name("small.yaml"), [override])


def test_checkpoint_trust_and_expired_deadline(tmp_path: Path) -> None:
    assert train(["--stage", "sft", "--resume", "some.ckpt"]) == 2
    assert evaluate(["--checkpoint", "some.ckpt"]) == 2
    (tmp_path / "rental_config.json").write_text('{"graceful_deadline_unix":1}')
    with pytest.raises(ValueError, match="deadline"):
        deadline(load_config(DEFAULT_CONFIG.with_name("small.yaml"), []), tmp_path)


@pytest.mark.parametrize("full_dataset", [False, True])
def test_real_pipeline_then_standalone_evaluation(
    tmp_path: Path, full_dataset: bool
) -> None:
    torch.set_num_threads(1)
    assets = create_assets(tmp_path / "model with space", "causal")
    Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=18,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=1,
        )
    ).save_pretrained(assets)
    paths = {}
    for split, count in (("train", 6), ("validation", 2), ("test", 2)):
        path = tmp_path / f"{split}.csv"
        path.write_text(
            "en,vi\n"
            + "\n".join(f"{split} source {i},{split} target {i}" for i in range(count)),
            encoding="utf-8",
        )
        paths[split] = path
    stages = tmp_path / "stages"
    if full_dataset:
        prepare_full(paths["train"], paths["test"], stages, validation_pairs=1)
    else:
        prepare(paths["train"], paths["validation"], paths["test"], stages)
    config = OmegaConf.create(
        load_config(
            DEFAULT_CONFIG.with_name("full.yaml" if full_dataset else "small.yaml"), []
        )
    )
    config.output_dir = str(tmp_path / "run")
    # load_config resolves interpolations before this fixture changes output_dir.
    config.sft_export_dir = str(tmp_path / "run/sft_export")
    if "length_cache_dir" in config:
        config.length_cache_dir = str(tmp_path / "run/length_cache")
    config.model_name = str(assets)
    config.policy_revision = "main"
    config.hf_home = str(tmp_path / "hf")
    config.dataset_manifest = str(stages / "manifest.json")
    config.data_files = {
        "sft": str(stages / "sft.csv"),
        "rl": str(stages / "rl.csv"),
        "validation": str(
            stages / "validation.csv" if full_dataset else paths["validation"]
        ),
        "test": str(paths["test"]),
    }
    config.batch_sizes = {"sft": 2, "rl": 2}
    config.budget_control = False
    config.max_hours = None
    config.data_loader.num_workers = 0
    config.trainer = {
        "accelerator": "cpu",
        "devices": 1,
        "precision": "32-true",
        "limit_train_batches": 1,
        "enable_progress_bar": False,
        "enable_model_summary": False,
    }
    config.load_dtype = "float32"
    config.lora.r = 2
    config.gradient_checkpointing = False
    config.source_cap = 64
    config.target_cap = 3
    config.group_size = 2
    config.reward = {"chrf_weight": 0.7, "bleu_weight": 0.3, "comet_weight": 0.0}
    config.wandb.enabled = False
    config.augmentation.enabled = False
    if not full_dataset:
        config.optimizer = "adamw"
        config.beta = 0.0
        config.handoff_checkpoint = "last"
        config.max_optimizer_steps = {"sft": 2, "rl": 2}
        config.max_epochs = {"sft": -1, "rl": -1}
    if full_dataset:
        config.trainer.max_steps = 5
        config.trainer.limit_train_batches = 4
        config.max_epochs = {"sft": 5, "rl": 5}
        config.kl_control.enabled = True
        config.kl_control.target = (
            0.1  # Synthetic functional test, not a calibrated MT target.
        )
        config.kl_control.update_every = 1
        config.kl_control.quality_floors = {
            "val/en_vi/chrf": 0.0,
            "val/vi_en/chrf": 0.0,
        }
    cfg_path = tmp_path / "config.yaml"
    OmegaConf.save(config, cfg_path)
    assert train(["--config", str(cfg_path)]) == 0
    runtime = json.loads((tmp_path / "run/rental_config.json").read_text())
    assert Path(runtime["sft_export_dir"]).is_dir()
    for stage in ("sft", "rl"):
        record = json.loads((tmp_path / f"run/{stage}/result.json").read_text())
        assert Path(record["checkpoint"]).is_file()
        if not full_dataset:
            assert record["global_step"] == 2
            assert Path(record["checkpoint"]).name == "handoff.ckpt"
    checkpoint = json.loads((tmp_path / "run/rl/result.json").read_text())["checkpoint"]
    if full_dataset:
        for stage in ("sft", "rl"):
            record = json.loads((tmp_path / f"run/{stage}/result.json").read_text())
            assert record["global_step"] == 5
        saved = torch.load(checkpoint, weights_only=False, map_location="cpu")
        assert saved["kl_controller"]["observations"] == 5
        assert 0.0025 <= saved["kl_controller"]["value"] <= 0.04
    assert (
        evaluate(
            [
                "--config",
                str(cfg_path),
                "--checkpoint",
                checkpoint,
                "--trust-checkpoint",
                "--output",
                str(tmp_path / "evaluated"),
            ]
        )
        == 0
    )
    report = json.loads((tmp_path / "evaluated/metrics.json").read_text())
    assert {
        "test/bleu_macro",
        "test/chrf_macro",
        "test/en_vi/bleu",
        "test/vi_en/bleu",
    } <= report["metrics"].keys()
    assert not (tmp_path / "run/.launcher.lock").exists()
