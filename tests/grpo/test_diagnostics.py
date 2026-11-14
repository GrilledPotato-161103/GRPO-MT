"""Policy-distribution correctness, actionable reward telemetry and layer selection."""

import json
from copy import deepcopy
from pathlib import Path

import lightning as L
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from scripts.verify.assets import create_assets
from src.data.translation_datamodule import TranslationDataModule
from src.models.components.adapters import validate_lora
from src.models.components.grpo import completion_mask
from src.models.components.reward_diagnostics import reward_diagnostics
from src.models.grpo_module import GRPOModule


@pytest.mark.parametrize("lora", [False, True])
def test_model_generation_defaults_cannot_change_rollout_likelihoods(
    tmp_path: Path, lora: bool
) -> None:
    torch.set_num_threads(1)
    asset = create_assets(tmp_path / "tiny", "causal")
    model = GRPOModule(
        model_name_or_path=str(asset),
        temperature=1.0,
        max_new_tokens=5,
        lora={"enabled": True, "target_modules": ["c_attn"], "r": 2} if lora else None,
    )
    base = model.policy.get_base_model() if lora else model.policy
    base.generation_config.do_sample = True
    base.generation_config.temperature = 0.7
    base.generation_config.top_p = 0.8
    base.generation_config.repetition_penalty = 1.05
    # Reproduce the reported Transformers >=4.50 default-merge behavior.
    legacy, _ = base._prepare_generation_config(model.generation_config)
    assert legacy.temperature == 0.7 and legacy.top_p == 0.8
    model.policy.eval()
    inputs = model._encode(["Hello", "Good morning"])
    with torch.no_grad():
        sampled = model._generate_policy(
            inputs,
            model.generation_config,
            return_dict_in_generate=True,
            output_scores=True,
        )
        expected = base.compute_transition_scores(
            sampled.sequences, sampled.scores, normalize_logits=True
        )
        prefix = inputs["input_ids"].shape[1]
        mask = completion_mask(
            sampled.sequences[:, prefix:], model.generation_config.eos_token_id
        )
        actual = model._logps(model.policy, inputs, sampled.sequences, mask)
    assert torch.allclose(actual[mask], expected[mask], atol=2e-5)
    greedy = deepcopy(model.generation_config)
    greedy.do_sample, greedy.temperature, greedy.top_k = False, 1.0, 50
    expected = model._generate_policy(inputs, greedy)
    assert model.translate(["Hello", "Good morning"]) == model.tokenizer.batch_decode(
        expected[:, prefix:], skip_special_tokens=True
    )


def test_reward_conflicts_are_within_source_and_exclude_ties() -> None:
    components = {"bleu": [0.0, 0.5, 0.2, 0.2], "comet": [0.9, 0.8, 0.7, 0.7]}
    metrics = reward_diagnostics(
        torch.tensor([0.63, 0.71, 0.55, 0.55]),
        components,
        2,
        ["a", "b", "same", "same"],
        ["en_vi", "vi_en"],
        {"bleu": 0.3, "comet": 0.7},
    )
    assert metrics["reward_conflict/bleu_comet_comparable_pairs"] == 1
    assert metrics["reward_conflict/bleu_comet_disagreement"] == 1
    assert metrics[
        "reward_conflict/bleu_comet_within_source_correlation"
    ] == pytest.approx(-1)
    assert metrics["rollout/unique_candidate_fraction"] == 0.75
    assert metrics["rollout/zero_variance_groups"] == 0.5
    tied = reward_diagnostics(
        torch.ones(2), {"comet": [0.8, 0.8], "bleu": [0, 0]}, 2, ["x", "x"]
    )
    assert "reward_conflict/comet_bleu_disagreement" not in tied
    assert "reward_conflict/comet_bleu_within_source_correlation" not in tied


def test_each_optimizer_update_and_evaluation_is_retained(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    L.seed_everything(42)
    asset = create_assets(tmp_path / "tiny")
    module = GRPOModule(
        model_name_or_path=str(asset),
        max_new_tokens=4,
        group_size=2,
        num_iterations=2,
        reward=lambda p, r: [float(i % 2) for i in range(len(p))],
    )
    data = TranslationDataModule(
        train_file="data/example_train.jsonl", val_file="data/example_val.jsonl"
    )
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=2,
        limit_train_batches=1,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path / "run",
    )
    trainer.fit(module, datamodule=data)
    updates = [
        json.loads(line)
        for line in (tmp_path / "run/diagnostics/updates.jsonl")
        .read_text()
        .splitlines()
    ]
    assert [row["global_step"] for row in updates] == [1, 2, 3, 4]
    assert all("optimizer/gradient_norm" in row["metrics"] for row in updates)
    assert updates[0]["metrics"]["rollout/id"] == updates[1]["metrics"]["rollout/id"]
    evaluations = [
        json.loads(line)
        for line in (tmp_path / "run/translation_metrics/evaluation_history.jsonl")
        .read_text()
        .splitlines()
    ]
    assert [row["global_step"] for row in evaluations] == [2, 4]
    assert all("val/bleu" in row["metrics"] for row in evaluations)


def test_only_selected_qwen_layer_adapters_train_and_resume(tmp_path: Path) -> None:
    pytest.importorskip("peft")
    asset = create_assets(tmp_path / "tiny", "causal")
    Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=18,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=1,
        )
    ).save_pretrained(asset)
    module = GRPOModule(
        model_name_or_path=str(asset),
        lora={
            "enabled": True,
            "r": 2,
            "target_modules": ["q_proj", "v_proj"],
            "layers_to_transform": [2],
            "layers_pattern": "layers",
        },
    )
    trainable = [
        name for name, p in module.policy.named_parameters() if p.requires_grad
    ]
    assert trainable and all(
        "layers.2." in name and "lora_" in name for name in trainable
    )
    assert all(
        not p.requires_grad
        for name, p in module.policy.named_parameters()
        if "lora_" not in name
    )
    module.hparams["max_new_tokens"] = 3
    module.generation_config.max_new_tokens = 3
    module.hparams["num_iterations"] = 1
    module.reward_fn = lambda p, r: [float(i % 2) for i in range(len(p))]
    frozen = {
        name: p.detach().clone()
        for name, p in module.policy.named_parameters()
        if not p.requires_grad
    }
    data = TranslationDataModule(
        train_file="data/example_train.jsonl", val_file="data/example_val.jsonl"
    )
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=1,
        max_epochs=1,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path / "selected",
    )
    trainer.fit(module, datamodule=data)
    assert all(
        torch.equal(value, dict(module.policy.named_parameters())[name])
        for name, value in frozen.items()
    )
    checkpoint = tmp_path / "selected.ckpt"
    trainer.save_checkpoint(checkpoint)
    restored = GRPOModule.load_from_checkpoint(checkpoint, weights_only=False)
    assert [
        name for name, p in restored.policy.named_parameters() if p.requires_grad
    ] == trainable
    assert all(
        torch.equal(value, restored.policy.state_dict()[name])
        for name, value in module.policy.state_dict().items()
    )
    for bad in ([True], [-1], [], [2, 2], "last"):
        with pytest.raises(ValueError):
            validate_lora(
                {
                    "enabled": True,
                    "target_modules": ["q_proj"],
                    "layers_to_transform": bad,
                }
            )


def test_relay_keeps_every_update_and_evaluation_curve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys
    from types import SimpleNamespace

    from scripts.rental.wandb_relay import WandbRelay

    monkeypatch.setenv("WANDB_API_KEY", "unit-local-key")
    records = []
    run = SimpleNamespace(
        url="https://wandb.ai/test/run",
        define_metric=lambda *a, **k: None,
        log=records.append,
        finish=lambda: None,
        summary={},
    )
    monkeypatch.setitem(
        sys.modules, "wandb", SimpleNamespace(init=lambda **kwargs: run)
    )
    root = tmp_path / "checkpoints_and_logs/rl"
    (root / "diagnostics").mkdir(parents=True)
    (root / "translation_metrics").mkdir()
    (root / "resolved_config.yaml").write_text("model: {learning_rate: 0.000005}")
    (root / "diagnostics/updates.jsonl").write_text(
        "".join(
            json.dumps({"global_step": i, "metrics": {"grpo/loss": 0.1 / i}}) + "\n"
            for i in (1, 2)
        )
    )
    (root / "translation_metrics/evaluation_history.jsonl").write_text(
        json.dumps({"global_step": 2, "metrics": {"val/bleu_macro": 35}}) + "\n"
    )
    relay = WandbRelay(tmp_path, "test-group", "test-entity")
    relay.sync()
    relay.sync()
    assert len(records) == 3
    assert [row["trainer/global_step"] for row in records] == [1, 2, 2]
    assert records[-1]["val/bleu_macro"] == 35


def test_diagnostic_experiments_compose() -> None:
    from hydra import compose, initialize

    with initialize(version_base="1.3", config_path="../../configs"):
        cfg = compose(config_name="grpo", overrides=["experiment=phomt_rl_top4"])
        assert cfg.model.warmup_steps == 8
        assert cfg.model.group_size == 4
        assert cfg.model.lora.layers_to_transform == [24, 25, 26, 27]
        assert cfg.trainer.val_check_interval == 10
