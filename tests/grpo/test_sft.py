"""Supervised masking, adapter preservation, checkpoint handoff, and bilingual RL."""

from functools import partial
from pathlib import Path

import lightning as L
import pytest
import torch

from scripts.verify.assets import create_assets
from src.data.translation_datamodule import TranslationDataModule
from src.models.components.optim import TransferAdamW
from src.models.components.translation_reward import TranslationReward
from src.models.grpo_module import GRPOModule
from src.models.sft_module import TranslationSFTModule, export_sft


@pytest.mark.parametrize("architecture", ["seq2seq", "causal"])
@pytest.mark.parametrize("adapters", [False, True])
def test_supervised_bilingual_checkpoint_export_and_rl_handoff(
    tmp_path: Path, architecture: str, adapters: bool
) -> None:
    torch.set_num_threads(1)
    L.seed_everything(42)
    if adapters:
        pytest.importorskip("peft")
    asset = create_assets(tmp_path / "base", architecture)
    module = TranslationSFTModule(
        model_name_or_path=str(asset),
        max_source_length=48,
        max_new_tokens=4,
        max_target_length=6,
        group_size=2,
        num_iterations=1,
        learning_rate=0.01,
        lora={
            "enabled": True,
            "r": 2,
            "target_modules": ["q", "v"] if architecture == "seq2seq" else ["c_attn"],
        }
        if adapters
        else None,
        optimizer=partial(TransferAdamW, lr=0.01, max_relative_drift=0.2),
        profiling={"enabled": True},
    )
    assert module.reference is None
    before = {
        name: parameter.detach().clone()
        for name, parameter in module.policy.named_parameters()
    }
    frozen = {
        name
        for name, parameter in module.policy.named_parameters()
        if not parameter.requires_grad
    }
    root = Path(__file__).resolve().parents[2]
    data = TranslationDataModule(
        train_file=str(root / "data/example_train.jsonl"),
        val_file=str(root / "data/example_val.jsonl"),
        test_file=str(root / "data/example_test.jsonl"),
        bidirectional=True,
        batch_size=2,
        augmentation={"enabled": True, "concat_probability": 0.0},
    )
    data.setup("fit")
    batch = next(iter(data.train_dataloader()))
    inputs = module.supervised_inputs(batch)
    assert (inputs["labels"] != -100).any()
    assert all(
        module.tokenizer.eos_token_id in row.tolist() for row in inputs["labels"]
    )
    if architecture == "causal":
        assert (inputs["labels"][:, :2] == -100).all()
    options = {
        "accelerator": "cpu",
        "devices": 1,
        "logger": False,
        "enable_checkpointing": False,
        "enable_model_summary": False,
        "enable_progress_bar": False,
        "num_sanity_val_steps": 0,
        "limit_train_batches": 2,
        "limit_val_batches": 1,
        "default_root_dir": tmp_path / "sft",
    }
    trainer = L.Trainer(max_epochs=1, **options)
    trainer.fit(module, datamodule=data)
    assert trainer.global_step == 2
    assert all(
        torch.equal(value, dict(module.policy.named_parameters())[name])
        for name, value in before.items()
        if name in frozen
    )
    assert any(
        not torch.equal(value, dict(module.policy.named_parameters())[name])
        for name, value in before.items()
        if name not in frozen
    )
    assert (
        "val/en_vi/chrf" in trainer.callback_metrics
        and "val/vi_en/chrf" in trainer.callback_metrics
    )
    checkpoint = tmp_path / "sft.ckpt"
    trainer.save_checkpoint(checkpoint)
    restored = TranslationSFTModule.load_from_checkpoint(checkpoint, weights_only=True)
    resumed = L.Trainer(
        max_epochs=2, **{**options, "default_root_dir": tmp_path / "resumed"}
    )
    resumed.fit(restored, datamodule=data, ckpt_path=str(checkpoint))
    assert resumed.global_step == 4
    export = tmp_path / "sft_dense"
    export_sft(checkpoint, export, load_dtype="float32")
    assert (export / "config.json").is_file() and not (
        export / "adapter_config.json"
    ).exists()
    with pytest.raises(FileExistsError):
        export_sft(checkpoint, export)
    calls: list[list[str]] = []

    def comet(
        sources: list[str], predictions: list[str], references: list[str]
    ) -> list[float]:
        calls.append(sources)
        return [0.2 if index % 2 == 0 else 0.9 for index in range(len(predictions))]

    rl = GRPOModule(
        model_name_or_path=str(export),
        max_source_length=48,
        max_new_tokens=4,
        group_size=2,
        num_iterations=1,
        reward=TranslationReward(
            chrf_weight=0, bleu_weight=0.3, comet_weight=0.7, comet_scorer=comet
        ),
        profiling={"enabled": True},
    )
    rollout = rl._rollout(batch)
    expected_sources = [source for source in batch["clean_source"] for _ in range(2)]
    assert calls[-1] == expected_sources
    assert len(rollout["rewards"]) == 4
    rl_trainer = L.Trainer(
        max_epochs=1,
        **{**options, "default_root_dir": tmp_path / "rl", "limit_train_batches": 1}
    )
    rl_trainer.fit(rl, datamodule=data)
    assert rl_trainer.global_step == 1
    phases = rl.phase_timing.summary()["phases"]
    assert {
        "translation/generation",
        "translation/old_policy_forward",
        "translation/reference_forward",
        "translation/current_forward",
        "translation/backward",
        "translation/optimizer",
        "translation/batch_total",
    }.issubset(phases)
    assert phases["translation/generation"]["output_tokens"] > 0


def test_causal_eos_equals_pad_is_a_supervised_target(tmp_path: Path) -> None:
    module = TranslationSFTModule(
        model_name_or_path=str(create_assets(tmp_path / "model", "causal")),
        max_target_length=4,
    )
    module.tokenizer.pad_token = module.tokenizer.eos_token
    inputs = module.supervised_inputs(
        {"source": ["Hello", "Good morning"], "reference": ["Xin", "Xin chào"]}
    )
    assert all(
        (row == module.tokenizer.eos_token_id).sum() == 1 for row in inputs["labels"]
    )
    assert (inputs["labels"] == -100).any()
