"""Adapter-only gradients, frozen shared references, warmup, resume, and export."""

from pathlib import Path
from typing import Any

import lightning as L
import pytest
import torch
from hydra import compose, initialize
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    Qwen2Config,
    Qwen2ForCausalLM,
)

from scripts.verify.assets import create_assets
from src.data.translation_datamodule import TranslationDataModule
from src.models.components.adapters import quantized_load_options, validate_lora
from src.models.grpo_module import GRPOModule

peft = pytest.importorskip("peft")


def rewards(predictions: list[str], references: list[str]) -> list[float]:
    return [float(i % 2) for i in range(len(predictions))]


@pytest.mark.parametrize("architecture", ["seq2seq", "causal", "qwen"])
def test_lora_reference_gradients_resume_and_adapter_export(
    tmp_path: Path, architecture: str
) -> None:
    torch.set_num_threads(1)
    L.seed_everything(42)
    asset = create_assets(
        tmp_path / "model", "causal" if architecture == "qwen" else architecture
    )
    if architecture == "qwen":
        tiny = Qwen2ForCausalLM(
            Qwen2Config(
                vocab_size=18,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=256,
                bos_token_id=1,
                eos_token_id=1,
                pad_token_id=0,
            )
        )
        tiny.save_pretrained(asset)
    targets = {
        "seq2seq": ["q", "v"],
        "causal": ["c_attn"],
        "qwen": ["q_proj", "v_proj"],
    }[architecture]
    module = GRPOModule(
        model_name_or_path=str(asset),
        group_size=2,
        max_new_tokens=4,
        num_iterations=2,
        reward=rewards,
        learning_rate=1e-2,
        lora={
            "enabled": True,
            "r": 2,
            "alpha": 4,
            "target_modules": targets,
            "warmup_steps": 4,
        },
        gradient_checkpointing=True,
        logprob_batch_size=1,
    )
    assert module.reference is None
    assert all(
        "lora_" in name
        for name, p in module.policy.named_parameters()
        if p.requires_grad
    )
    frozen = {
        name: p.detach().clone()
        for name, p in module.policy.named_parameters()
        if not p.requires_grad
    }
    adapters = {
        name: p.detach().clone()
        for name, p in module.policy.named_parameters()
        if p.requires_grad
    }
    sample = module._sample_prompts(["Hello", "Good morning"], 1)
    ref_before = module._reference_logps(
        sample["inputs"], sample["sequences"], sample["mask"]
    ).clone()
    assert torch.allclose(sample["old_logps"], ref_before, atol=1e-5)
    module._prepare_gradient_scoring()
    assert module.policy.training and module.policy.is_gradient_checkpointing
    with torch.no_grad():
        train_mode_logps = module._logps(
            module.policy, sample["inputs"], sample["sequences"], sample["mask"]
        )
    assert torch.allclose(train_mode_logps, ref_before, atol=1e-5)
    root = Path(__file__).resolve().parents[2]
    data = TranslationDataModule(
        train_file=str(root / "data/example_train.jsonl"),
        val_file=str(root / "data/example_val.jsonl"),
        test_file=str(root / "data/example_test.jsonl"),
    )
    options = {
        "accelerator": "cpu",
        "devices": 1,
        "logger": False,
        "enable_checkpointing": False,
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "num_sanity_val_steps": 0,
        "limit_train_batches": 1,
        "limit_val_batches": 1,
        "default_root_dir": tmp_path,
    }
    trainer = L.Trainer(max_epochs=1, **options)
    trainer.fit(module, datamodule=data)
    assert trainer.global_step == 2
    assert trainer.optimizers[0].param_groups[0]["lr"] == pytest.approx(0.005)
    assert all(
        torch.equal(value, dict(module.policy.named_parameters())[name])
        for name, value in frozen.items()
    )
    assert any(
        not torch.equal(value, dict(module.policy.named_parameters())[name])
        for name, value in adapters.items()
    )
    module.policy.eval()
    assert torch.allclose(
        module._reference_logps(sample["inputs"], sample["sequences"], sample["mask"]),
        ref_before,
        atol=1e-5,
    )
    checkpoint = tmp_path / "lora.ckpt"
    trainer.save_checkpoint(checkpoint)
    saved = torch.load(checkpoint, weights_only=False)
    assert not any(name.startswith("reference.") for name in saved["state_dict"])
    assert saved["adapter_checkpoint"]["model"] == str(asset)
    assert all("lora_" in name for name in saved["state_dict"])
    restored = GRPOModule.load_from_checkpoint(
        checkpoint, reward=rewards, weights_only=False
    )
    resumed = L.Trainer(max_epochs=2, **options)
    resumed.fit(restored, datamodule=data, ckpt_path=str(checkpoint))
    assert resumed.global_step == 4
    assert resumed.optimizers[0].param_groups[0]["lr"] == pytest.approx(0.01)
    assert torch.allclose(
        restored._reference_logps(
            sample["inputs"], sample["sequences"], sample["mask"]
        ),
        ref_before,
        atol=1e-5,
    )
    export = tmp_path / "adapter_export"
    restored.export_pretrained(export)
    assert (export / "adapter_config.json").exists()
    factory: Any = (
        AutoModelForSeq2SeqLM if architecture == "seq2seq" else AutoModelForCausalLM
    )
    reloaded = peft.PeftModel.from_pretrained(
        factory.from_pretrained(asset), export
    ).eval()
    restored.policy.eval()
    assert torch.allclose(
        restored._logps(
            restored.policy, sample["inputs"], sample["sequences"], sample["mask"]
        ),
        restored._logps(
            reloaded, sample["inputs"], sample["sequences"], sample["mask"]
        ),
        atol=1e-5,
    )


@pytest.mark.parametrize(
    "settings",
    [
        {"r": 0},
        {"alpha": 0},
        {"dropout": 1},
        {"warmup_steps": -1},
        {"bias": "all"},
        {"modules_to_save": ["lm_head"]},
        {"target_modules": []},
    ],
)
def test_invalid_lora_options_fail_early(settings: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        validate_lora({"enabled": True, **settings})


def test_quantization_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    assert quantized_load_options(None, "float32", False) == {}
    with pytest.raises(ValueError):
        quantized_load_options("nf4", "bfloat16", False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA"):
        quantized_load_options("nf4", "bfloat16", True)


def test_adapter_experiment_configs() -> None:
    with initialize(version_base="1.3", config_path="../../configs"):
        for config in ("grpo", "self_play"):
            lora = compose(config_name=config, overrides=["experiment=qwen_7b_lora"])
            assert lora.model.lora.enabled and lora.model.load_dtype == "bfloat16"
            assert (
                lora.model.gradient_checkpointing and lora.model.logprob_batch_size == 1
            )
            qlora = compose(config_name=config, overrides=["experiment=qwen_7b_qlora"])
            assert qlora.model.quantization == "nf4" and qlora.model.lora.enabled
            if config == "self_play":
                assert qlora.model.proposal_max_new_tokens == 64


def test_nf4_qwen_cuda_training_and_resume(tmp_path: Path) -> None:
    pytest.importorskip("bitsandbytes")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.set_num_threads(1)
    L.seed_everything(42)
    asset = create_assets(tmp_path / "qwen", "causal")
    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=18,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            bos_token_id=1,
            eos_token_id=1,
            pad_token_id=0,
        )
    )
    model.save_pretrained(asset)
    module = GRPOModule(
        model_name_or_path=str(asset),
        load_dtype="bfloat16",
        quantization="nf4",
        lora={"enabled": True, "r": 2, "target_modules": ["q_proj", "v_proj"]},
        gradient_checkpointing=True,
        logprob_batch_size=1,
        max_new_tokens=4,
        group_size=2,
        reward=rewards,
        num_iterations=1,
    )
    assert module.policy.is_loaded_in_4bit and module.reference is None
    root = Path(__file__).resolve().parents[2]
    data = TranslationDataModule(
        train_file=str(root / "data/example_train.jsonl"),
        val_file=str(root / "data/example_val.jsonl"),
        test_file=str(root / "data/example_test.jsonl"),
    )
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        enable_checkpointing=False,
        logger=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        default_root_dir=tmp_path,
    )
    trainer.fit(module, datamodule=data)
    assert trainer.global_step == 1
    path = tmp_path / "quantized.ckpt"
    trainer.save_checkpoint(path)
    restored = GRPOModule.load_from_checkpoint(path, reward=rewards, weights_only=False)
    assert restored.policy.is_loaded_in_4bit and restored.reference is None
    assert all(
        torch.equal(value.cpu(), restored.policy.state_dict()[name].cpu())
        for name, value in module.policy.state_dict().items()
        if "lora_" in name
    )
    resumed = L.Trainer(
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        max_epochs=2,
        limit_train_batches=1,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        enable_checkpointing=False,
        logger=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        default_root_dir=tmp_path,
    )
    resumed.fit(restored, datamodule=data, ckpt_path=str(path))
    assert resumed.global_step == 2
