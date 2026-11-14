"""End-to-end GRPO tests with actual tiny Transformers models and Lightning."""

from pathlib import Path

import lightning as L
import pytest
import torch
from hydra import compose, initialize

from scripts.verify.assets import create_assets
from src.data.translation_datamodule import TranslationDataModule
from src.models.grpo_module import GRPOModule


@pytest.fixture(autouse=True)
def reproducible() -> None:
    torch.set_num_threads(1)
    L.seed_everything(42)


def indexed_reward(predictions: list[str], references: list[str]) -> list[float]:
    """Guarantee nonconstant group rewards to test parameter updates."""
    return [float(i % 2) for i in range(len(predictions))]


@pytest.mark.parametrize("architecture", ["seq2seq", "causal"])
def test_train_checkpoint_resume_and_export(tmp_path: Path, architecture: str) -> None:
    asset = create_assets(tmp_path / "model", architecture)
    model = GRPOModule(
        model_name_or_path=str(asset),
        group_size=2,
        num_iterations=2,
        max_new_tokens=4,
        learning_rate=1e-3,
        reward=indexed_reward,
    )
    before = {k: v.clone() for k, v in model.policy.state_dict().items()}
    reference = {k: v.clone() for k, v in model.reference.state_dict().items()}
    root = Path(__file__).resolve().parents[2]
    data = TranslationDataModule(
        train_file=str(root / "data/example_train.jsonl"),
        val_file=str(root / "data/example_val.jsonl"),
        test_file=str(root / "data/example_test.jsonl"),
    )
    trainer = L.Trainer(
        default_root_dir=tmp_path,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        limit_test_batches=1,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        accelerator="cpu",
        devices=1,
    )
    trainer.fit(model, datamodule=data)
    assert trainer.global_step == 2
    assert any(
        not torch.equal(before[k], v) for k, v in model.policy.state_dict().items()
    )
    assert all(
        torch.equal(reference[k], v) for k, v in model.reference.state_dict().items()
    )
    assert all(p.grad is None for p in model.reference.parameters())
    assert torch.isfinite(trainer.callback_metrics["train/loss"])
    assert "val/chrf" in trainer.callback_metrics
    trainer.test(model, datamodule=data)
    assert "test/bleu" in trainer.callback_metrics
    checkpoint = tmp_path / "resume.ckpt"
    trainer.save_checkpoint(checkpoint)
    restored = GRPOModule.load_from_checkpoint(
        checkpoint, reward=indexed_reward, weights_only=False
    )
    assert all(
        torch.equal(reference[k], v) for k, v in restored.reference.state_dict().items()
    )
    resumed = L.Trainer(
        max_epochs=2,
        limit_train_batches=1,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="cpu",
        devices=1,
    )
    resumed.fit(restored, datamodule=data, ckpt_path=str(checkpoint))
    assert resumed.global_step == 4
    assert all(
        torch.equal(reference[k], v) for k, v in restored.reference.state_dict().items()
    )
    restored.export_pretrained(tmp_path / "export")
    inference = GRPOModule(
        model_name_or_path=str(tmp_path / "export"), max_new_tokens=3
    )
    assert len(inference.translate(["Hello", "Good morning"])) == 2


@pytest.mark.parametrize("architecture", ["seq2seq", "causal"])
def test_likelihood_matches_generation_distribution(
    tmp_path: Path, architecture: str
) -> None:
    asset = create_assets(tmp_path / "model", architecture)
    model = GRPOModule(model_name_or_path=str(asset), temperature=0.7, max_new_tokens=4)
    model.policy.eval()
    inputs = model._encode(["Hello", "Good morning"])
    with torch.no_grad():
        output = model.policy.generate(
            **inputs,
            generation_config=model.generation_config,
            return_dict_in_generate=True,
            output_scores=True
        )
        expected = model.policy.compute_transition_scores(
            output.sequences, output.scores, normalize_logits=True
        )
        prefix = 1 if model.is_encoder_decoder else inputs["input_ids"].shape[1]
        from src.models.components.grpo import completion_mask

        mask = completion_mask(
            output.sequences[:, prefix:], model.generation_config.eos_token_id
        )
        actual = model._logps(model.policy, inputs, output.sequences, mask)
    assert torch.allclose(actual[mask], expected[mask], atol=2e-5)


def test_configurations_compose() -> None:
    with initialize(version_base="1.3", config_path="../../configs"):
        for overrides in ([], ["experiment=phomt"], ["experiment=causal"]):
            cfg = compose(config_name="grpo", overrides=overrides)
            assert cfg.model.group_size >= 2
            assert cfg.callbacks.model_checkpoint.monitor == "val/chrf"
        evaluation = compose(config_name="grpo_eval")
        assert evaluation.model._target_.endswith("GRPOModule")


def test_forced_language_token_keeps_decoder_context(tmp_path: Path) -> None:
    from src.models.components.grpo import completion_mask

    asset = create_assets(tmp_path / "model")
    model = GRPOModule(
        model_name_or_path=str(asset), forced_bos_token_id=13, max_new_tokens=4
    )
    model.policy.eval()
    inputs = model._encode(["Hello", "Good morning"])
    with torch.no_grad():
        output = model.policy.generate(
            **inputs,
            generation_config=model.generation_config,
            return_dict_in_generate=True,
            output_scores=True
        )
        assert (output.sequences[:, 1] == 13).all()
        expected = model.policy.compute_transition_scores(
            output.sequences, output.scores, normalize_logits=True
        )
        mask = completion_mask(
            output.sequences[:, 1:], model.generation_config.eos_token_id
        )
        mask[:, 0] = False
        actual = model._logps(model.policy, inputs, output.sequences, mask)
    assert torch.allclose(actual[mask], expected[mask], atol=2e-5)
    rollout = model._rollout({"source": ["Hello"], "reference": ["Xin chào"]})
    assert not rollout["mask"][:, 0].any()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"group_size": 1},
        {"num_iterations": 0},
        {"temperature": 0},
        {"beta": -1},
        {"clip_epsilon": 2},
        {"max_new_tokens": 0},
    ],
)
def test_invalid_hyperparameters_fail_before_loading(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        GRPOModule(**kwargs)
