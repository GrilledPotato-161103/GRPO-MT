"""COMET-first anchor selection, source verification, and FP16 execution."""

import pytest
import torch

from tests.grpo.test_correction_objective import make_module, tiny_nllb


@pytest.mark.parametrize("verified", [True, False])
def test_comet_winner_requires_source_verification_without_reference_fallback(
    tmp_path, monkeypatch, verified
):
    from src.models.anchored_grpo_module import NoUsableFeedback
    from src.models.components.correction_objective import quality_scores
    from src.models.grpo_module import GRPOModule

    module = make_module(
        tmp_path,
        correction={
            "noise_aware": True,
            "anchor_selection": "best_comet",
            "allow_reference_fallback": False,
        },
    )
    batch = {
        "source": ["Hello world", "Xin"],
        "reference": ["Xin", "Hello world"],
        "source_lang": ["English", "Vietnamese"],
        "target_lang": ["Vietnamese", "English"],
    }
    rollout = GRPOModule._rollout(module, batch)
    rollout["capped"][:] = False
    rollout["predictions"] = ["Xin", "world", "Hello", "world"]
    c = rollout["reward_components"]
    c["comet"] = [0.9, 0.8, 0.9, 0.8]
    c["sac"] = [1, 0.7, 1, 0.7] if verified else [0.7, 1, 0.7, 1]
    c["reward"] = quality_scores(c["comet"], c["sac"])
    rollout["rewards"] = torch.tensor(c["reward"])
    monkeypatch.setattr(GRPOModule, "_rollout", lambda self, batch: rollout)
    monkeypatch.setattr(
        module,
        "_reward_components",
        lambda *a, **kw: pytest.fail("Reference fallback was called"),
    )
    if verified:
        result = module._rollout(batch)
        assert result["correction_routes"]["reference_anchors"] == 0
        assert len(result["correction_groups"]) == 2
        assert all(g.provenance == "candidate" for g in result["correction_groups"])
        loss, _ = module._backward_rollout(
            result, backward_fn=lambda loss: loss.backward()
        )
        assert torch.isfinite(loss)
    else:
        with pytest.raises(NoUsableFeedback):
            module._rollout(batch)


def test_invalid_comet_selection_options():
    from src.models.components.correction_objective import CorrectionOptions

    with pytest.raises(ValueError):
        CorrectionOptions(anchor_selection="random")
    with pytest.raises(TypeError):
        CorrectionOptions(allow_reference_fallback="false")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FP16 requires CUDA")
def test_native_fp16_lora_optimizer_updates(tmp_path):
    import lightning as L
    from lightning.pytorch.plugins.precision import MixedPrecision
    from torch.utils.data import DataLoader

    from src.models.sft_module import TranslationSFTModule

    model = TranslationSFTModule(
        model_name_or_path=str(tiny_nllb(tmp_path / "native")),
        language_codes={"English": "eng_Latn", "Vietnamese": "vie_Latn"},
        prompt_template="{source}",
        beta=0,
        load_dtype="float16",
        lora={
            "enabled": True,
            "r": 2,
            "alpha": 4,
            "dropout": 0,
            "target_modules": "all-linear",
        },
        max_target_length=8,
        max_new_tokens=7,
        learning_rate=1e-3,
        diagnostics={"enabled": False},
    )
    batch = {
        "source": ["Hello world", "Xin"],
        "reference": ["Xin", "Hello world"],
        "source_lang": ["English", "Vietnamese"],
        "target_lang": ["Vietnamese", "English"],
    }
    before = {
        n: p.detach().clone()
        for n, p in model.policy.named_parameters()
        if p.requires_grad
    }
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        max_steps=2,
        max_epochs=1,
        plugins=[
            MixedPrecision(
                "16-mixed", "cuda", scaler=torch.amp.GradScaler("cuda", init_scale=1024)
            )
        ],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        limit_val_batches=0,
    )
    trainer.fit(model, train_dataloaders=DataLoader([batch, batch], batch_size=None))
    assert trainer.global_step == 2
    assert any(
        not torch.equal(p.detach().cpu(), before[n])
        for n, p in model.policy.named_parameters()
        if n in before
    )
    assert all(torch.isfinite(p).all() for p in model.policy.parameters())
