"""SAC transport, reward polarity, and real LoRA training without paid API calls."""

import csv
import json
from pathlib import Path
from typing import Any

import httpx
import lightning as L
import pytest
import torch

from scripts.main.common import ROOT, check_feedback, compose_stage, load_config
from scripts.verify.assets import create_assets
from src.data.translation_datamodule import TranslationDataModule
from src.feedback.backends import APIChatBackend, FeedbackError
from src.feedback.sac import SAC_SCORES, SACScorer, parse_sac
from src.models.components.grpo import grpo_loss
from src.models.components.translation_reward import TranslationReward
from src.models.grpo_module import GRPOModule


@pytest.mark.parametrize("label,score", list(SAC_SCORES.items()))
def test_categories(label: str, score: float) -> None:
    assert SAC_SCORES[parse_sac(f" [{label.upper()}] ")] == score


@pytest.mark.parametrize(
    "text",
    [
        "",
        "0.9",
        '{"label":"Full match"}',
        "Full match because it is correct",
        "unknown",
    ],
)
def test_invalid_feedback_does_not_become_reward(text: str) -> None:
    with pytest.raises(FeedbackError):
        parse_sac(text)


def test_openrouter_route_order_cache_and_no_unsupported_temperature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CLAUDE_AUTH_KEY", "mock-key")
    calls = []

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        assert body["model"] == "openai/gpt-6-luna"
        assert str(request.url) == "https://openrouter.ai/api/v1/chat/completions"
        assert "temperature" not in body and "response_format" not in body
        payload = json.loads(body["messages"][-1]["content"])
        label = "Full match" if payload["candidate"] == "good" else "Low Partial match"
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": label}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 3},
            },
        )

    api = APIChatBackend(
        model="openai/gpt-6-luna",
        temperature=None,
        json_mode=False,
        transport=httpx.MockTransport(respond),
    )
    scorer = SACScorer(evaluator=api, cache_dir=str(tmp_path))
    args = (
        ["source"] * 4,
        ["good", "bad", "good", ""],
        ["English"] * 4,
        ["Vietnamese"] * 4,
    )
    assert scorer.evaluate(*args) == [1.0, 0.3, 1.0, 0.1]
    assert len(calls) == 2 and scorer.last_metrics["api_prompt_tokens"] == 24
    assert scorer.evaluate(*args) == [1.0, 0.3, 1.0, 0.1]
    assert len(calls) == 2 and scorer.last_metrics["api_requests"] == 0
    cached = SACScorer(evaluator=api, cache_dir=str(tmp_path), cache_only=True)
    assert cached.evaluate(*args) == [1.0, 0.3, 1.0, 0.1]
    with pytest.raises(FeedbackError, match="cache miss"):
        cached.evaluate(["source"], ["new"], ["English"], ["Vietnamese"])
    # Direction changes invalidate the cache, including same-language reconstruction.
    assert scorer.evaluate(["source"], ["good"], ["English"], ["English"]) == [1.0]
    assert len(calls) == 3
    other_model = SACScorer(
        backend={"model": "other/model"}, cache_dir=str(tmp_path), cache_only=True
    )
    with pytest.raises(FeedbackError, match="cache miss"):
        other_model.evaluate(["source"], ["good"], ["English"], ["Vietnamese"])


class Labels:
    """In-memory fixed judge recording the actual language-tagged comparisons."""

    def __init__(self, invalid: bool = False) -> None:
        self.payloads: list[dict[str, str]] = []
        self.invalid = invalid

    async def generate(self, conversations: list[Any]) -> list[str]:
        self.payloads.extend(
            json.loads(messages[-1]["content"]) for messages in conversations
        )
        return ["bad response" if self.invalid else "Full match" for _ in conversations]


def test_malformed_response_is_not_cached(tmp_path: Path) -> None:
    scorer = SACScorer(evaluator=Labels(invalid=True), cache_dir=str(tmp_path))
    with pytest.raises(FeedbackError):
        scorer.evaluate(["source"], ["candidate"], ["English"], ["Vietnamese"])
    assert not list(tmp_path.glob("*.json"))


def test_same_scorer_round_trip_penalty_polarity_and_no_reference_dependency() -> None:
    class FixedSAC:
        def evaluate(
            self,
            sources: list[str],
            predictions: list[str],
            src: list[str],
            tgt: list[str],
        ) -> list[float]:
            assert src == ["English", "Vietnamese"] * 2
            assert tgt == ["Vietnamese", "English", "English", "Vietnamese"]
            assert predictions == [
                "candidate",
                "candidate",
                "reconstruction",
                "reconstruction",
            ]
            return [1.0, 1.0, 1.0, 0.3]

    reward = TranslationReward(
        chrf_weight=0,
        comet_weight=1,
        comet_scorer=lambda *args: [0.8, 0.8],
        sac={"enabled": True, "round_trip_weight": 0.2},
        sac_scorer=FixedSAC(),
    )
    kwargs = {
        "sources": ["original"] * 2,
        "source_langs": ["English", "Vietnamese"],
        "target_langs": ["Vietnamese", "English"],
        "reconstructions": ["reconstruction"] * 2,
    }
    result = reward.components(["candidate"] * 2, ["unrelated reference"] * 2, **kwargs)
    assert result["round_trip_penalty"] == pytest.approx([0.0, 0.14])
    assert result["reward"] == pytest.approx([0.8, 0.66])
    assert (
        reward.components(["candidate"] * 2, ["candidate"] * 2, **kwargs)["reward"]
        == result["reward"]
    )
    with pytest.raises(ValueError, match="reconstruction"):
        reward.components(
            ["candidate"] * 2,
            ["ref"] * 2,
            sources=["source"] * 2,
            source_langs=["English"] * 2,
            target_langs=["Vietnamese"] * 2,
        )


def test_no_kl_avoids_overflow_and_keeps_policy_gradient() -> None:
    current = torch.zeros(2, 2, requires_grad=True)
    loss, metrics = grpo_loss(
        current,
        current.detach(),
        torch.full_like(current, 10000),
        torch.tensor([1.0, -1.0]),
        torch.ones_like(current, dtype=torch.bool),
        beta=0,
    )
    loss.backward()
    assert torch.isfinite(loss) and metrics["kl"] == 0
    assert current.grad is not None and current.grad.norm() > 0


def test_direct_sac_gates_comet_without_round_trip() -> None:
    class FourTiers:
        def evaluate(
            self,
            sources: list[str],
            candidates: list[str],
            src: list[str],
            tgt: list[str],
        ) -> list[float]:
            assert (
                len(candidates) == 4
            )  # No reconstruction comparisons in this ablation.
            return [1.0, 0.7, 0.3, 0.1]

    reward = TranslationReward(
        chrf_weight=0,
        comet_weight=1,
        comet_scorer=lambda *args: [1.4] * 4,
        sac={"enabled": True, "round_trip_weight": 0},
        sac_scorer=FourTiers(),
    )
    values = reward.components(
        ["same candidate"] * 4,
        ["same candidate"] * 4,
        sources=["source"] * 4,
        source_langs=["English"] * 4,
        target_langs=["Vietnamese"] * 4,
    )
    assert values["comet"] == [1.4] * 4  # Keep raw QE scores for diagnostics.
    assert values["reward"] == pytest.approx([1.0, 0.7, 0.3, 0.1])
    assert "round_trip_penalty" not in values


def test_new_profile_and_feedback_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = load_config(ROOT / "configs/runs/sac.yaml", ["wandb.enabled=false"])
    sft, rl = (compose_stage(cfg, stage, None) for stage in ("sft", "rl"))
    assert sft.trainer.max_steps == 2000 and rl.trainer.max_steps == 10000
    assert sft.trainer.max_epochs == rl.trainer.max_epochs == -1
    assert rl.model.beta == 0 and not rl.model.kl_control.enabled
    assert rl.model.optimizer._target_ == "torch.optim.AdamW"
    assert "max_relative_drift" not in rl.model.optimizer
    assert rl.model.reward.bleu_weight == 0 and rl.model.reward.sac.enabled
    assert rl.model.reward.sac.backend.model == "openai/gpt-6-luna"
    assert not rl.model.reward.harness.enabled
    assert sft.model.reward.sac.enabled and cfg["handoff_checkpoint"] == "last"
    monkeypatch.setattr("src.utils.credentials.environment_secret", lambda name: None)
    with pytest.raises(ValueError, match="ANTHROPIC_AUTH_TOKEN"):
        check_feedback(cfg)


def test_real_lora_grpo_both_directions_and_sac_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.set_num_threads(1)
    L.seed_everything(42)
    assets = create_assets(tmp_path / "base", "causal")
    judge = Labels()
    reward = TranslationReward(
        chrf_weight=0,
        comet_weight=1,
        comet_scorer=lambda s, p, r: [0.4 + 0.2 * (i % 2) for i in range(len(p))],
        sac={
            "enabled": True,
            "round_trip_batch_size": 2,
            "round_trip_max_new_tokens": 4,
        },
        sac_scorer=SACScorer(evaluator=judge),
    )
    model = GRPOModule(
        model_name_or_path=str(assets),
        reward=reward,
        beta=0,
        max_source_length=40,
        max_new_tokens=4,
        group_size=2,
        num_iterations=2,
        learning_rate=1e-3,
        lora={"enabled": True, "r": 2, "alpha": 4, "target_modules": ["c_attn"]},
        save_predictions=True,
        optimizer={"_target_": "torch.optim.AdamW", "_partial_": True},
    )
    assert model.reference is None
    before = {
        name: parameter.clone() for name, parameter in model.policy.named_parameters()
    }

    def no_reference(*args: Any) -> Any:
        raise AssertionError("Reference inference must not run with beta=0")

    monkeypatch.setattr(model, "_reference_logps", no_reference)
    reverse_calls = []
    translate = model.translate

    def recorded_translate(sources: list[str], **kwargs: Any) -> list[str]:
        reverse_calls.append(kwargs)
        result = translate(sources, **kwargs)
        assert all(parameter.grad is None for parameter in model.policy.parameters())
        return result

    monkeypatch.setattr(model, "translate", recorded_translate)
    data = TranslationDataModule(
        train_file=str(ROOT / "data/example_train.jsonl"),
        val_file=str(ROOT / "data/example_val.jsonl"),
        test_file=str(ROOT / "data/example_test.jsonl"),
        batch_size=2,
        bidirectional=True,
        num_workers=0,
    )
    trainer = L.Trainer(
        default_root_dir=tmp_path,
        accelerator="cpu",
        devices=1,
        max_steps=2,
        limit_train_batches=1,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    data.setup("fit")
    # Fix the first batch to a parallel pair's two directions; random shuffling
    # is allowed to return two samples of the same direction in production.
    loader = torch.utils.data.DataLoader(
        [data.splits["train"][0], data.splits["train"][1]], batch_size=2
    )
    trainer.fit(model, train_dataloaders=loader)
    assert trainer.global_step == 2 and torch.isfinite(
        trainer.callback_metrics["train/loss"]
    )
    changed = [
        name
        for name, parameter in model.policy.named_parameters()
        if not torch.equal(before[name], parameter)
    ]
    assert changed and all("lora_" in name for name in changed)
    assert reverse_calls and all(call["max_new_tokens"] == 4 for call in reverse_calls)
    assert {lang for call in reverse_calls for lang in call["source_langs"]} == {
        "English",
        "Vietnamese",
    }
    # Clear gradients left by the last update before testing detached validation.
    model.policy.zero_grad(set_to_none=True)
    evaluator = L.Trainer(
        default_root_dir=tmp_path,
        accelerator="cpu",
        devices=1,
        limit_val_batches=2,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    evaluator.validate(model, datamodule=data)
    report = json.loads((tmp_path / "translation_metrics/val.json").read_text())
    assert {"en_vi", "vi_en"} <= report["directions"].keys()
    assert (
        "val/sac_macro" in report["metrics"]
        and "val/round_trip_sac_macro" in report["metrics"]
    )
    with (tmp_path / "translation_metrics/val_predictions.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert rows and all("sac" in row and "round_trip_penalty" in row for row in rows)
    checkpoint = tmp_path / "sac.ckpt"
    trainer.save_checkpoint(checkpoint)
    restored = GRPOModule.load_from_checkpoint(
        checkpoint, reward=reward, weights_only=False
    )
    assert restored.reference is None and restored.kl_controller.value == 0
