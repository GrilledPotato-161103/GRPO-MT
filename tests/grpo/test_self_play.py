"""Self-play curriculum, role rewards, checkpoints, and actual policy gradient tests."""

import json
from functools import partial
from pathlib import Path
from typing import Any

import lightning as L
import pytest
import torch
from hydra import compose, initialize

from scripts.verify.assets import create_assets
from src.data.self_play_datamodule import SelfPlayDataModule
from src.feedback.judge import TranslationJudge, TranslationTask
from src.models.components.optim import TransferAdamW
from src.models.components.self_play import (
    TaskMemory,
    learnability_rewards,
    parse_proposal,
)
from src.models.self_play_module import SelfPlayGRPOModule
from src.utils.self_play_timing import SelfPlayTiming


class ScriptedFeedback:
    """Deterministic judge; each solver group has a success and a failure."""

    async def generate(self, conversations: list[Any]) -> list[str]:
        result = []
        for i, conversation in enumerate(conversations):
            data = json.loads(conversation[-1]["content"])
            score = (1.0 if i % 2 else 0.2) if "candidate" in data else 1.0
            result.append(
                json.dumps({"valid": True, "score": score, "feedback": "test"})
            )
        return result


class ScriptedProposals(SelfPlayGRPOModule):
    """Script curriculum text while retaining real sampled tokens and likelihoods."""

    def _sample_prompts(
        self, prompts: list[str], group_size: int, max_new_tokens: int | None = None
    ) -> dict[str, Any]:
        rollout = super()._sample_prompts(prompts, group_size, max_new_tokens)
        if max_new_tokens is not None:
            rollout["predictions"] = ['{"source":"Hello"}', "not JSON"] * len(prompts)
        return rollout


def make_model(tmp_path: Path, **kwargs: Any) -> ScriptedProposals:
    torch.set_num_threads(1)
    L.seed_everything(42)
    asset = create_assets(tmp_path / "model", "causal")
    return ScriptedProposals(
        feedback=TranslationJudge(ScriptedFeedback()),
        model_name_or_path=str(asset),
        group_size=2,
        proposal_group_size=2,
        num_iterations=2,
        max_new_tokens=3,
        proposal_max_new_tokens=4,
        max_source_length=64,
        use_chat_template=False,
        learning_rate=1e-3,
        **kwargs,
    )


def test_learnability_and_proposal_validation() -> None:
    scores = learnability_rewards(torch.tensor([[0, 0], [1, 0], [1, 1]]))
    assert scores.tolist() == [0, 0.5, 0]
    with pytest.raises(ValueError):
        learnability_rewards(torch.ones(2))
    assert parse_proposal('{"source":"Hello"}', "en", "vi") == TranslationTask(
        "Hello", "en", "vi"
    )
    for value in ("{}", "[]", "not-json", '{"source":""}', '{"source":2}'):
        assert parse_proposal(value, "en", "vi") is None
    assert parse_proposal('{"source":"long"}', "en", "vi", max_chars=2) is None


def test_task_memory_is_bounded_and_deduplicated() -> None:
    memory = TaskMemory(1)
    assert memory.add(TranslationTask("Hello", "en", "vi"))
    assert not memory.add(TranslationTask("HELLO", "en", "vi"))
    memory.add(TranslationTask("new", "en", "vi"))
    clone = TaskMemory(2)
    clone.load_state_dict(memory.state_dict())
    assert clone.tasks == memory.tasks
    with pytest.raises(ValueError):
        clone.load_state_dict(
            [{"source": "", "source_lang": "en", "target_lang": "vi"}]
        )
    with pytest.raises(ValueError):
        TaskMemory(0)


@pytest.mark.parametrize("accelerator", ["cpu", "gpu"])
@pytest.mark.parametrize("optimizer_name", ["adamw", "transfer"])
def test_joint_training_and_checkpoint_memory(
    tmp_path: Path, accelerator: str, optimizer_name: str
) -> None:
    if accelerator == "gpu" and not torch.cuda.is_available():
        pytest.skip("CUDA GPU unavailable")
    model = make_model(
        tmp_path,
        optimizer=partial(TransferAdamW, lr=1e-3)
        if optimizer_name == "transfer"
        else None,
    )
    root = Path(__file__).resolve().parents[2]
    data = SelfPlayDataModule(
        topics=["daily life"],
        steps_per_epoch=1,
        val_file=str(root / "data/example_val.jsonl"),
        test_file=str(root / "data/example_test.jsonl"),
    )
    before = {k: v.clone() for k, v in model.policy.state_dict().items()}
    reference = {k: v.clone() for k, v in model.reference.state_dict().items()}
    trainer = L.Trainer(
        max_epochs=1,
        accelerator=accelerator,
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        limit_val_batches=1,
        limit_test_batches=1,
        num_sanity_val_steps=0,
        default_root_dir=tmp_path,
        callbacks=[SelfPlayTiming(str(tmp_path / "timing.json"))],
    )
    trainer.fit(model, datamodule=data)
    assert trainer.global_step == 2
    if optimizer_name == "transfer":
        assert isinstance(trainer.optimizers[0], TransferAdamW)
        assert "train/transfer/max_relative_drift" in trainer.callback_metrics
    timing = json.loads((tmp_path / "timing.json").read_text())
    assert timing["batches"] == 1
    assert timing["mean_seconds_per_batch"] > 0
    assert any(
        not torch.equal(before[k], v.cpu())
        for k, v in model.policy.state_dict().items()
    )
    assert all(
        torch.equal(reference[k], v.cpu())
        for k, v in model.reference.state_dict().items()
    )
    assert model.last_self_play["proposer_rewards"] == [0.5, -1]
    assert len(model.memory.tasks) == 1
    assert "train/proposer/reward" in trainer.callback_metrics
    assert "train/solver/reward" in trainer.callback_metrics
    trainer.test(model, datamodule=data)
    assert "test/chrf" in trainer.callback_metrics
    checkpoint = tmp_path / "self-play.ckpt"
    trainer.save_checkpoint(checkpoint)
    saved = torch.load(checkpoint, weights_only=False)
    assert "feedback" not in saved["hyper_parameters"]
    assert "CLAUDE_AUTH_KEY" not in repr(saved["hyper_parameters"])
    restored = SelfPlayGRPOModule.load_from_checkpoint(
        checkpoint, feedback=TranslationJudge(ScriptedFeedback()), weights_only=False
    )
    assert restored.memory.tasks == model.memory.tasks
    assert all(
        torch.equal(reference[k], v) for k, v in restored.reference.state_dict().items()
    )
    resumed = L.Trainer(
        max_epochs=2,
        accelerator=accelerator,
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        default_root_dir=tmp_path,
    )
    resumed.fit(restored, datamodule=data, ckpt_path=str(checkpoint))
    assert resumed.global_step == 4
    assert restored.memory.tasks == model.memory.tasks
    # The next proposal is a duplicate, but the solver can replay validated memory.
    rollout = model._self_play_rollouts(["travel"])
    assert len(rollout) == 2
    assert rollout[0][1]["rewards"].tolist() == [-0.5, -1]


def test_no_valid_proposals_is_handled_without_judge_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = make_model(tmp_path)
    original = model._sample_prompts

    def invalid(
        prompts: list[str], group_size: int, max_new_tokens: int | None = None
    ) -> dict[str, Any]:
        rollout = original(prompts, group_size, max_new_tokens)
        rollout["predictions"] = ["invalid"] * len(rollout["predictions"])
        return rollout

    monkeypatch.setattr(model, "_sample_prompts", invalid)
    rollout = model._self_play_rollouts(["topic"])
    assert len(rollout) == 1
    assert (rollout[0][1]["advantages"] == 0).all()
    with pytest.raises(ValueError, match="chat template"):
        model.hparams["use_chat_template"] = True
        model._role_prompt("x")


def test_self_play_configs() -> None:
    with initialize(version_base="1.3", config_path="../../configs"):
        cfg = compose(config_name="self_play")
        assert cfg.feedback.backend.model == "deepseek_v4.1_flash_1017"
        assert cfg.feedback.backend.api_key_env == "CLAUDE_AUTH_KEY"
        assert cfg.model._target_.endswith("SelfPlayGRPOModule")
        cfg = compose(
            config_name="self_play",
            overrides=[
                "feedback=self_hosted",
                "feedback.backend.model_name_or_path=local/judge",
            ],
        )
        assert cfg.feedback.backend.model_name_or_path == "local/judge"
        assert compose(config_name="self_play_eval").model._target_.endswith(
            "SelfPlayGRPOModule"
        )
        evaluation = compose(
            config_name="self_play_eval",
            overrides=["experiment=self_play_phomt", "paths.root_dir=."],
        )
        assert evaluation.data.val_file.endswith("phomt_prepared/validation.csv")
