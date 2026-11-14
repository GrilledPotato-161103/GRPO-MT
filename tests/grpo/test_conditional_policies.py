"""Criterion reliability, non-compensation, caching, group weights and KL resume."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.feedback.backends import FeedbackError
from src.feedback.criteria import Criterion, CriterionJudge, parse_criteria
from src.feedback.judge import TranslationTask
from src.models.components.grpo import group_advantages
from src.models.components.kl_control import KLController
from src.models.components.reward_harness import (
    ConditionalRewardHarness,
    group_update_weights,
)


def annotation(error: bool = False) -> tuple[Criterion, ...]:
    return tuple(
        Criterion(
            name,
            True,
            0.1 if error and name == "omission" else 0.95,
            0.99,
            "major" if error and name == "omission" else "none",
            ("source clause omitted",) if error and name == "omission" else (),
        )
        for name in ("adequacy", "omission", "addition", "mistranslation", "negation")
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"schema_version": 1, "criteria": []},
        {"schema_version": 1, "criteria": [{"name": "unknown"}]},
        {
            "schema_version": 1,
            "criteria": [
                {
                    "name": "adequacy",
                    "applicable": False,
                    "value": 1,
                    "confidence": 0.9,
                    "severity": "none",
                    "evidence": [],
                }
            ],
        },
        {
            "schema_version": 1,
            "criteria": [
                {
                    "name": "adequacy",
                    "applicable": True,
                    "value": True,
                    "confidence": 0.9,
                    "severity": "none",
                    "evidence": [],
                }
            ],
        },
        {
            "schema_version": 1,
            "criteria": [
                {
                    "name": "adequacy",
                    "applicable": True,
                    "value": 0.1,
                    "confidence": 0.9,
                    "severity": "major",
                    "evidence": [],
                }
            ],
        },
    ],
)
def test_malformed_criteria_do_not_become_rewards(payload: dict) -> None:
    with pytest.raises(FeedbackError):
        parse_criteria(json.dumps(payload))


def test_semantic_veto_shadow_and_whole_group_weights() -> None:
    judge = SimpleNamespace(
        evaluate=lambda tasks, candidates: [annotation(True), annotation(False)]
    )
    raw = {"reward": [0.95, 0.45], "comet": [0.99, 0.8], "bleu": [0.9, 0.1]}
    args = (
        ["fluent but incomplete", "faithful paraphrase"],
        ["source"] * 2,
        ["English"] * 2,
        ["Vietnamese"] * 2,
    )
    shadow = ConditionalRewardHarness(enabled=True, judge=judge)
    result = shadow.apply(raw, *args)
    assert result["reward"] == raw["reward"]
    assert result["group_confidence"] == [1, 1]
    empty_raw = {**raw, "reward": [0.0, 0.45]}
    shadow_args = (["", args[0][1]], *args[1:])
    assert shadow.apply(empty_raw, *shadow_args)["reward"] == empty_raw["reward"]
    active = ConditionalRewardHarness(enabled=True, calibrated=True, judge=judge)
    result = active.apply(raw, *args)
    assert result["reward"][0] < 0 < result["reward"][1]
    assert result["audit_reward"] == raw["reward"]
    assert result["lexical_relaxed"] == [0, 1]
    assert group_update_weights(result, 2, 2) == [1, 1]
    assert group_update_weights(
        {"group_confidence": [0.9, 0.2], "constraint_failure": [0, 0]}, 2, 2
    ) == [0.2, 0.2]
    assert group_update_weights({"constraint_failure": [1, 1]}, 2, 2) == [0, 0]
    a = group_advantages(torch.tensor([0.2, 0.8]), 2)
    assert torch.allclose(a, group_advantages(torch.tensor([-0.3, 0.3]), 2))
    assert torch.linalg.vector_norm(a * 0.2) == pytest.approx(
        float(torch.linalg.vector_norm(a)) * 0.2
    )


def test_unknown_missing_and_unrequested_obligations_are_not_passes() -> None:
    criteria = (
        Criterion("adequacy", True, None, None, "none", ()),
        Criterion("terminology", True, 0.1, 0.99, "major", ("no supplied glossary",)),
    )
    judge = SimpleNamespace(evaluate=lambda tasks, candidates: [criteria])
    result = ConditionalRewardHarness(enabled=True, calibrated=True, judge=judge).apply(
        {"reward": [0.7]}, ["candidate"], ["source"], ["English"], ["Vietnamese"]
    )
    assert result["semantic_known"] == [0] and result["group_confidence"] == [0.25]
    assert result["constraint_failure"] == [0]


def test_criterion_cache_preserves_direction_and_never_calls_backend_on_hits(
    tmp_path: Path,
) -> None:
    class Backend:
        calls = 0

        async def generate(self, conversations):
            self.calls += 1
            from dataclasses import asdict

            return [
                json.dumps(
                    {
                        "schema_version": 1,
                        "criteria": [asdict(item) for item in annotation()],
                    }
                )
                for _ in conversations
            ]

    backend = Backend()
    judge = CriterionJudge(backend, str(tmp_path))
    task = TranslationTask("source", "English", "Vietnamese")
    assert judge.evaluate([task], ["translation"]) == [annotation()]
    assert judge.evaluate([task], ["translation"]) == [annotation()]
    assert backend.calls == 1
    offline = CriterionJudge(backend, str(tmp_path), cache_only=True)
    assert offline.evaluate([task], ["translation"]) == [annotation()]
    with pytest.raises(FeedbackError, match="cache miss"):
        offline.evaluate(
            [TranslationTask("source", "Vietnamese", "English")], ["translation"]
        )


def test_adaptive_kl_guards_bounds_and_exact_resume() -> None:
    cfg = {
        "enabled": True,
        "target": 0.1,
        "update_every": 1,
        "ema_decay": 0.0,
        "quality_floors": {"val/en_vi/chrf": 50.0},
    }
    controller = KLController(0.01, **cfg)
    assert controller.observe(0.0) == 0.01  # no quality check yet: no relaxation
    controller.observe_validation({"val/en_vi/chrf": 51.0})
    smaller = controller.observe(0.0)
    assert smaller < 0.01
    controller.observe_validation({"val/en_vi/chrf": 49.0})
    assert controller.observe(0.0) == smaller
    assert controller.observe(1.0) > smaller
    restored = KLController(0.01, **cfg)
    restored.load_state_dict(controller.state_dict())
    assert restored.state_dict() == controller.state_dict()
    assert restored.observe(0.8) == controller.observe(0.8)
    with pytest.raises(ValueError, match="changed"):
        KLController(0.02, **cfg).load_state_dict(controller.state_dict())
    with pytest.raises(ValueError, match="calibrated"):
        KLController(0.01, enabled=True)
    fixed = KLController(0.04)
    assert fixed.observe(99.0) == 0.04
