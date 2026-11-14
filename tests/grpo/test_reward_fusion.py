"""Curriculum signs, group normalization and the requested final reward formula."""

import math

import pytest

from src.models.components.reward_fusion import DynamicRewardFusion
from src.models.components.translation_reward import TranslationReward


def test_exact_raw_formula_and_both_streams_reward_higher_scores() -> None:
    fusion = DynamicRewardFusion(
        enabled=True,
        alpha_start=0,
        alpha_end=0.5,
        total_steps=10,
        normalize_groups=False,
    )
    values = fusion.apply([0.9, 0.1], [0.2, 0.8], 5, 2)
    assert values["fusion_alpha"] == [0.25, 0.25]
    assert values["reward"] == pytest.approx(
        [1 / (1 + math.exp(-0.725)), 1 / (1 + math.exp(-0.275))]
    )
    base = fusion.apply([0.4], [0.4], 5, 1)["reward"][0]
    assert fusion.apply([0.5], [0.4], 5, 1)["reward"][0] > base
    assert fusion.apply([0.4], [0.5], 5, 1)["reward"][0] > base


def test_curriculum_clips_to_horizon_and_normalizes_only_inside_each_group() -> None:
    fusion = DynamicRewardFusion(enabled=True, total_steps=10)
    assert [fusion.alpha(step) for step in (0, 5, 10, 100)] == pytest.approx(
        [0.1, 0.5, 0.9, 0.9]
    )
    result = fusion.apply([0.1, 0.2, 0.8, 0.9], [0.3, 0.7, 0.1, 1.0], 5, 2)
    assert result["fused_comet_score"] == pytest.approx([-1, 1, -1, 1], abs=3e-5)
    assert result["reward"][0] == pytest.approx(result["reward"][2], abs=1e-5)
    tied = fusion.apply([0.5] * 4, [0.7] * 4, 5, 2)
    assert tied["reward"] == [0.5] * 4
    greedy = fusion.apply([0.6], [0.8], 5, 1)
    assert greedy["fusion_normalized"] == [0]
    assert greedy["reward"] == pytest.approx([1 / (1 + math.exp(-0.7))])


def test_fused_sac_combines_direct_and_symmetric_rewards_without_subtraction() -> None:
    class Scores:
        def evaluate(self, *args: object) -> list[float]:
            return [1.0, 0.7, 0.3, 1.0, 1.0, 0.3, 0.7, 1.0]

    reward = TranslationReward(
        chrf_weight=0,
        comet_weight=1,
        comet_scorer=lambda *args: [0.8] * 4,
        sac={
            "enabled": True,
            "consistency_mode": "symmetric_endpoints",
            "fusion": {"enabled": True, "normalize_groups": False, "total_steps": 10},
        },
        sac_scorer=Scores(),
    )
    result = reward.components(
        ["v0", "v1", "e0", "e1"],
        ["v", "v", "e", "e"],
        sources=["e", "e", "v", "v"],
        source_langs=["English"] * 2 + ["Vietnamese"] * 2,
        target_langs=["Vietnamese"] * 2 + ["English"] * 2,
        partner_indices=[2, 3, 0, 1],
        optimizer_step=5,
        candidate_group_size=2,
    )
    assert result["comet_reward"] == [0.8] * 4
    assert result["sac_reward"] == pytest.approx([0.925, 0.675, 0.575, 0.825])
    assert result["reward"] == pytest.approx(
        [1 / (1 + math.exp(-0.5 * (0.8 + score))) for score in result["sac_reward"]]
    )
    assert "symmetric_penalty" not in result and "round_trip_penalty" not in result


def test_invalid_or_penalty_signed_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        DynamicRewardFusion(alpha_start=0.9, alpha_end=0.1)
    with pytest.raises(ValueError):
        DynamicRewardFusion(total_steps=0)
    fusion = DynamicRewardFusion()
    with pytest.raises(ValueError):
        fusion.apply([-0.1], [0.7], 0, 1)
    with pytest.raises(ValueError):
        fusion.apply([0.8], [0.7], 0, 2)
    with pytest.raises(ValueError):
        fusion.alpha(-1)


def test_direct_only_ablation_does_not_request_unused_endpoint_feedback() -> None:
    class Direct:
        def evaluate(self, sources: list[str], *args: object) -> list[float]:
            assert len(sources) == 2
            return [1.0, 0.3]

    reward = TranslationReward(
        chrf_weight=0,
        comet_weight=1,
        comet_scorer=lambda *args: [0.8] * 2,
        sac={
            "enabled": True,
            "consistency_mode": "symmetric_endpoints",
            "fusion": {"enabled": True, "sac_direct_weight": 1},
        },
        sac_scorer=Direct(),
    )
    result = reward.components(
        ["a", "b"],
        ["target"] * 2,
        sources=["source"] * 2,
        source_langs=["English"] * 2,
        target_langs=["Vietnamese"] * 2,
        candidate_group_size=2,
    )
    assert result["sac_reward"] == [1.0, 0.3]
    assert "endpoint_sac" not in result
