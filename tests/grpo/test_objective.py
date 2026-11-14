"""Mathematical regression tests for policy gradients, clipping, and masking."""

import pytest
import torch

from src.models.components.grpo import completion_mask, group_advantages, grpo_loss
from src.models.components.translation_reward import TranslationReward


@pytest.mark.parametrize("eos", [1, [1, 5]])
def test_eos_is_included_but_padding_is_not(eos: int | list[int]) -> None:
    tokens = torch.tensor([[4, 1, 1, 1], [4, 3, 2, 4], [1, 1, 1, 1]])
    assert completion_mask(tokens, eos).tolist() == [
        [True, True, False, False],
        [True] * 4,
        [True, False, False, False],
    ]
    assert completion_mask(tokens, None).all()


def test_group_normalization_is_local_and_equal_rewards_are_zero() -> None:
    rewards = torch.tensor([1.0, 3.0, 100.0, 100.0])
    advantages = group_advantages(rewards, 2)
    assert torch.allclose(advantages, torch.tensor([-1.0, 1.0, 0.0, 0.0]), atol=1e-3)
    for size in (1, 3):
        with pytest.raises(ValueError):
            group_advantages(rewards, size)


def test_gradient_improves_rewarded_translation_and_ignores_padding() -> None:
    logps = torch.zeros(2, 3, requires_grad=True)
    mask = torch.tensor([[True, True, False], [True, False, False]])
    loss, metrics = grpo_loss(
        logps, logps.detach(), logps.detach(), torch.tensor([1.0, -1.0]), mask
    )
    loss.backward()
    assert logps.grad is not None
    assert logps.grad[0, 0] < 0 < logps.grad[1, 0]
    assert (logps.grad[~mask] == 0).all()
    assert metrics["kl"] == 0
    assert metrics["completion_length"] == 1.5


def test_clipping_stops_gradient_and_kl_penalizes_drift() -> None:
    current = torch.tensor([[0.5]], requires_grad=True)
    old = torch.zeros_like(current)
    mask = torch.ones_like(current, dtype=torch.bool)
    loss, metrics = grpo_loss(current, old, old, torch.ones(1), mask, beta=0)
    loss.backward()
    assert current.grad is not None and current.grad.item() == 0
    assert metrics["clip_fraction"] == 1
    loss_kl, _ = grpo_loss(current, old, old, torch.ones(1), mask, beta=0.5)
    assert loss_kl > loss


def test_translation_reward_scale_and_validation() -> None:
    reward = TranslationReward(chrf_weight=0.7, bleu_weight=0.3)
    scores = reward(["Xin chào bạn", "", "wrong text"], ["Xin chào bạn"] * 3)
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == 0
    assert scores[2] < scores[0]
    with pytest.raises(ValueError):
        TranslationReward(-1, 1)
    with pytest.raises(ValueError):
        reward(["one"], [])
