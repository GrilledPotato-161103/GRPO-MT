"""Pure tensor operations for group-relative policy optimization."""

import torch
from torch import Tensor


def completion_mask(tokens: Tensor, eos_token_id: int | list[int] | None) -> Tensor:
    """Keep tokens through the first EOS, including EOS even when it is also PAD."""
    if eos_token_id is None:
        return torch.ones_like(tokens, dtype=torch.bool)
    eos_ids = torch.as_tensor(eos_token_id, device=tokens.device).reshape(-1)
    is_eos = (tokens.unsqueeze(-1) == eos_ids).any(-1)
    return (is_eos.cumsum(-1) - is_eos.long()) == 0


def group_advantages(rewards: Tensor, group_size: int, eps: float = 1e-4) -> Tensor:
    """Normalize rewards within each source's contiguous group of translations."""
    if group_size < 2 or rewards.ndim != 1 or rewards.numel() % group_size:
        raise ValueError("Rewards must be a vector of complete groups of size >= 2")
    grouped = rewards.float().reshape(-1, group_size)
    return (
        (grouped - grouped.mean(-1, keepdim=True))
        / (grouped.std(-1, keepdim=True, correction=0) + eps)
    ).reshape(-1)


def grpo_loss(
    logps: Tensor,
    old_logps: Tensor,
    reference_logps: Tensor,
    advantages: Tensor,
    mask: Tensor,
    clip_epsilon: float = 0.2,
    beta: float = 0.04,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Original GRPO: token-clipped surrogate and k3 KL, averaged per completion.

    All log probabilities describe the same temperature-scaled, full-support policy.
    Old-policy and reference probabilities must be detached by the rollout caller.
    """
    ratio = (logps - old_logps).exp()
    surrogate = torch.minimum(
        ratio * advantages[:, None],
        ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * advantages[:, None],
    )
    # Avoid a reference graph and 0 * inf when KL is explicitly disabled.
    if beta:
        delta = reference_logps - logps
        kl = delta.exp() - delta - 1
    else:
        kl = torch.zeros_like(logps)
    lengths = mask.sum(-1).clamp_min(1)
    loss = ((-surrogate + beta * kl) * mask).sum(-1).div(lengths).mean()
    metrics = {
        "kl": (kl * mask).sum().div(mask.sum().clamp_min(1)),
        "clip_fraction": (((ratio - 1).abs() > clip_epsilon) * mask)
        .sum()
        .div(mask.sum().clamp_min(1)),
        "completion_length": mask.sum(-1).float().mean(),
        "policy_loss": (-surrogate * mask).sum(-1).div(lengths).mean(),
        "kl_loss": (kl * mask).sum(-1).div(lengths).mean(),
        "weighted_kl_loss": beta * (kl * mask).sum(-1).div(lengths).mean(),
        "ratio_mean": (ratio * mask).sum().div(mask.sum().clamp_min(1)),
        "approx_kl_to_old": ((ratio - 1 - (logps - old_logps)) * mask)
        .sum()
        .div(mask.sum().clamp_min(1)),
        "sampled_token_nll": -(logps * mask).sum().div(mask.sum().clamp_min(1)),
    }
    return loss, metrics
