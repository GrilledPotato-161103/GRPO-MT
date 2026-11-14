"""Within-source reward diagnostics; no text or reference translations are recorded."""

from itertools import combinations
from typing import Any

import torch
from torch import Tensor


@torch.no_grad()
def reward_diagnostics(
    rewards: Tensor,
    components: dict[str, list[float]],
    group_size: int,
    predictions: list[str],
    directions: list[str] | None = None,
    weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Separate candidate disagreement from differences between source difficulties.

    Correlation is computed after centering *within each source*. Pairwise
    disagreement excludes ties and reports its denominator; constant components
    have no reported correlation rather than a fabricated zero correlation.
    """
    grouped = rewards.detach().float().cpu().reshape(-1, group_size)
    if len(predictions) != rewards.numel():
        raise ValueError("Each reward must have a decoded candidate")
    std = grouped.std(-1, correction=0)
    metrics = {
        "rollout/reward_mean": float(grouped.mean()),
        "rollout/reward_std": float(grouped.std(correction=0)),
        "rollout/reward_min": float(grouped.min()),
        "rollout/reward_max": float(grouped.max()),
        "rollout/group_reward_std": float(std.mean()),
        "rollout/group_reward_range": float(
            (grouped.max(-1).values - grouped.min(-1).values).mean()
        ),
        "rollout/zero_variance_groups": float((std == 0).float().mean()),
        "rollout/near_tie_groups": float((std < 1e-3).float().mean()),
        "rollout/empty_fraction": sum(not value.strip() for value in predictions)
        / len(predictions),
        "rollout/unique_candidate_fraction": sum(
            len({value.strip() for value in predictions[offset : offset + group_size]})
            / group_size
            for offset in range(0, len(predictions), group_size)
        )
        / len(grouped),
    }
    tensors: dict[str, Tensor] = {}
    for name, values in components.items():
        if name == "reward":
            continue
        if name in {
            "audit_reward",
            "constraint_failure",
            "semantic_known",
            "group_confidence",
            "lexical_relaxed",
        }:
            metrics[f"harness/{name}_mean"] = sum(values) / len(values)
            continue
        value = torch.tensor(values, dtype=torch.float32).reshape_as(grouped)
        effective = value.clamp(0, 1) if name == "comet" else value
        penalty = name in {"round_trip_penalty", "symmetric_penalty", "sac_penalty_severity"}
        computed = name in {
            "fused_comet_score", "fused_sac_score", "fusion_alpha", "fusion_normalized",
            "sac_penalty_severity", "sac_negative_reward", "sac_signed_reward",
            "sac_match_mask", "sac_penalty_gate", "sac_positive_fused_reward",
        }
        if not penalty and not computed:
            effective = effective.masked_fill(
                torch.tensor([not text.strip() for text in predictions]).reshape_as(
                    grouped
                ),
                0,
            )
        # Correlations compare effects on reward: smaller penalties are better.
        tensors[name] = -effective if penalty else effective
        metrics[f"reward/{name}_mean"] = float(value.mean())
        metrics[f"reward/{name}_group_std"] = float(value.std(-1, correction=0).mean())
        if weights and name in weights:
            # Match TranslationReward's component transformations for variance accounting.
            metrics[f"reward/{name}_weighted_group_std"] = float(
                (effective * weights[name]).std(-1, correction=0).mean()
            )
        if name == "comet":
            metrics["reward/comet_clipped_fraction"] = float(
                ((value < 0) | (value > 1)).float().mean()
            )
    pairs = torch.triu_indices(group_size, group_size, offset=1)
    for left, right in combinations(tensors, 2):
        a, b = tensors[left], tensors[right]
        da, db = a[:, pairs[0]] - a[:, pairs[1]], b[:, pairs[0]] - b[:, pairs[1]]
        comparable = (da.abs() > 1e-6) & (db.abs() > 1e-6)
        prefix = f"reward_conflict/{left}_{right}"
        metrics[prefix + "_comparable_pairs"] = float(comparable.sum())
        if comparable.any():
            metrics[prefix + "_disagreement"] = float(
                ((da * db < 0) & comparable).sum() / comparable.sum()
            )
        a, b = (
            (a - a.mean(-1, keepdim=True)).flatten(),
            (b - b.mean(-1, keepdim=True)).flatten(),
        )
        denominator = a.norm() * b.norm()
        if denominator > 0:
            metrics[prefix + "_within_source_correlation"] = float(
                (a @ b) / denominator
            )
    if directions is not None:
        if len(directions) != len(grouped):
            raise ValueError(
                "Directions describe source groups, not individual candidates"
            )
        for direction in sorted(set(directions)):
            selected = torch.tensor([value == direction for value in directions])
            metrics[f"direction/{direction}/reward"] = float(grouped[selected].mean())
            for name, value in tensors.items():
                raw = (
                    -value
                    if name in {"round_trip_penalty", "symmetric_penalty", "sac_penalty_severity"}
                    else value
                )
                metrics[f"direction/{direction}/{name}"] = float(raw[selected].mean())
    if "sac_match_mask" in components:
        good = torch.tensor(components["sac_match_mask"]).bool().reshape_as(grouped)
        bad = ~good
        metrics["polarity/match_fraction"] = float(good.float().mean())
        metrics["polarity/all_unmatch_groups"] = float(bad.all(-1).float().mean())
        metrics["polarity/mixed_groups"] = float((good.any(-1) & bad.any(-1)).float().mean())
        metrics["polarity/all_unmatch_tied_groups"] = float((bad.all(-1) & (std == 0)).float().mean())
        metrics["polarity/unmatch_positive_advantage_fraction"] = float(
            (((grouped > grouped.mean(-1, keepdim=True)) & bad).sum() / bad.sum().clamp_min(1))
        )
    return metrics


def component_weights(reward: Any) -> dict[str, float]:
    """Normalized effective mixing coefficients, if exposed by a reward callable."""
    if getattr(reward, "sac_enabled", False):
        # Multiplicative SAC gating is not an additive COMET/BLEU mixture.
        return {}
    weights = {
        name: float(getattr(reward, name + "_weight", 0))
        for name in ("chrf", "bleu", "comet")
    }
    total = sum(weights.values())
    return {name: weight / total for name, weight in weights.items()} if total else {}
