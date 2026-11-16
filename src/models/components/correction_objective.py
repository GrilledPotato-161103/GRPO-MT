"""Judge-verified anchors and a bounded, quality-weighted preference objective."""

import math
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class CorrectionOptions:
    """Thresholds are explicit heuristics until calibrated on inspected EN/VI data."""

    beta: float = 1.0
    anchor_weight: float = 1.0
    min_quality_gap: float = 0.05
    anchor_comet_min: float = 0.5
    rejected_sac_below: float = 0.7
    anchor_sac_min: float = 1.0
    max_consecutive_skips: int = 20
    anchor_selection: str = "best_verified_quality"
    allow_reference_fallback: bool = True
    noise_aware: bool = False
    reward_noise_floor: float = 0.02
    advantage_scale: float = 0.1
    max_advantage: float = 2.0
    comet_disagreement_tolerance: float = 0.02
    label_smoothing: float = 0.1
    candidate_likelihood_weight: float = 0.1
    reference_likelihood_weight: float = 0.25
    anchor_nll_floor: float = 0.5
    criterion_pair_weight: float = 0.1
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.anchor_selection not in {"best_verified_quality", "best_comet"}:
            raise ValueError("Unknown correction.anchor_selection")
        if not isinstance(self.allow_reference_fallback, bool):
            raise TypeError("correction.allow_reference_fallback must be boolean")
        if not isinstance(self.noise_aware, bool):
            raise TypeError("correction.noise_aware must be boolean")
        for name in (
            "reward_noise_floor",
            "advantage_scale",
            "max_advantage",
            "anchor_nll_floor",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"correction.{name} must be finite and positive")
        for name in (
            "comet_disagreement_tolerance",
            "candidate_likelihood_weight",
            "reference_likelihood_weight",
            "label_smoothing",
            "criterion_pair_weight",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError(f"correction.{name} must be finite in [0,1]")
        if self.label_smoothing >= 0.5:
            raise ValueError("correction.label_smoothing must be below .5")
        if not isinstance(self.evidence, dict) or (
            self.evidence and not self.noise_aware
        ):
            raise ValueError("correction.evidence requires noise_aware mode")
        if self.evidence.get("enabled") and self.evidence.get("calibrated") is not True:
            raise ValueError(
                "Enabled training evidence requires external calibration; collect uncalibrated judgments offline"
            )
        for name in ("beta", "anchor_weight"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"correction.{name} must be finite and positive")
        for name in ("min_quality_gap", "anchor_comet_min"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not math.isfinite(value)
                or not 0 <= value < 1
            ):
                raise ValueError(f"correction.{name} must be in [0, 1)")
        if self.rejected_sac_below != 0.7 or self.anchor_sac_min != 1.0:
            raise ValueError(
                "Initial correction objective requires SAC < .7 rejected and 1.0 anchors"
            )
        if (
            isinstance(self.max_consecutive_skips, bool)
            or not isinstance(self.max_consecutive_skips, int)
            or self.max_consecutive_skips < 1
        ):
            raise ValueError("max_consecutive_skips must be a positive integer")


@dataclass(frozen=True)
class CorrectionGroup:
    """Indices point into one deduplicated policy-score pool."""

    anchor: int
    rejected: tuple[int, ...]
    weights: tuple[float, ...]
    source_index: int
    provenance: str
    likelihood_weight: float = 1.0


def noise_aware_advantages(
    rewards: Tensor, group_size: int, options: CorrectionOptions
) -> Tensor:
    """Antisymmetric pairwise preferences with a dead band and a fixed scale.

    A tiny observed spread cannot turn into a unit update. The floor is a
    sensitivity parameter, not an estimated judge standard deviation.
    """
    if group_size < 2 or rewards.ndim != 1 or rewards.numel() % group_size:
        raise ValueError("Rewards must contain complete groups of size >= 2")
    if not torch.isfinite(rewards).all():
        raise ValueError("Rewards must be finite")
    grouped = rewards.detach().float().reshape(-1, group_size)
    differences = grouped[:, :, None] - grouped[:, None, :]
    contrasts = differences.sign() * (
        differences.abs() - options.reward_noise_floor
    ).clamp_min(0)
    return (
        (contrasts / options.advantage_scale)
        .clamp(-options.max_advantage, options.max_advantage)
        .mean(-1)
        .reshape(-1)
    )


def correction_loss(
    mean_logps: Tensor,
    groups: list[CorrectionGroup],
    options: CorrectionOptions,
    source_count: int,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Average over source groups; uncertain small gaps retain small coefficients.

    No per-group re-normalization of gap weights, no winner selected from an
    all-failed set, and no probability ratio for off-policy reference targets.
    """
    if mean_logps.ndim != 1 or source_count < 1:
        raise ValueError("Need a score vector and positive source_count")
    zero = mean_logps.sum() * 0
    preference, nll, margins, pair_weight = zero, zero, zero, zero
    raw_nll, saturated = zero, zero
    pairs = 0
    for group in groups:
        if (
            not math.isfinite(group.likelihood_weight)
            or not 0 <= group.likelihood_weight <= 1
        ):
            raise ValueError("Anchor likelihood weight must be bounded and finite")
        if len(group.rejected) != len(group.weights):
            raise ValueError("Correction indices and weights must align")
        if any(not math.isfinite(w) or not 0 <= w <= 1 for w in group.weights):
            raise ValueError("Correction weights must be bounded and finite")
        chosen = mean_logps[group.anchor]
        anchor_nll = -chosen
        raw_nll = raw_nll + anchor_nll
        if options.noise_aware:
            saturated = saturated + (anchor_nll <= options.anchor_nll_floor).float()
            # Stop self-distillation after the target is already likely enough.
            anchor_nll = (anchor_nll - options.anchor_nll_floor).clamp_min(0)
        nll = nll + group.likelihood_weight * anchor_nll
        if group.rejected:
            rejected = mean_logps[list(group.rejected)]
            weights = mean_logps.new_tensor(group.weights)
            gap = chosen - rejected
            terms = -F.logsigmoid(options.beta * gap)
            if options.noise_aware:
                terms = (
                    1 - options.label_smoothing
                ) * terms - options.label_smoothing * F.logsigmoid(-options.beta * gap)
            preference = preference + (weights * terms).mean()
            margins = margins + gap.sum()
            pair_weight = pair_weight + weights.sum()
            pairs += len(group.rejected)
    loss = (preference + options.anchor_weight * nll) / source_count
    return loss, {
        "correction/loss": loss.detach(),
        "correction/preference_loss": preference.detach() / source_count,
        "correction/anchor_nll": nll.detach() / max(1, len(groups)),
        "correction/anchor_nll_raw": raw_nll.detach() / max(1, len(groups)),
        "correction/anchor_nll_saturated_fraction": saturated.detach()
        / max(1, len(groups)),
        "correction/logprob_margin": margins.detach() / max(1, pairs),
        "correction/pair_weight": pair_weight.detach() / max(1, pairs),
        "correction/pairs": mean_logps.new_tensor(pairs),
    }


def quality_scores(comet: list[float], sac: list[float]) -> list[float]:
    """Absolute COMET x SAC; never normalize away an acceptance threshold."""
    if len(comet) != len(sac) or any(not math.isfinite(x) for x in (*comet, *sac)):
        raise ValueError("COMET and SAC must be aligned and finite")
    if any(x not in {0.1, 0.3, 0.7, 1.0} for x in sac):
        raise ValueError("Correction routing requires exact categorical SAC tiers")
    return [min(1.0, max(0.0, c)) * s for c, s in zip(comet, sac, strict=True)]
