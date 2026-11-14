"""Judge-verified anchors and a bounded, quality-weighted preference objective."""

import math
from dataclasses import dataclass

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

    def __post_init__(self) -> None:
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
    pairs = 0
    for group in groups:
        if len(group.rejected) != len(group.weights):
            raise ValueError("Correction indices and weights must align")
        if any(not math.isfinite(w) or not 0 <= w <= 1 for w in group.weights):
            raise ValueError("Correction weights must be bounded and finite")
        chosen = mean_logps[group.anchor]
        nll = nll - chosen
        if group.rejected:
            rejected = mean_logps[list(group.rejected)]
            weights = mean_logps.new_tensor(group.weights)
            gap = chosen - rejected
            preference = (
                preference - (weights * F.logsigmoid(options.beta * gap)).mean()
            )
            margins = margins + gap.sum()
            pair_weight = pair_weight + weights.sum()
            pairs += len(group.rejected)
    loss = (preference + options.anchor_weight * nll) / source_count
    return loss, {
        "correction/loss": loss.detach(),
        "correction/preference_loss": preference.detach() / source_count,
        "correction/anchor_nll": nll.detach() / max(1, len(groups)),
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
