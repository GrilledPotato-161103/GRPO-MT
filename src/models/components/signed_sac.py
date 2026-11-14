"""Separate semantic matches from mismatches before GRPO advantage estimation."""

import math


class SignedSAC:
    """A signed categorical branch, not an additional differentiable reward model.

    Match candidates receive COMET/SAC fusion. Unmatch candidates receive only a
    negative severity score. The ordinary GRPO baseline is applied afterwards.
    """

    def __init__(
        self,
        enabled: bool = False,
        penalty_kind: str = "linear",
        penalty_weight: float = 1.0,
        penalty_temperature: float = 1.0,
        partial_unmatch_severity: float = 0.5,
    ) -> None:
        if penalty_kind not in {"linear", "centered_inverse_sigmoid"}:
            raise ValueError("penalty_kind must be linear or centered_inverse_sigmoid")
        for name, value in (
            ("penalty_weight", penalty_weight),
            ("penalty_temperature", penalty_temperature),
        ):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            isinstance(partial_unmatch_severity, bool)
            or not math.isfinite(partial_unmatch_severity)
            or not 0 < partial_unmatch_severity < 1
        ):
            raise ValueError(
                "Partial-unmatch severity must be strictly between zero and one"
            )
        self.enabled = enabled
        self.penalty_kind, self.weight = penalty_kind, penalty_weight
        self.temperature, self.partial = penalty_temperature, partial_unmatch_severity

    def separate(
        self, scores: list[float]
    ) -> tuple[list[float], list[float], list[float]]:
        """Legacy scalar tiers retain the same semantic meaning under either schema."""
        positive, severity, matched = [], [], []
        for value in scores:
            if value not in {0.1, 0.3, 0.7, 1.0}:
                raise ValueError(
                    "Signed SAC needs exact categorical tiers before any averaging"
                )
            good = value >= 0.7
            positive.append(value if good else 0.0)
            severity.append(0.0 if good else self.partial if value == 0.3 else 1.0)
            matched.append(float(good))
        return positive, severity, matched

    def apply(
        self, scores: list[float], fused_positive: list[float]
    ) -> dict[str, list[float]]:
        if len(scores) != len(fused_positive) or any(
            not math.isfinite(q) or not 0 <= q <= 1 for q in fused_positive
        ):
            raise ValueError(
                "Signed SAC requires aligned bounded positive fusion scores"
            )
        positive, severity, matched = self.separate(scores)
        # 2*(1-sigmoid(k*p))-1 = -tanh(k*p/2), evaluated stably.
        centered = [-math.tanh(self.temperature * p / 2) for p in severity]
        negative = (
            [-self.weight * p for p in severity]
            if self.penalty_kind == "linear"
            else [self.weight * p for p in centered]
        )
        reward = [
            q if good else p
            for q, p, good in zip(fused_positive, negative, matched, strict=True)
        ]
        return {
            "sac_positive_reward": positive,
            "sac_penalty_severity": severity,
            "sac_match_mask": matched,
            "sac_penalty_gate": [(p + 1) / 2 for p in centered],
            "sac_positive_fused_reward": [
                q if good else 0.0
                for q, good in zip(fused_positive, matched, strict=True)
            ],
            "sac_negative_reward": negative,
            "sac_signed_reward": reward,
            "reward": reward,
        }
