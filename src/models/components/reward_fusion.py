"""Fuse two higher-is-better rewards with a checkpoint-progress curriculum."""

import math


class DynamicRewardFusion:
    """User-specified COMET/SAC adaptation of M2PO's curriculum fusion.

    Training can standardize each stream inside the source's candidate group.
    Greedy evaluation (group size one) uses raw bounded component rewards,
    because a singleton z-score contains no quality information.
    """

    def __init__(
        self,
        enabled: bool = False,
        alpha_start: float = 0.1,
        alpha_end: float = 0.9,
        total_steps: int = 10000,
        normalize_groups: bool = True,
        sac_direct_weight: float = 0.5,
        epsilon: float = 1e-6,
    ) -> None:
        for value in (alpha_start, alpha_end, sac_direct_weight):
            if (
                isinstance(value, bool)
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError("Fusion weights must be finite in [0,1]")
        if (
            alpha_end < alpha_start
            or isinstance(total_steps, bool)
            or not isinstance(total_steps, int)
            or total_steps < 1
            or isinstance(epsilon, bool)
            or not math.isfinite(epsilon)
            or epsilon <= 0
        ):
            raise ValueError(
                "Fusion requires a nondecreasing schedule, positive horizon and epsilon"
            )
        self.enabled, self.normalize_groups = enabled, normalize_groups
        self.alpha_start, self.alpha_end, self.total_steps = (
            alpha_start,
            alpha_end,
            total_steps,
        )
        self.sac_direct_weight, self.epsilon = sac_direct_weight, epsilon

    def alpha(self, step: int) -> float:
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError(
                "Curriculum progress must be a nonnegative optimizer-update count"
            )
        return self.alpha_start + (self.alpha_end - self.alpha_start) * min(
            step / self.total_steps, 1.0
        )

    def _normalize(self, values: list[float], group_size: int) -> list[float]:
        result: list[float] = []
        for start in range(0, len(values), group_size):
            group = values[start : start + group_size]
            mean = sum(group) / group_size
            deviation = math.sqrt(
                sum((value - mean) ** 2 for value in group) / group_size
            )
            result.extend(
                (value - mean) / (deviation + self.epsilon) for value in group
            )
        return result

    def apply(
        self, comet: list[float], sac: list[float], step: int, group_size: int
    ) -> dict[str, list[float]]:
        if (
            len(comet) != len(sac)
            or isinstance(group_size, bool)
            or not isinstance(group_size, int)
            or group_size < 1
            or len(comet) % group_size
        ):
            raise ValueError(
                "Fusion streams require matching complete candidate groups"
            )
        if any(
            not math.isfinite(value) or not 0 <= value <= 1 for value in (*comet, *sac)
        ):
            raise ValueError(
                "Both fusion streams must be bounded higher-is-better rewards"
            )
        normalized = self.normalize_groups and group_size > 1
        quality = self._normalize(comet, group_size) if normalized else list(comet)
        alignment = self._normalize(sac, group_size) if normalized else list(sac)
        alpha = self.alpha(step)
        logits = [
            (1 - alpha) * q + alpha * a for q, a in zip(quality, alignment, strict=True)
        ]
        reward = [
            1 / (1 + math.exp(-value))
            if value >= 0
            else math.exp(value) / (1 + math.exp(value))
            for value in logits
        ]
        return {
            "comet_reward": comet,
            "sac_reward": sac,
            "fused_comet_score": quality,
            "fused_sac_score": alignment,
            "fusion_alpha": [alpha] * len(comet),
            "fusion_normalized": [float(normalized)] * len(comet),
            "reward": reward,
        }
