"""Checkpointable adaptive control of sampled divergence to a frozen reference."""

import math
from typing import Any


class KLController:
    """Fixed beta by default; adaptive relaxation optionally requires quality floors.

    The caller supplies the sampled reference-divergence proxy, not old-policy
    KL. Targets are operator-calibrated; there is no implicit universal target.
    """

    def __init__(
        self,
        beta: float,
        enabled: bool = False,
        target: float | None = None,
        min_beta: float = 0.0025,
        max_beta: float = 0.04,
        rate: float = 0.1,
        ema_decay: float = 0.95,
        update_every: int = 100,
        quality_floors: dict[str, float] | None = None,
    ):
        numbers = (beta, min_beta, max_beta, rate, ema_decay)
        if any(isinstance(v, bool) or not math.isfinite(v) for v in numbers):
            raise ValueError("KL controller settings must be finite numbers")
        if (
            beta < 0
            or min_beta <= 0
            or max_beta < min_beta
            or rate <= 0
            or not 0 <= ema_decay < 1
        ):
            raise ValueError("Invalid KL coefficient bounds/rate/EMA")
        if (
            isinstance(update_every, bool)
            or not isinstance(update_every, int)
            or update_every < 1
        ):
            raise ValueError("update_every must be a positive integer")
        if enabled and (
            target is None
            or isinstance(target, bool)
            or not math.isfinite(target)
            or target <= 0
            or not min_beta <= beta <= max_beta
        ):
            raise ValueError(
                "Adaptive KL requires a calibrated positive target and initial beta within bounds"
            )
        self.enabled, self.target = enabled, target
        self.initial_beta = beta
        self.value, self.min_beta, self.max_beta = beta, min_beta, max_beta
        self.rate, self.ema_decay, self.update_every = rate, ema_decay, update_every
        self.quality_floors = dict(quality_floors or {})
        if any(not math.isfinite(v) for v in self.quality_floors.values()):
            raise ValueError("Quality floors must be finite")
        self.relaxation_allowed = not bool(self.quality_floors)
        self.ema: float | None = None
        self.observations = 0

    def observe_validation(self, metrics: dict[str, float]) -> None:
        """Unknown/missing protected metrics cannot authorize relaxation."""
        self.relaxation_allowed = all(
            name in metrics and math.isfinite(metrics[name]) and metrics[name] >= floor
            for name, floor in self.quality_floors.items()
        )

    def observe(self, reference_divergence: float) -> float:
        if not self.enabled:
            return self.value
        if not math.isfinite(reference_divergence) or reference_divergence < 0:
            raise ValueError("Reference divergence must be finite and nonnegative")
        self.observations += 1
        self.ema = (
            reference_divergence
            if self.ema is None
            else self.ema_decay * self.ema + (1 - self.ema_decay) * reference_divergence
        )
        if self.observations % self.update_every == 0:
            assert self.target is not None
            change = max(-1.0, min(1.0, self.ema / self.target - 1))
            if change >= 0 or self.relaxation_allowed:
                self.value = max(
                    self.min_beta,
                    min(self.max_beta, self.value * math.exp(self.rate * change)),
                )
        return self.value

    def state_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "ema": self.ema,
            "observations": self.observations,
            "relaxation_allowed": self.relaxation_allowed,
            "configuration": {
                k: getattr(self, k)
                for k in (
                    "initial_beta",
                    "enabled",
                    "target",
                    "min_beta",
                    "max_beta",
                    "rate",
                    "ema_decay",
                    "update_every",
                    "quality_floors",
                )
            },
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = self.state_dict()["configuration"]
        if state.get("configuration") != expected:
            raise ValueError(
                "KL controller settings changed on resume; use a fresh ablation run"
            )
        value, ema, observations = state["value"], state["ema"], state["observations"]
        if (
            not math.isfinite(value)
            or (self.enabled and not self.min_beta <= value <= self.max_beta)
            or (ema is not None and (not math.isfinite(ema) or ema < 0))
            or not isinstance(observations, int)
            or observations < 0
        ):
            raise ValueError("Invalid KL controller checkpoint state")
        self.value, self.ema, self.observations = value, ema, observations
        self.relaxation_allowed = bool(state["relaxation_allowed"])
