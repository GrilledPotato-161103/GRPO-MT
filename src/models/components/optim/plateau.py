"""Checkpointed native ReduceLROnPlateau for manual Lightning optimization."""

import math
from typing import Any

from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau


def plateau_options(config: dict[str, Any]) -> dict[str, Any]:
    """Validate plain configuration before loading a policy or optimizer."""
    values: dict[str, Any] = {
        "kind": "reduce_on_plateau",
        "monitor": "val/comet_macro",
        "mode": "max",
        "factor": 0.5,
        "patience": 1,
        "threshold": 0.001,
        "threshold_mode": "abs",
        "cooldown": 0,
        "min_lr": 1e-6,
        "eps": 1e-8,
        "initial_best": None,
    }
    if set(config) - set(values):
        raise ValueError("Unknown plateau scheduler options")
    values.update(config)
    if values["kind"] != "reduce_on_plateau":
        raise ValueError("Only reduce_on_plateau is supported")
    if not isinstance(values["monitor"], str) or not values["monitor"].startswith(
        "val/"
    ):
        raise ValueError("Scheduler monitor must name a validation metric")
    if values["mode"] not in {"min", "max"} or values["threshold_mode"] not in {
        "abs",
        "rel",
    }:
        raise ValueError("Invalid plateau mode or threshold_mode")
    for key in ("patience", "cooldown"):
        value = values[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    for key in ("factor", "threshold", "min_lr", "eps"):
        value = values[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"{key} must be finite and nonnegative")
    if not 0 < values["factor"] < 1:
        raise ValueError("Plateau factor must be between zero and one")
    best = values["initial_best"]
    if best is not None and (
        isinstance(best, bool)
        or not isinstance(best, (int, float))
        or not math.isfinite(best)
    ):
        raise ValueError("initial_best must be finite")
    return values


class ValidationPlateau:
    """Step once per completed validation; preserve reductions and Adam moments."""

    def __init__(
        self,
        optimizer: Optimizer,
        learning_rate: float,
        config: dict[str, Any],
        restored: dict[str, Any] | None = None,
    ) -> None:
        self.config = plateau_options(config)
        if self.config["min_lr"] > learning_rate:
            raise ValueError("min_lr cannot exceed the starting learning rate")
        self.optimizer, self.learning_rate = optimizer, learning_rate
        options = {
            key: value
            for key, value in self.config.items()
            if key not in {"kind", "monitor", "initial_best"}
        }
        self.scheduler = ReduceLROnPlateau(optimizer, **options)
        self.last_validation_step = -1
        self.resumed = bool(
            restored
            and restored.get("config") == self.config
            and restored.get("learning_rate") == learning_rate
        )
        if self.resumed:
            assert restored is not None
            self.scheduler.load_state_dict(restored["scheduler"])
            self.last_validation_step = int(restored["last_validation_step"])
        else:
            # A deliberate new schedule overrides restored group rates, not moments.
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] = learning_rate
            self.scheduler._last_lr = [group["lr"] for group in optimizer.param_groups]
            if self.config["initial_best"] is not None:
                self.scheduler.best = float(self.config["initial_best"])

    def step(self, metrics: dict[str, float], global_step: int) -> dict[str, float]:
        """Skip repeated final validation of unchanged weights, never missing metrics."""
        if global_step <= self.last_validation_step:
            return {}
        monitor = self.config["monitor"]
        if monitor not in metrics or not math.isfinite(float(metrics[monitor])):
            raise ValueError(f"Missing/non-finite scheduler metric: {monitor}")
        previous = float(self.optimizer.param_groups[0]["lr"])
        self.scheduler.step(float(metrics[monitor]))
        self.last_validation_step = global_step
        return {
            "optimizer/lr": float(self.optimizer.param_groups[0]["lr"]),
            "optimizer/lr_reduced": float(
                self.optimizer.param_groups[0]["lr"] < previous
            ),
            "optimizer/plateau_bad_evaluations": float(self.scheduler.num_bad_epochs),
            "optimizer/plateau_best": float(self.scheduler.best),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "learning_rate": self.learning_rate,
            "scheduler": self.scheduler.state_dict(),
            "last_validation_step": self.last_validation_step,
        }
