"""AdamW with a pretrained-weight anchor and optional parameter-drift projection.

L2-SP adds ``lambda / 2 * ||parameter - initial_parameter||**2`` to the
optimization objective. The optional projection bounds each tensor's distance
from its anchor. These controls limit parameter drift; they do not guarantee
retention of a model's behavioral capabilities.
"""

from collections.abc import Callable, Iterable
from math import isfinite, sqrt
from numbers import Real
from typing import Any

import torch
from torch import Tensor


class TransferAdamW(torch.optim.AdamW):
    """Native AdamW with L2-SP regularization and a per-tensor trust region.

    Args:
        params: Trainable tensors or ordinary PyTorch parameter groups.
        lr: AdamW learning rate; schedulers can change each group's rate.
        betas: Adam moment coefficients.
        eps: Adam denominator epsilon.
        weight_decay: Ordinary decay toward zero, independent of L2-SP.
        amsgrad: Enable AdamW's AMSGrad variant.
        l2sp_weight: Nonnegative coefficient for the starting-point penalty.
        max_relative_drift: Optional nonnegative per-tensor drift bound.
        min_reference_norm: Positive norm floor, allowing zero anchors to learn.
        maximize: Maximize the task objective while still minimizing L2-SP.
        foreach: Forward to PyTorch's foreach AdamW implementation.

    Group dictionaries may override any of these options. Projection uses
    ``radius = max_relative_drift * max(norm(anchor), min_reference_norm)``.
    Anchors are FP32 (FP64 for FP64 parameters), captured at construction or
    ``add_param_group``, and saved with the optimizer. Frozen tensors are never
    updated or snapshotted. Pass only currently trainable parameters when using
    gradual unfreezing, then add newly unfrozen tensors as a new group.

    Capturable, differentiable, and fused steps are unsupported. Use ordinary
    AMP/GradScaler with the non-fused optimizer. With both controls disabled,
    this follows AdamW exactly and stores no anchors.
    """

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        *,
        l2sp_weight: float = 1e-3,
        max_relative_drift: float | None = 0.1,
        min_reference_norm: float = 1.0,
        maximize: bool = False,
        foreach: bool | None = None,
    ) -> None:
        self._anchors: dict[Tensor, Tensor] = {}
        self._transfer_defaults: dict[str, Any] = {
            "l2sp_weight": l2sp_weight,
            "max_relative_drift": max_relative_drift,
            "min_reference_norm": min_reference_norm,
        }
        self._validate_group(self._transfer_defaults)
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            maximize=maximize,
            foreach=foreach,
        )
        self.defaults.update(self._transfer_defaults)

    @staticmethod
    def _validate_group(group: dict[str, Any]) -> None:
        """Reject invalid protection settings and unsupported Adam execution modes."""
        for name in ("l2sp_weight", "max_relative_drift", "min_reference_norm"):
            value = group[name]
            if name == "max_relative_drift" and value is None:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not isfinite(value)
                or value < 0
                or (name == "min_reference_norm" and value == 0)
            ):
                raise ValueError(f"Invalid {name}: {value!r}")
        if any(
            group.get(name, False) for name in ("capturable", "differentiable", "fused")
        ):
            raise ValueError(
                "TransferAdamW requires non-fused, non-capturable, nondifferentiable steps"
            )

    @staticmethod
    def _enabled(group: dict[str, Any]) -> bool:
        return bool(group["l2sp_weight"] or group["max_relative_drift"] is not None)

    @staticmethod
    def _anchor_dtype(parameter: Tensor) -> torch.dtype:
        return torch.float64 if parameter.dtype == torch.float64 else torch.float32

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        """Add a normal parameter group and capture its trainable starting point."""
        group = dict(param_group)
        for name, value in self._transfer_defaults.items():
            group.setdefault(name, value)
        self._validate_group(group)
        params = group["params"]
        group["params"] = [params] if isinstance(params, Tensor) else list(params)
        if len({id(parameter) for parameter in group["params"]}) != len(
            group["params"]
        ):
            raise ValueError("Duplicate parameters in a transfer optimizer group")
        for parameter in group["params"]:
            if (
                isinstance(parameter, Tensor)
                and parameter.requires_grad
                and not parameter.is_floating_point()
            ):
                raise ValueError(
                    "TransferAdamW supports real floating-point trainable tensors"
                )
        super().add_param_group(group)
        if self._enabled(group):
            for parameter in group["params"]:
                if parameter.requires_grad:
                    self._anchors[parameter] = (
                        parameter.detach()
                        .to(dtype=self._anchor_dtype(parameter))
                        .clone()
                    )

    def _anchor(self, parameter: Tensor) -> Tensor:
        """Move restored anchors lazily when Lightning moves optimizer state/devices."""
        if parameter not in self._anchors:
            raise RuntimeError(
                "Missing transfer anchor; add newly trainable tensors as a new parameter group"
            )
        anchor = self._anchors[parameter].to(
            device=parameter.device, dtype=self._anchor_dtype(parameter)
        )
        self._anchors[parameter] = anchor
        return anchor

    @torch.no_grad()
    def step(self, closure: Callable[[], Any] | None = None) -> Any:
        """Apply the anchored gradient, AdamW update, and optional drift projection."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        original_grads: dict[Tensor, Tensor] = {}
        updated: list[tuple[Tensor, dict[str, Any]]] = []
        try:
            for group in self.param_groups:
                self._validate_group(group)
                for parameter in group["params"]:
                    gradient = parameter.grad
                    if gradient is None:
                        continue
                    original_grads[parameter] = gradient
                    if not parameter.requires_grad:
                        parameter.grad = None
                        continue
                    if gradient.is_sparse:
                        raise RuntimeError(
                            "TransferAdamW does not support sparse gradients"
                        )
                    if self._enabled(group):
                        anchor = self._anchor(parameter)
                        if group["l2sp_weight"]:
                            delta = (parameter.detach().to(anchor.dtype) - anchor).to(
                                gradient.dtype
                            )
                            sign = -1.0 if group["maximize"] else 1.0
                            parameter.grad = gradient.add(
                                delta, alpha=sign * group["l2sp_weight"]
                            )
                        updated.append((parameter, group))
            super().step()
            for parameter, group in updated:
                if group["max_relative_drift"] is not None:
                    anchor = self._anchor(parameter)
                    delta = parameter.detach().to(anchor.dtype) - anchor
                    radius = group["max_relative_drift"] * anchor.norm().clamp_min(
                        group["min_reference_norm"]
                    )
                    scale = (
                        radius / delta.norm().clamp_min(torch.finfo(delta.dtype).tiny)
                    ).clamp(max=1.0)
                    parameter.copy_(anchor + delta * scale)
        finally:
            for parameter, gradient in original_grads.items():
                parameter.grad = gradient
        return loss

    def state_dict(self) -> dict[str, Any]:
        """Serialize Adam state and immutable anchors with normal parameter IDs."""
        result: dict[str, Any] = super().state_dict()
        anchors = {
            identifier: self._anchors[parameter]
            for live, saved in zip(
                self.param_groups, result["param_groups"], strict=True
            )
            for parameter, identifier in zip(
                live["params"], saved["params"], strict=True
            )
            if parameter in self._anchors
        }
        result["transfer_state"] = {"version": 1, "anchors": anchors}
        return result

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore the original anchors; never silently re-anchor on resume."""
        saved_groups = state_dict["param_groups"]
        transfer = state_dict.get("transfer_state", {"version": 1, "anchors": {}})
        if transfer["version"] != 1:
            raise ValueError("Unsupported transfer optimizer checkpoint version")
        anchors = transfer["anchors"]
        restored: dict[Tensor, Tensor] = {}
        known: set[int] = set()
        groups: list[dict[str, Any]] = []
        for live, saved in zip(self.param_groups, saved_groups, strict=True):
            group = {**self._transfer_defaults, **saved}
            self._validate_group(group)
            groups.append(group)
            for parameter, identifier in zip(
                live["params"], saved["params"], strict=True
            ):
                known.add(identifier)
                anchor = anchors.get(identifier)
                if anchor is None and (
                    not parameter.requires_grad or not self._enabled(group)
                ):
                    continue
                if (
                    not isinstance(anchor, Tensor)
                    or anchor.shape != parameter.shape
                    or not anchor.is_floating_point()
                    or not bool(torch.isfinite(anchor).all())
                ):
                    raise ValueError(
                        "Missing or incompatible transfer anchor in checkpoint; resume with the original optimizer"
                    )
                restored[parameter] = (
                    anchor.detach()
                    .to(device=parameter.device, dtype=self._anchor_dtype(parameter))
                    .clone()
                )
        if not set(anchors).issubset(known):
            raise ValueError(
                "Transfer checkpoint anchors do not match the trainable parameters"
            )
        super().load_state_dict({**state_dict, "param_groups": groups})
        self._anchors = restored

    @torch.no_grad()
    def drift_metrics(self) -> dict[str, float]:
        """Measure drift and L2-SP penalty; intended for epoch-level monitoring."""
        squared_distance = 0.0
        penalty = 0.0
        maximum = 0.0
        for group in self.param_groups:
            if not self._enabled(group):
                continue
            for parameter in group["params"]:
                if not parameter.requires_grad:
                    continue
                anchor = self._anchor(parameter)
                distance = float((parameter.detach().to(anchor.dtype) - anchor).norm())
                denominator = max(float(anchor.norm()), group["min_reference_norm"])
                squared_distance += distance**2
                penalty += 0.5 * group["l2sp_weight"] * distance**2
                maximum = max(maximum, distance / denominator)
        return {
            "distance_l2": sqrt(squared_distance),
            "max_relative_drift": maximum,
            "l2sp_penalty": penalty,
        }
