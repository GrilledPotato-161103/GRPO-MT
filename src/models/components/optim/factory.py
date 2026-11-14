"""Plain-data optimizer factory configuration for Lightning checkpoints."""

from collections.abc import Callable
from functools import partial
from typing import Any

from hydra.utils import instantiate
from torch import Tensor
from torch.optim import Optimizer


def _checkpoint_value(value: Any) -> Any:
    """Keep optimizer arguments compatible with restricted torch checkpoint loading."""
    if value is None or isinstance(value, (bool, int, float, str, Tensor)):
        return value
    if isinstance(value, (list, tuple)):
        return type(value)(_checkpoint_value(item) for item in value)
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _checkpoint_value(item) for key, item in value.items()}
    raise TypeError(
        "Optimizer factory arguments must contain tensors or plain checkpoint data"
    )


def resolve_optimizer_factory(
    optimizer: Callable[..., Optimizer] | dict[str, Any] | None,
) -> tuple[Callable[..., Optimizer] | None, dict[str, Any] | None]:
    """Resolve a Hydra spec or importable partial without pickling callables.

    The returned configuration reconstructs the factory, not a live optimizer.
    Parameter groups and moments remain in the standard optimizer state dict.
    """
    if optimizer is None:
        return None, None
    if isinstance(optimizer, dict):
        config = _checkpoint_value(optimizer)
        if config.get("_partial_") is not True:
            raise ValueError("Optimizer config requires _partial_: true")
        factory: Any = instantiate(config)
    else:
        factory = optimizer
        target = optimizer.func if isinstance(optimizer, partial) else optimizer
        module = getattr(target, "__module__", None)
        name = getattr(target, "__qualname__", None)
        if not module or not name or "<" in name:
            raise ValueError(
                "Use an importable optimizer function/class or functools.partial"
            )
        arguments = list(optimizer.args) if isinstance(optimizer, partial) else []
        keywords = (optimizer.keywords or {}) if isinstance(optimizer, partial) else {}
        config = {
            **_checkpoint_value(keywords),
            "_target_": f"{module}.{name}",
            "_partial_": True,
            "_args_": _checkpoint_value(arguments),
        }
    if not callable(factory):
        raise TypeError("optimizer must be a callable optimizer factory")
    return factory, config
