"""Append-only scalar histories and W&B line-chart axes for manual optimization."""

import json
from pathlib import Path
from typing import Any

import torch


def append_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")


def scalars(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(value.detach().cpu())
        if isinstance(value, torch.Tensor)
        else float(value)
        for key, value in metrics.items()
    }


def define_chart_axes(run: Any) -> None:
    """W&B scalar histories use optimizer updates instead of wall time or epochs."""
    run.define_metric("trainer/global_step")
    run.define_metric("*", step_metric="trainer/global_step")
    for namespace in (
        "grpo/*",
        "rollout/*",
        "reward/*",
        "reward_conflict/*",
        "direction/*",
        "optimizer/*",
        "val/*",
        "test/*",
    ):
        run.define_metric(namespace, step_metric="trainer/global_step")
