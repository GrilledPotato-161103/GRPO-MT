"""Measure complete self-play batch latency, including external feedback wait."""

import json
import statistics
import time
from pathlib import Path
from typing import Any

import lightning as L
import torch


class SelfPlayTiming(L.Callback):
    """Write throughput evidence for pricing a subsequent training run."""

    def __init__(self, output_file: str, warmup_batches: int = 1) -> None:
        super().__init__()
        if warmup_batches < 0:
            raise ValueError("warmup_batches must be nonnegative")
        self.output_file = Path(output_file)
        self.warmup_batches = warmup_batches
        self.durations: list[float] = []
        self.started = 0.0
        self.batch_started = 0.0

    def _synchronize(self, module: L.LightningModule) -> None:
        if module.device.type == "cuda":
            torch.cuda.synchronize(module.device)

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Start the wall clock and reset peak GPU memory accounting."""
        self.durations.clear()
        self._synchronize(pl_module)
        if pl_module.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(pl_module.device)
        self.started = time.perf_counter()

    def on_train_batch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Start a synchronized timer before both role rollouts and feedback."""
        self._synchronize(pl_module)
        self.batch_started = time.perf_counter()

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Finish timing all optimizer iterations for this rollout batch."""
        self._synchronize(pl_module)
        self.durations.append(time.perf_counter() - self.batch_started)

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Save measured batch latency and metadata without any API credentials."""
        measured = self.durations[self.warmup_batches :] or self.durations
        backend = getattr(getattr(pl_module, "feedback", None), "backend", None)
        report = {
            "device": str(pl_module.device),
            "gpu": torch.cuda.get_device_name(pl_module.device)
            if pl_module.device.type == "cuda"
            else None,
            "training_loop_seconds": time.perf_counter() - self.started,
            "batches": len(self.durations),
            "batch_seconds": list(self.durations),
            "timed_batches_after_warmup": len(measured),
            "mean_seconds_per_batch": statistics.mean(measured) if measured else None,
            "median_seconds_per_batch": statistics.median(measured)
            if measured
            else None,
            "optimizer_steps": trainer.global_step,
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(
                pl_module.device
            )
            if pl_module.device.type == "cuda"
            else 0,
            "feedback_requests": getattr(backend, "requests", None),
            "feedback_prompt_tokens": getattr(backend, "prompt_tokens", None),
            "feedback_completion_tokens": getattr(backend, "completion_tokens", None),
        }
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.output_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
