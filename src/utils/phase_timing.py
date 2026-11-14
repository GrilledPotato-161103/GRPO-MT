"""CUDA-synchronized phase timings, work counts, and policy throughput reports."""

import json
import statistics
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch


class PhaseTiming:
    """Record elapsed time per phase without confusing CUDA launch and execution time."""

    def __init__(self, enabled: bool = False, synchronize_cuda: bool = True) -> None:
        self.enabled, self.synchronize_cuda = enabled, synchronize_cuda
        self.events: list[dict[str, Any]] = []
        self.output: Path | None = None

    def configure(self, output: Path) -> None:
        """Choose the run-local destination once; secrets and input text are excluded."""
        if self.enabled and self.output is None:
            output.mkdir(parents=True, exist_ok=True)
            self.output = output

    def _sync(self, device: torch.device) -> None:
        if self.synchronize_cuda and device.type == "cuda":
            torch.cuda.synchronize(device)

    def record(self, phase: str, seconds: float, **work: Any) -> None:
        """Store an externally measured phase with counts, never user text."""
        if not self.enabled:
            return
        event = {"phase": phase, "seconds": seconds, **work}
        self.events.append(event)
        if self.output is not None:
            with (self.output / "events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event) + "\n")

    @contextmanager
    def measure(
        self, phase: str, device: torch.device, **work: Any
    ) -> Iterator[dict[str, Any]]:
        """Measure a phase; fill returned work counts after dynamic generation if needed."""
        if not self.enabled:
            yield work
            return
        self._sync(device)
        started = time.perf_counter()
        succeeded = False
        try:
            yield work
            succeeded = True
        finally:
            self._sync(device)
            self.record(
                phase,
                time.perf_counter() - started,
                device=str(device),
                succeeded=succeeded,
                **work,
            )

    def summary(self) -> dict[str, Any]:
        """Aggregate phase latency and work/second with explicit measurement semantics."""
        phases: dict[str, Any] = {}
        for phase in sorted({event["phase"] for event in self.events}):
            rows = [
                event
                for event in self.events
                if event["phase"] == phase and event.get("succeeded", True)
            ]
            if not rows:
                continue
            times = sorted(float(row["seconds"]) for row in rows)
            total = sum(times)
            result: dict[str, Any] = {
                "calls": len(times),
                "total_seconds": total,
                "mean_seconds": statistics.mean(times),
                "median_seconds": statistics.median(times),
                "p95_seconds": times[min(len(times) - 1, int(0.95 * len(times)))],
                "min_seconds": times[0],
                "max_seconds": times[-1],
            }
            for unit in ("input_tokens", "output_tokens", "samples"):
                count = sum(int(row.get(unit, 0)) for row in rows)
                if count:
                    result[unit] = count
                    result[f"{unit}_per_second"] = count / total if total else 0.0
            phases[phase] = result
        return {
            "cuda_synchronized": self.synchronize_cuda,
            "units": "seconds",
            "backward_includes_checkpoint_recomputation": True,
            "overlapping_parent_child_phases_must_not_be_summed": True,
            "events": len(self.events),
            "phases": phases,
        }

    def write(self) -> None:
        """Write a summary; raw events remain available after interrupted runs."""
        if self.enabled and self.output is not None:
            (self.output / "summary.json").write_text(
                json.dumps(self.summary(), indent=2), encoding="utf-8"
            )
