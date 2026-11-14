"""Save resumable checkpoints before an external rental budget deadline."""

import json
import time
from pathlib import Path
from typing import Any

from lightning import Callback


class RentalQuotaReached(RuntimeError):
    """Signal a controlled stop to the rental stage driver."""


class RentalGuard(Callback):
    """Check quota between batches, publish progress, and preserve interrupted state.

    The independent local supervisor controls billing. This callback stops GPU
    work early enough to upload/download checkpoints before instance destruction.
    """

    def __init__(
        self, stop_file: str, deadline_unix: float | None, progress_every: int = 20
    ):
        if (deadline_unix is not None and deadline_unix <= 0) or progress_every < 1:
            raise ValueError("A positive deadline and progress interval are required")
        self.stop_file = Path(stop_file)
        self.deadline_unix = deadline_unix
        self.progress_every = progress_every
        self.last_step = -1

    def _check(self) -> None:
        if self.stop_file.exists() or (
            self.deadline_unix is not None and time.time() >= self.deadline_unix
        ):
            raise RentalQuotaReached("Rental quota reached; preserving training state")

    def _progress(self, trainer: Any, module: Any, status: str) -> None:
        output = Path(trainer.default_root_dir)
        output.mkdir(parents=True, exist_ok=True)
        metrics = {}
        for key, value in trainer.callback_metrics.items():
            try:
                metrics[key] = float(value)
            except (TypeError, ValueError):
                continue
        record = {
            "status": status,
            "global_step": trainer.global_step,
            "epoch": trainer.current_epoch,
            "unix_time": time.time(),
            "metrics": metrics,
        }
        temporary = output / "progress.json.tmp"
        temporary.write_text(json.dumps(record, indent=2), encoding="utf-8")
        temporary.replace(output / "progress.json")
        with (output / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        module.phase_timing.write()
        if status == "training":
            timings = {
                f"timing/{phase}/mean_seconds": values["mean_seconds"]
                for phase, values in module.phase_timing.summary()["phases"].items()
            }
            for logger in trainer.loggers:
                logger.log_metrics(timings, step=trainer.global_step)

    def on_train_batch_start(self, trainer: Any, pl_module: Any, *args: Any) -> None:
        self._check()

    def on_validation_batch_start(
        self, trainer: Any, pl_module: Any, *args: Any
    ) -> None:
        self._check()

    def on_test_batch_start(self, trainer: Any, pl_module: Any, *args: Any) -> None:
        self._check()

    def on_train_batch_end(self, trainer: Any, pl_module: Any, *args: Any) -> None:
        if (
            trainer.global_step - self.last_step >= self.progress_every
            or self.last_step < 0
        ):
            self._progress(trainer, pl_module, "training")
            self.last_step = trainer.global_step

    def on_exception(
        self, trainer: Any, pl_module: Any, exception: BaseException
    ) -> None:
        if not trainer.optimizers:
            self._progress(trainer, pl_module, "failed_before_optimizer")
            return
        if trainer.testing:
            # Testing may have reloaded best weights while retaining the last
            # optimizer state. Preserve the existing fit checkpoints instead.
            self._progress(trainer, pl_module, "test_interrupted")
            return
        checkpoint = Path(trainer.default_root_dir) / "checkpoints/interrupted.ckpt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(checkpoint)
        self._progress(
            trainer,
            pl_module,
            "quota_stopped" if isinstance(exception, RentalQuotaReached) else "failed",
        )
        # This checkpoint is outside ModelCheckpoint, so explicitly archive it.
        for logger in trainer.loggers:
            if hasattr(logger, "experiment") and hasattr(
                logger.experiment, "log_artifact"
            ):
                import wandb

                artifact = wandb.Artifact(
                    f"interrupted-{logger.experiment.id}", type="model"
                )
                artifact.add_file(str(checkpoint), name="model.ckpt")
                logger.experiment.log_artifact(artifact, aliases=["latest"])


class TrainingProgress(RentalGuard):
    """Record progress and checkpoint manual stops, without a budget or time limit."""

    def __init__(self, stop_file: str, progress_every: int = 20):
        super().__init__(stop_file, deadline_unix=None, progress_every=progress_every)
