"""Publish downloaded training telemetry to W&B without putting keys on a rental."""

import json
import os
from pathlib import Path
from typing import Any


class WandbRelay:
    """Relay complete progress records and checkpoint artifacts from a local backup."""

    def __init__(self, root: Path, group: str, entity: str):
        from src.utils.credentials import environment_secret

        key = environment_secret("WANDB_API_KEY")
        if not key:
            raise ValueError("WANDB_API_KEY is required locally for the W&B relay")
        os.environ["WANDB_API_KEY"] = key
        os.environ["WANDB_DISABLE_CODE"] = "true"
        os.environ["WANDB_DISABLE_GIT"] = "true"
        os.environ["WANDB_CACHE_DIR"] = str(root / "wandb-cache")
        os.environ["WANDB_DATA_DIR"] = str(root / "wandb-data")
        self.root, self.group, self.entity = root, group, entity
        self.run: Any = None
        self.stage: str | None = None
        self.steps: dict[str, int] = {}
        self.completed: set[str] = set()
        self.evaluations_seen: dict[str, int] = {}

    def _start(self, stage: str, directory: Path) -> None:
        if self.stage == stage:
            return
        self.finish()
        import wandb
        from omegaconf import OmegaConf

        cfg = OmegaConf.to_container(OmegaConf.load(directory / "resolved_config.yaml"))
        self.run = wandb.init(
            project="Machine Translation",
            entity=self.entity,
            id=f"{self.group}-{stage}",
            resume="allow",
            group=self.group,
            name=f"Qwen2.5-7B-LoRA-{stage}",
            job_type=stage,
            dir=str(self.root),
            config={
                "training": cfg,
                "telemetry": "local relay; API key never sent to rental",
            },
        )
        from src.utils.translation_telemetry import define_chart_axes

        define_chart_axes(self.run)
        self.stage = stage
        (self.root / f"wandb_{stage}_url.txt").write_text(
            self.run.url, encoding="utf-8"
        )

    def sync(self) -> None:
        for stage in ("sft", "rl"):
            directory = self.root / "checkpoints_and_logs" / stage
            progress = directory / "diagnostics/updates.jsonl"
            if not progress.is_file():
                progress = directory / "progress.jsonl"
            history = directory / "translation_metrics/evaluation_history.jsonl"
            if stage in self.completed or (
                not progress.is_file() and not history.is_file()
            ):
                continue
            records = []
            lines = (
                progress.read_text(encoding="utf-8").splitlines(keepends=True)
                if progress.is_file()
                else []
            )
            for line in lines:
                if not line.endswith("\n"):
                    continue
                record = json.loads(line)
                if record["global_step"] > self.steps.get(stage, -1):
                    records.append(record)
            evaluations = (
                history.read_text(encoding="utf-8").splitlines(keepends=True)
                if history.is_file()
                else []
            )
            new_evaluations = []
            for index in range(self.evaluations_seen.get(stage, 0), len(evaluations)):
                if not evaluations[index].endswith("\n"):
                    break
                new_evaluations.append(json.loads(evaluations[index]))
                self.evaluations_seen[stage] = index + 1
            if not records and not new_evaluations and self.stage != stage:
                continue
            self._start(stage, directory)
            for record in records:
                self.run.log(
                    {"trainer/global_step": record["global_step"], **record["metrics"]}
                )
                self.steps[stage] = record["global_step"]
            for record in new_evaluations:
                self.run.log(
                    {"trainer/global_step": record["global_step"], **record["metrics"]}
                )
                self.steps.setdefault(stage, record["global_step"])
            summary = directory / "phase_timing/summary.json"
            if summary.is_file():
                phases = json.loads(summary.read_text(encoding="utf-8"))["phases"]
                self.run.log(
                    {
                        "trainer/global_step": self.steps[stage],
                        **{
                            f"timing/{phase}/mean_seconds": stats["mean_seconds"]
                            for phase, stats in phases.items()
                        },
                    }
                )
            result = directory / "result.json"
            if result.is_file():
                data = json.loads(result.read_text(encoding="utf-8"))
                self.run.log(
                    {"trainer/global_step": data["global_step"], **data["metrics"]}
                )
                self.run.summary["stage_status"] = "completed"
                self.completed.add(stage)
                self.finish()

    def finish(self, status: str | None = None) -> None:
        if self.run is not None:
            if status:
                self.run.summary["rental_status"] = status
            self.run.finish()
            self.run, self.stage = None, None
