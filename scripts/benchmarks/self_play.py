"""Benchmark a small pretrained self-play policy; mock feedback is explicit."""

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import httpx
import lightning as L
import torch

from src.data.self_play_datamodule import SelfPlayDataModule
from src.feedback.backends import APIChatBackend
from src.feedback.judge import TranslationJudge
from src.models.self_play_module import SelfPlayGRPOModule
from src.utils.self_play_timing import SelfPlayTiming


class CurriculumAudit(L.Callback):
    """Count validated proposals so an empty curriculum cannot pass the benchmark."""

    def __init__(self) -> None:
        self.valid_tasks = 0
        self.batches = 0
        self.solver_tasks_per_batch: list[int] = []

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if not isinstance(pl_module, SelfPlayGRPOModule):
            raise TypeError("CurriculumAudit requires a self-play module")
        self.valid_tasks += int(pl_module.last_self_play["valid_tasks"])
        self.solver_tasks_per_batch.append(
            int(pl_module.last_self_play["solver_tasks"])
        )
        self.batches += 1


def benchmark(
    output: Path,
    model: str,
    steps: int,
    mock_feedback: bool,
    save_checkpoint: bool = False,
) -> None:
    """Measure CUDA policy work, checkpointing, and independent held-out evaluation."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing a CPU fallback")
    if steps < 2:
        raise ValueError("Use at least two batches for timing after warmup")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    L.seed_everything(42)
    counts = {"validation_calls": 0, "scoring_calls": 0}
    transport: httpx.AsyncBaseTransport | None = None
    if mock_feedback:

        def respond(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            text = json.loads(payload["messages"][-1]["content"])
            if "candidate" in text:
                index = counts["scoring_calls"]
                counts["scoring_calls"] += 1
                score = 1.0 if index % 2 else 0.2
            else:
                counts["validation_calls"] += 1
                score = 1.0
            content = json.dumps(
                {
                    "valid": True,
                    "score": score,
                    "feedback": "Synthetic transport test; not a quality judgment",
                }
            )
            return httpx.Response(
                200, json={"choices": [{"message": {"content": content}}]}
            )

        transport = httpx.MockTransport(respond)
    key_name = "GRPO_BENCHMARK_TEST_KEY" if mock_feedback else "CLAUDE_AUTH_KEY"
    backend = APIChatBackend(api_key_env=key_name, transport=transport)
    old_test_key = os.environ.get(key_name)
    if mock_feedback:
        os.environ[key_name] = "synthetic-test-credential"
    report: dict[str, Any] = {
        "model": model,
        "gpu": torch.cuda.get_device_name(),
        "feedback": "mock HTTP transport" if mock_feedback else "live OpenRouter",
        "steps_requested": steps,
        "status": "running",
    }
    started = time.perf_counter()
    try:
        module = SelfPlayGRPOModule(
            feedback=TranslationJudge(backend),
            model_name_or_path=model,
            group_size=2,
            proposal_group_size=2,
            num_iterations=2,
            max_source_length=512,
            max_new_tokens=32,
            proposal_max_new_tokens=96,
            memory_examples=1,
        )
        report["model_load_seconds"] = time.perf_counter() - started
        root = Path(__file__).resolve().parents[2]
        data = SelfPlayDataModule(
            topics=["daily life", "travel", "science"],
            steps_per_epoch=steps,
            val_file=str(root / "data/example_val.jsonl"),
            test_file=str(root / "data/example_test.jsonl"),
        )
        audit = CurriculumAudit()
        trainer = L.Trainer(
            accelerator="gpu",
            devices=1,
            precision="bf16-mixed",
            max_epochs=1,
            callbacks=[SelfPlayTiming(str(output / "timing.json")), audit],
            default_root_dir=output,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
        )
        trainer.fit(module, datamodule=data)
        report["valid_tasks"] = audit.valid_tasks
        report["solver_tasks_per_batch"] = list(audit.solver_tasks_per_batch)
        report["rollout_config"] = {
            "topic_batch_size": 1,
            "proposal_group_size": 2,
            "solver_group_size": 2,
            "proposal_max_new_tokens": 96,
            "solver_max_new_tokens": 32,
            "num_iterations": 2,
            "precision": "bf16-mixed",
        }
        timing = json.loads((output / "timing.json").read_text())
        full_durations = [
            seconds
            for i, seconds in enumerate(timing["batch_seconds"])
            if i > 0 and audit.solver_tasks_per_batch[i] > 0
        ]
        report["full_solver_batches_timed"] = len(full_durations)
        report["full_solver_batch_mean_seconds"] = (
            statistics.mean(full_durations) if full_durations else None
        )
        report["optimizer_steps"] = trainer.global_step
        trainer.test(module, datamodule=data)
        report["metrics"] = {
            key: float(value.cpu()) for key, value in trainer.callback_metrics.items()
        }
        if save_checkpoint:
            trainer.save_checkpoint(output / "benchmark.ckpt")
        if audit.valid_tasks == 0:
            raise RuntimeError(
                "No valid proposals were produced; benchmark did not exercise the solver"
            )
        if counts["scoring_calls"] == 0 and mock_feedback:
            raise RuntimeError("Mock transport did not receive solver scoring requests")
        report["status"] = "passed"
    except Exception:
        report["status"] = "failed"
        raise
    finally:
        if mock_feedback:
            if old_test_key is None:
                os.environ.pop(key_name, None)
            else:
                os.environ[key_name] = old_test_key
        report["elapsed_seconds"] = time.perf_counter() - started
        report["feedback_calls"] = dict(counts)
        (output / "benchmark.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )


def main() -> None:
    """Run a finite pilot using the selected pretrained policy and feedback mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("logs/self_play_benchmark"))
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--mock-feedback", action="store_true")
    parser.add_argument("--save-checkpoint", action="store_true")
    args = parser.parse_args()
    benchmark(
        args.output, args.model, args.steps, args.mock_feedback, args.save_checkpoint
    )


if __name__ == "__main__":
    main()
