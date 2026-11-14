"""Measure generation and joint proposer/solver backward memory on a CUDA GPU.

The shape stress test uses synthetic teacher-forced completions and fixed rewards.
It measures capacity, not translation quality or an on-policy learning result.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import lightning as L
import torch
from torch.utils.data import DataLoader

from src.data.self_play_datamodule import TopicDataset
from src.feedback.judge import TranslationJudge
from src.models.components.grpo import group_advantages
from src.models.self_play_module import SelfPlayGRPOModule


class UnusedFeedback:
    """No network feedback is needed for the explicit memory-capacity probe."""

    async def generate(self, conversations: list[Any]) -> list[str]:
        raise RuntimeError("Shape profiling must not call a real or simulated judge")


class ShapeProbe(SelfPlayGRPOModule):
    """Exercise the real joint GRPO optimizer at fixed, fully populated token shapes."""

    def stress_prompts(self, count: int) -> list[str]:
        """Fill the source cap rather than benchmark unusually short inputs."""
        return [
            "Translate English into Vietnamese. Source: "
            + "translation " * self.hparams["max_source_length"]
            for _ in range(count)
        ]

    @torch.no_grad()
    def _shape_rollout(
        self, contexts: int, group_size: int, completion_length: int
    ) -> dict[str, Any]:
        self.policy.eval()
        inputs = {
            key: value.repeat_interleave(group_size, 0)
            for key, value in self._tokenize_prompts(
                self.stress_prompts(contexts)
            ).items()
        }
        token_id = self.tokenizer.encode(" hello", add_special_tokens=False)[0]
        completions = torch.full(
            (inputs["input_ids"].shape[0], completion_length),
            token_id,
            device=self.device,
            dtype=torch.long,
        )
        alternate_id = self.tokenizer.encode(" world", add_special_tokens=False)[0]
        if alternate_id == token_id:
            alternate_id = (token_id + 1) % self.policy.config.vocab_size
        completions[1::2] = alternate_id
        sequences = torch.cat([inputs["input_ids"], completions], dim=1)
        mask = torch.ones_like(completions, dtype=torch.bool)
        rewards = (
            torch.arange(len(sequences), device=self.device)
            .remainder(group_size)
            .float()
        )
        return {
            "inputs": inputs,
            "sequences": sequences,
            "mask": mask,
            "old_logps": self._logps(self.policy, inputs, sequences, mask),
            "reference_logps": self._reference_logps(inputs, sequences, mask),
            "rewards": rewards,
            "advantages": group_advantages(rewards, group_size),
        }

    def _self_play_rollouts(
        self, topics: list[str]
    ) -> list[tuple[str, dict[str, Any], float]]:
        self.last_self_play = {
            "valid_tasks": 2,
            "memory_size": 0,
            "solver_quality": 0.5,
            "solver_tasks": 2,
        }
        return [
            (
                "proposer",
                self._shape_rollout(1, 2, self.hparams["proposal_max_new_tokens"]),
                1.0,
            ),
            (
                "solver",
                self._shape_rollout(
                    2, self.hparams["group_size"], self.hparams["max_new_tokens"]
                ),
                1.0,
            ),
        ]


def profile(
    model: str,
    mode: str,
    output: Path,
    source_tokens: int,
    completion_tokens: int,
    iterations: int,
) -> None:
    """Record real peak memory for one full stress batch, including Adam state."""
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    if (
        mode not in {"lora", "qlora"}
        or min(source_tokens, completion_tokens, iterations) < 1
    ):
        raise ValueError("Invalid adapter profile settings")
    output.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "model": model,
        "mode": mode,
        "gpu": torch.cuda.get_device_name(),
        "device_total_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
        "source_tokens": source_tokens,
        "completion_tokens": completion_tokens,
        "proposal_candidates": 2,
        "solver_sources": 2,
        "solver_candidates_per_source": 2,
        "logprob_batch_size": 1,
        "gradient_checkpointing": True,
        "reference": "shared frozen base with adapters disabled",
        "workload": "synthetic full-shape memory stress, not MT quality training",
        "status": "running",
    }
    started = time.perf_counter()
    try:
        module = ShapeProbe(
            model_name_or_path=model,
            feedback=TranslationJudge(UnusedFeedback()),
            load_dtype="bfloat16",
            quantization="nf4" if mode == "qlora" else None,
            lora={
                "enabled": True,
                "r": 8,
                "alpha": 16,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            },
            max_source_length=source_tokens,
            max_new_tokens=completion_tokens,
            proposal_max_new_tokens=completion_tokens,
            proposal_group_size=2,
            group_size=2,
            num_iterations=iterations,
            gradient_checkpointing=True,
            logprob_batch_size=1,
        ).to("cuda")
        report["model_load_seconds"] = time.perf_counter() - started
        trainable, total = module.policy.get_nb_trainable_parameters()
        report.update(
            base_revision=module.hparams["base_revision"],
            trainable_parameters=trainable,
            total_parameters_with_adapters=total,
            separate_reference_parameters=0,
            loaded_allocated_gib=torch.cuda.memory_allocated() / 2**30,
        )
        if module.reference is not None:
            raise AssertionError(
                "Adapter profile unexpectedly allocated a separate reference"
            )
        torch.cuda.reset_peak_memory_stats()
        inputs = module._tokenize_prompts(module.stress_prompts(1))
        generation = {
            key: value.repeat_interleave(4, 0) for key, value in inputs.items()
        }
        with torch.no_grad():
            module.policy.eval()
            module.policy.generate(
                **generation,
                generation_config=module.generation_config,
                min_new_tokens=completion_tokens,
            )
        torch.cuda.synchronize()
        report["generation_peak_allocated_gib"] = (
            torch.cuda.max_memory_allocated() / 2**30
        )
        del inputs, generation
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        before = {
            name: p.detach().cpu().clone()
            for name, p in module.policy.named_parameters()
            if p.requires_grad
        }
        trainer = L.Trainer(
            accelerator="gpu",
            devices=1,
            precision="bf16-mixed",
            max_epochs=1,
            limit_val_batches=0,
            num_sanity_val_steps=0,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        step_started = time.perf_counter()
        trainer.fit(
            module,
            train_dataloaders=DataLoader(
                TopicDataset(["shape stress"], 1), batch_size=1
            ),
        )
        torch.cuda.synchronize()
        report["train_seconds"] = time.perf_counter() - step_started
        report["train_peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        report["train_peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
        report["optimizer_steps"] = trainer.global_step
        report["adapter_updated"] = any(
            not torch.equal(
                value, dict(module.policy.named_parameters())[name].detach().cpu()
            )
            for name, value in before.items()
        )
        if not report["adapter_updated"]:
            raise AssertionError("No adapter parameter changed")
        report["status"] = "fits"
    except torch.OutOfMemoryError:
        report["status"] = "out_of_memory"
        report["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        raise
    except Exception as error:
        report["status"] = "error"
        report["error_type"] = type(error).__name__
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        (output / "memory_profile.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )


def main() -> None:
    """Profile a single mode/process so allocator state cannot leak across comparisons."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--mode", choices=["lora", "qlora"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-tokens", type=int, default=512)
    parser.add_argument("--completion-tokens", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=2)
    args = parser.parse_args()
    profile(
        args.model,
        args.mode,
        args.output,
        args.source_tokens,
        args.completion_tokens,
        args.iterations,
    )


if __name__ == "__main__":
    main()
