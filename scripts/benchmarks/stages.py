"""Benchmark synthetic bilingual SFT, paired GRPO, and shared-policy self-play on CUDA."""

import argparse
import csv
import gc
import json
import time
from functools import partial
from pathlib import Path
from typing import Any, cast

import lightning as L
import torch
from transformers import AutoTokenizer

from scripts.data.prepare_stages import prepare
from src.data.self_play_datamodule import SelfPlayDataModule
from src.data.translation_datamodule import TranslationDataModule
from src.feedback.judge import TranslationJudge, TranslationTask
from src.models.components.optim import TransferAdamW
from src.models.components.translation_reward import TranslationReward
from src.models.grpo_module import GRPOModule
from src.models.self_play_module import SelfPlayGRPOModule
from src.models.sft_module import TranslationSFTModule, export_sft


class SyntheticJudge:
    """Explicitly mocked feedback; policy generation and gradients remain real."""

    async def generate(self, conversations: list[Any]) -> list[str]:
        return [
            json.dumps(
                {
                    "valid": True,
                    "score": 0.2 if index % 2 == 0 else 0.9,
                    "feedback": "synthetic benchmark feedback",
                }
            )
            for index in range(len(conversations))
        ]


def _pairs(path: Path, start: int, count: int) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["en", "vi"])
        writer.writeheader()
        writer.writerows(
            {
                "en": f"The blue book has {index} pages.",
                "vi": f"Cuốn sách màu xanh có {index} trang.",
            }
            for index in range(start, start + count)
        )


def _release() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def _trainer(output: Path, steps: int) -> L.Trainer:
    return L.Trainer(
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        max_epochs=1,
        limit_train_batches=steps,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=output,
    )


def benchmark(
    output: Path,
    work: Path,
    model: str,
    mode: str,
    steps: int = 3,
    comet_device: str = "cuda:0",
    comet_model: str = "Unbabel/wmt22-cometkiwi-da",
) -> dict[str, Any]:
    """Exercise real optimizers/COMET; all examples are synthetic and public-safe."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the rental benchmark")
    if mode not in {"full", "lora", "qlora"} or steps < 2:
        raise ValueError("Use full/lora/qlora and at least two batches")
    output.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    L.seed_everything(42)
    train, val, test = (
        work / f"{name}.csv" for name in ("train", "validation", "test")
    )
    _pairs(train, 1, 20)
    _pairs(val, 101, 2)
    _pairs(test, 201, 2)
    manifest = prepare(train, val, test, work / "stages")
    reward = TranslationReward(
        chrf_weight=0.0,
        bleu_weight=0.3,
        comet_weight=0.7,
        comet_device=comet_device,
        comet_model_name=comet_model,
        comet_cache_dir=str(work.parent / "comet-cache"),
    )
    options: dict[str, Any] = {
        "max_source_length": 128,
        "max_new_tokens": 16,
        "group_size": 2,
        "num_iterations": 2,
        "learning_rate": 1e-5,
        "optimizer": partial(TransferAdamW, lr=1e-5),
        "load_dtype": "float32" if mode == "full" else "bfloat16",
        "lora": {
            "enabled": True,
            "r": 8,
            "alpha": 16,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        }
        if mode != "full"
        else None,
        "quantization": "nf4" if mode == "qlora" else None,
        "gradient_checkpointing": True,
        "logprob_batch_size": 1,
        "profiling": {"enabled": True, "synchronize_cuda": True},
        "prompt_template": "Translate {source_lang} to {target_lang}. Return only the translation.\nSource: {source}\nTranslation:",
    }
    report: dict[str, Any] = {
        "model": model,
        "mode": mode,
        "gpu": torch.cuda.get_device_name(),
        "synthetic_data_only": True,
        "comet_model": comet_model,
        "comet_device": comet_device,
        "stage_counts": manifest["counts"],
        "stages": {},
        "status": "running",
        "settings": {
            "steps_per_stage": steps,
            "batch_size": 1,
            "group_size": 2,
            "optimizer_iterations_per_rl_rollout": 2,
            "source_cap": 128,
            "translation_cap": 16,
            "proposal_cap": 24,
            "precision": "bf16-mixed",
            "comet_precision": "float32, AMP disabled",
            "gradient_checkpointing": True,
            "logprob_batch_size": 1,
            "rank": 8 if mode != "full" else None,
        },
    }
    started = time.perf_counter()
    try:
        for stage in ("sft", "rl"):
            stage_output = output / stage
            data = TranslationDataModule(
                train_file=str(work / f"stages/{stage}.csv"),
                val_file=str(val),
                test_file=str(test),
                batch_size=1,
                bidirectional=True,
                augmentation={"enabled": True, "concat_probability": 0.1},
            )
            load_started = time.perf_counter()
            if stage == "sft":
                module: Any = TranslationSFTModule(
                    model_name_or_path=model,
                    max_target_length=32,
                    reward=reward,
                    **options,
                )
            else:
                module = GRPOModule(
                    model_name_or_path=str(work / "sft_dense"), reward=reward, **options
                )
            stage_report: dict[str, Any] = {
                "model_load_seconds": time.perf_counter() - load_started
            }
            trainer = _trainer(stage_output, steps)
            torch.cuda.reset_peak_memory_stats()
            fit_started = time.perf_counter()
            trainer.fit(module, datamodule=data)
            stage_report["fit_seconds_including_validation"] = (
                time.perf_counter() - fit_started
            )
            trainer.test(module, datamodule=data)
            stage_report.update(
                optimizer_steps=trainer.global_step,
                peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                parameter_dtype=str(next(module.policy.parameters()).dtype),
                metrics={
                    key: float(value.cpu())
                    for key, value in trainer.callback_metrics.items()
                },
                phases=module.phase_timing.summary(),
            )
            if stage == "sft":
                if mode == "full":
                    module.export_pretrained(work / "sft_dense")
                else:
                    checkpoint = work / "sft_adapter.ckpt"
                    trainer.save_checkpoint(checkpoint)
                    export_sft(checkpoint, work / "sft_dense", load_dtype="bfloat16")
            report["stages"][stage] = stage_report
            del trainer, module, data
            _release()
        # Seeded memory guarantees solver work even if short proposals are malformed.
        # The judge is mocked; no claim about live API throughput or source quality.
        feedback = TranslationJudge(SyntheticJudge())
        tokenizer: Any = cast(Any, AutoTokenizer).from_pretrained(work / "sft_dense")
        shared = SelfPlayGRPOModule(
            feedback=feedback,
            model_name_or_path=str(work / "sft_dense"),
            proposal_group_size=2,
            proposal_max_new_tokens=24,
            memory_examples=1,
            use_chat_template=bool(tokenizer.chat_template),
            reward=reward,
            **options,
        )
        shared.memory.add(
            TranslationTask("The book has ten pages.", "English", "Vietnamese")
        )
        contexts = SelfPlayDataModule(
            topics=["daily life"],
            steps_per_epoch=steps,
            val_file=str(val),
            test_file=str(test),
            bidirectional=True,
        )
        trainer = _trainer(output / "self_play", steps)
        torch.cuda.reset_peak_memory_stats()
        fit_started = time.perf_counter()
        trainer.fit(shared, datamodule=contexts)
        trainer.test(shared, datamodule=contexts)
        phases = shared.phase_timing.summary()
        if (
            "solver/current_forward" not in phases["phases"]
            or "proposer/current_forward" not in phases["phases"]
        ):
            raise AssertionError("Both shared-policy roles must execute")
        report["stages"]["self_play"] = {
            "fit_and_test_seconds": time.perf_counter() - fit_started,
            "optimizer_steps": trainer.global_step,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "feedback": "explicit synthetic mock; not API/model compute",
            "memory_seeded": True,
            "phases": phases,
        }
        report["status"] = "passed"
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        (output / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
    return report


def main() -> None:
    """Run one mode per process to avoid allocator contamination between comparisons."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--mode", choices=("full", "lora", "qlora"), required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--comet-device", default="cuda:0")
    parser.add_argument("--comet-model", default="Unbabel/wmt22-cometkiwi-da")
    args = parser.parse_args()
    print(
        json.dumps(
            benchmark(
                args.output,
                args.work,
                args.model,
                args.mode,
                args.steps,
                args.comet_device,
                args.comet_model,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
