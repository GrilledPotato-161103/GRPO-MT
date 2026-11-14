"""Measure actual NLLB LoRA VRAM/FLOPs and an offline correction-training smoke run.

Only the translation policy is real in this benchmark. COMET/SAC are explicitly
mocked: no API credentials, paid requests, or rental instance are used.
"""

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any, cast


class OfflineJudge:
    def evaluate(
        self,
        sources: list[str],
        predictions: list[str],
        source_langs: list[str],
        target_langs: list[str],
    ) -> list[float]:
        return [0.3 if len(predictions) > 2 else 1.0] * len(predictions)

    def check_ready(self) -> None:
        pass


def run(
    model: str,
    output: Path,
    steps: int,
    target_lengths: list[int],
    batch_sizes: list[int],
    source_length: int = 8,
    quantization: str | None = None,
    direct_gpu_load: bool = False,
) -> dict[str, Any]:
    import lightning as L
    import torch
    from torch.utils.data import DataLoader

    from src.models.anchored_grpo_module import AnchoredGRPOModule
    from src.models.components.translation_reward import TranslationReward

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    torch.set_num_threads(4)
    L.seed_everything(42)
    output.parent.mkdir(parents=True, exist_ok=True)
    reward = TranslationReward(
        chrf_weight=0,
        comet_weight=1,
        comet_scorer=lambda sources, predictions, references: [0.8] * len(predictions),
        sac={"enabled": True, "consistency_weight": 0, "fusion": {"enabled": False}},
        sac_scorer=OfflineJudge(),
    )
    started = time.perf_counter()
    injected: dict[str, Any] = {}
    if direct_gpu_load and quantization is None:
        from transformers import AutoModelForSeq2SeqLM

        injected["policy"] = cast(Any, AutoModelForSeq2SeqLM).from_pretrained(
            model,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            low_cpu_mem_usage=True,
        )
    module = AnchoredGRPOModule(
        model_name_or_path=model,
        load_dtype="bfloat16",
        quantization=quantization,
        reward=reward,
        beta=0,
        temperature=1,
        language_codes={"English": "eng_Latn", "Vietnamese": "vie_Latn"},
        prompt_template="{source}",
        group_size=2,
        num_iterations=1,
        max_new_tokens=32,
        max_source_length=258,
        learning_rate=2e-5,
        gradient_checkpointing=True,
        logprob_batch_size=1,
        logprob_token_budget=1024,
        lora={"enabled": True, "r": 8, "alpha": 16, "target_modules": "all-linear"},
        diagnostics={"enabled": False},
        **injected,
    )
    # Update Lightning's device tracker even when HF loaded the child policy
    # directly onto CUDA (including NF4); otherwise input tensors stay on CPU.
    module.cuda()
    trainable = [p for p in module.policy.parameters() if p.requires_grad]
    # Allocate AdamW state without changing the policy; measurements include it.
    optimizer = torch.optim.AdamW(trainable, lr=0, weight_decay=0)
    for p in trainable:
        p.grad = torch.zeros_like(p)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    result: dict[str, Any] = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "model": model,
        "quantization": quantization,
        "load_seconds": time.perf_counter() - started,
        "parameters": sum(p.numel() for p in module.policy.parameters()),
        "trainable_parameters": sum(p.numel() for p in trainable),
        "model_parameter_bytes": sum(
            p.numel() * p.element_size() for p in module.policy.parameters()
        ),
        "base_allocated_gib": torch.cuda.memory_allocated() / 2**30,
        "comet_and_sac": "MOCKED; their VRAM and network wait are excluded",
        "cases": [],
    }
    lang = module.tokenizer.convert_tokens_to_ids("vie_Latn")
    body_token = module.tokenizer("book", add_special_tokens=False)["input_ids"][0]
    for batch_size in batch_sizes:
        for length in target_lengths:
            source_ids = (
                [module.tokenizer.convert_tokens_to_ids("eng_Latn")]
                + [body_token] * (source_length - 2)
                + [2]
            )
            inputs = {
                "input_ids": torch.tensor(
                    [source_ids] * batch_size, device=module.device
                ),
                "attention_mask": torch.ones(
                    (batch_size, source_length), dtype=torch.long, device=module.device
                ),
            }
            sequence = torch.tensor(
                [[2, lang] + [body_token] * max(1, length - 2) + [2]] * batch_size,
                device=module.device,
            )
            mask = torch.ones(
                (batch_size, sequence.shape[1] - 1),
                dtype=torch.bool,
                device=module.device,
            )
            mask[:, 0] = False
            case: dict[str, Any] = {
                "batch_size": batch_size,
                "decoder_positions": mask.shape[1],
                "source_positions": inputs["input_ids"].shape[1],
            }
            try:
                module._prepare_gradient_scoring()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                times = []
                for _ in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    a, b, c = [
                        cast(Any, torch.cuda.Event)(enable_timing=True)
                        for _ in range(3)
                    ]
                    a.record()
                    logps = module._score_batch(module.policy, inputs, sequence, mask)
                    loss = -(logps * mask).sum() / mask.sum()
                    b.record()
                    cast(Any, loss).backward()
                    c.record()
                    torch.cuda.synchronize()
                    times.append(
                        {
                            "forward_ms": a.elapsed_time(b),
                            "backward_ms": b.elapsed_time(c),
                        }
                    )
                    del logps, loss
                case.update(
                    times[-1],
                    peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                    peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                    status="ok",
                )
                if batch_size == batch_sizes[0] and length == target_lengths[0]:
                    optimizer.zero_grad(set_to_none=True)
                    with torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU],
                        with_flops=True,
                    ) as profile:
                        lp = module._score_batch(module.policy, inputs, sequence, mask)
                        cast(Any, -(lp * mask).sum() / mask.sum()).backward()
                    case["registered_operator_flops_forward_backward"] = sum(
                        event.flops for event in profile.key_averages()
                    )
                    case[
                        "flops_scope"
                    ] = "Registered matrix operators, including checkpoint recomputation; excludes unsupported ops"
                    del lp
            except torch.OutOfMemoryError:
                optimizer.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
                case["status"] = "oom"
            result["cases"].append(case)
            output.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(json.dumps(case), flush=True)
    optimizer.zero_grad(set_to_none=True)
    # Teacher-forced SFT has different logit/loss buffers from GRPO scoring.
    # Measure it separately so larger-model planning does not undercount SFT.
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        labels = sequence[:, 1:].clone()
        labels[:, 0] = -100
        module._prepare_gradient_scoring()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            sft_loss = module.policy(
                **inputs,
                decoder_input_ids=sequence[:, :-1],
                labels=labels,
                use_cache=False,
            ).loss
        cast(Any, sft_loss).backward()
        torch.cuda.synchronize()
        result["sft_probe"] = {
            "batch_size": sequence.shape[0],
            "source_positions": inputs["input_ids"].shape[1],
            "target_positions": labels.shape[1],
            "base_gib": result["base_allocated_gib"],
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            "seconds_forward_backward": time.perf_counter() - started,
        }
        del sft_loss
    except torch.OutOfMemoryError:
        result["sft_probe"] = {"status": "oom"}
    optimizer.zero_grad(set_to_none=True)
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()
    if steps == 0:
        result["smoke"] = {"optimizer_steps": 0, "mock_judge": True}
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result
    batch = {
        "source": ["Hello.", "Xin chÃ o."],
        "reference": ["Xin chÃ o.", "Hello."],
        "source_lang": ["English", "Vietnamese"],
        "target_lang": ["Vietnamese", "English"],
        "direction": ["en_vi", "vi_en"],
    }
    trainer = L.Trainer(
        max_steps=steps,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        num_sanity_val_steps=0,
        limit_val_batches=0,
        default_root_dir=output.parent / "smoke",
        log_every_n_steps=1,
    )
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    trainer.fit(
        module,
        train_dataloaders=cast(Any, DataLoader)([batch] * steps, batch_size=None),
    )
    torch.cuda.synchronize()
    result["smoke"] = {
        "optimizer_steps": trainer.global_step,
        "directions": batch["direction"],
        "seconds": time.perf_counter() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "mock_judge": True,
    }
    if trainer.global_step != steps:
        raise RuntimeError("Smoke training did not reach the requested optimizer steps")
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="logs/assets/nllb-600m")
    parser.add_argument(
        "--output", type=Path, default=Path("diagnostics/correction_20261114/gpu.json")
    )
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--target-lengths", type=int, nargs="+", default=[32, 64, 256])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--source-length", type=int, default=8)
    parser.add_argument("--quantization", choices=["nf4"], default=None)
    parser.add_argument(
        "--direct-gpu-load",
        action="store_true",
        help="Avoid a full CPU copy while loading large public checkpoints",
    )
    args = parser.parse_args()
    result = run(**vars(args))
    print(
        json.dumps(
            {
                "gpu": result["gpu"],
                "smoke": result["smoke"],
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
