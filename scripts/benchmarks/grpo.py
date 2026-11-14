"""Replay fixed candidates to compare legacy GRPO graphs with batched backward.

No training checkpoint is saved. All trials restore the same adapter parameters
and Adam state. Synthetic bilingual prompts avoid uploading a private dataset.
The feedback service is excluded; optional VRAM reservation is explicitly an
allocation, not a COMET execution benchmark. Default schedule: ten updates total.
"""

import argparse
import gc
import hashlib
import json
import statistics
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from src.models.components.grpo import group_advantages, grpo_loss
from src.models.grpo_module import GRPOModule


def synthetic_pairs(count: int) -> dict[str, list[str]]:
    pairs = [
        ("The book has ten pages.", "Cuốn sách có mười trang."),
        (
            "Please send the report before Friday.",
            "Vui lòng gửi báo cáo trước thứ Sáu.",
        ),
        (
            "The train will arrive at seven in the morning.",
            "Tàu sẽ đến lúc bảy giờ sáng.",
        ),
        (
            "She checked the numbers carefully before submitting the document.",
            "Cô ấy kiểm tra kỹ các con số trước khi nộp tài liệu.",
        ),
    ]
    rows = []
    for index in range(count):
        en, vi = pairs[(index // 2) % len(pairs)]
        repeat = (1, 2, 4, 8)[(index // 2) % 4]
        reverse = index % 2
        rows.append(
            {
                "source": " ".join([vi if reverse else en] * repeat),
                "reference": " ".join([en if reverse else vi] * repeat),
                "source_lang": "Vietnamese" if reverse else "English",
                "target_lang": "English" if reverse else "Vietnamese",
                "direction": "vi_en" if reverse else "en_vi",
            }
        )
    return {key: [row[key] for row in rows] for key in rows[0]}


def adapter_digest(module: GRPOModule) -> str:
    digest = hashlib.sha256()
    for name, parameter in module.policy.named_parameters():
        if parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().float().cpu().numpy().tobytes())
    return digest.hexdigest()


def clone_to_cpu(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: clone_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_to_cpu(item) for item in value]
    return deepcopy(value)


def legacy_logps(module: GRPOModule, model: Any, rollout: dict[str, Any]) -> Tensor:
    """The old one-row forwards, globally padded, with all graphs retained."""
    return torch.cat(
        [
            module._score_batch(
                model,
                {key: value[i : i + 1] for key, value in rollout["inputs"].items()},
                rollout["sequences"][i : i + 1],
                rollout["mask"][i : i + 1],
            )
            for i in range(len(rollout["sequences"]))
        ]
    )


def score_inference(
    module: GRPOModule, rollout: dict[str, Any], legacy: bool
) -> tuple[Tensor, Tensor, float, float]:
    module.policy.eval()
    with torch.no_grad():
        torch.cuda.synchronize()
        started = time.perf_counter()
        old = (
            legacy_logps(module, module.policy, rollout)
            if legacy
            else module._logps(
                module.policy, rollout["inputs"], rollout["sequences"], rollout["mask"]
            )
        )
        torch.cuda.synchronize()
        old_seconds = time.perf_counter() - started
        started = time.perf_counter()
        with module.policy.disable_adapter():
            reference = (
                legacy_logps(module, module.policy, rollout)
                if legacy
                else module._logps(
                    module.policy,
                    rollout["inputs"],
                    rollout["sequences"],
                    rollout["mask"],
                )
            )
        torch.cuda.synchronize()
        reference_seconds = time.perf_counter() - started
    return old, reference, old_seconds, reference_seconds


def update_once(
    module: GRPOModule,
    optimizer: torch.optim.Optimizer,
    rollout: dict[str, Any],
    legacy: bool,
) -> dict[str, Any]:
    """Time one full optimizer update; no checkpoint or external logger is used."""
    optimizer.zero_grad(set_to_none=True)
    module._prepare_gradient_scoring()
    module.phase_timing.events.clear()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        if legacy:
            with module.phase_timing.measure(
                "translation/current_forward", module.device
            ):
                current = legacy_logps(module, module.policy, rollout)
            loss, metrics = grpo_loss(
                current,
                rollout["old_logps"],
                rollout["reference_logps"],
                rollout["advantages"],
                rollout["mask"],
                module.hparams["clip_epsilon"],
                module.hparams["beta"],
            )
            with module.phase_timing.measure("translation/backward", module.device):
                torch.autograd.backward(loss)
            metrics = {key: value.detach() for key, value in metrics.items()}
            loss = loss.detach()
            del current
        else:
            loss, metrics = module._backward_rollout(
                rollout, backward_fn=torch.autograd.backward
            )
    parameters = [
        parameter for parameter in module.policy.parameters() if parameter.requires_grad
    ]
    norm = torch.nn.utils.clip_grad_norm_(
        parameters, module.hparams["max_grad_norm"], error_if_nonfinite=True
    )
    with module.phase_timing.measure("translation/optimizer", module.device):
        optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Non-finite benchmark loss")
    phases = module.phase_timing.summary()["phases"]
    result = {
        "seconds": elapsed,
        "forward_seconds": phases["translation/current_forward"]["total_seconds"],
        "backward_seconds": phases["translation/backward"]["total_seconds"],
        "optimizer_seconds": phases["translation/optimizer"]["total_seconds"],
        "forward_model_calls": len(rollout["sequences"])
        if legacy
        else int(metrics["scoring_microbatches"]),
        "backward_calls": 1 if legacy else int(metrics["scoring_microbatches"]),
        "loss": float(loss),
        "gradient_norm": float(norm),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    }
    if not legacy:
        result["padding_fraction"] = float(metrics["scoring_padding_fraction"])
        result["padded_tokens"] = int(metrics["scoring_padded_tokens"])
    return result


def run(
    module: GRPOModule, output: Path, sources: int = 16, reserve_gib: float = 3.0
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    initial_digest = adapter_digest(module)
    initial = {
        name: p.detach().cpu().clone()
        for name, p in module.policy.named_parameters()
        if p.requires_grad
    }
    reservation = torch.empty(
        int(reserve_gib * 2**30), dtype=torch.uint8, device=module.device
    )
    reservation.zero_()
    report: dict[str, Any] = {
        "status": "running",
        "gpu": torch.cuda.get_device_name(),
        "total_gpu_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
        "model": module.hparams["model_name_or_path"],
        "revision": module.hparams["base_revision"],
        "sources": sources,
        "group_size": module.hparams["group_size"],
        "source_cap": module.hparams["max_source_length"],
        "completion_cap": module.hparams["max_new_tokens"],
        "checkpointing": module.hparams["gradient_checkpointing"],
        "reserved_feedback_gib": reserve_gib,
        "feedback_executed": False,
        "precision": "BF16 autocast with FP32 LoRA parameters/log-softmax",
        "data": "fixed synthetic bidirectional EN/VI prompts; generated candidates",
        "updates_executed": 0,
        "max_updates": 10,
        "trials": [],
    }

    def save() -> None:
        (output / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )

    def restore() -> None:
        with torch.no_grad():
            for name, parameter in module.policy.named_parameters():
                if name in initial:
                    parameter.copy_(initial[name])

    def invoke(
        optimizer: torch.optim.Optimizer,
        rollout: dict[str, Any],
        legacy: bool,
        warmup: bool,
    ) -> dict[str, Any]:
        if report["updates_executed"] >= 10:
            raise RuntimeError("Ten-update benchmark cap reached")
        restore()
        optimizer.zero_grad(set_to_none=True)
        if optimizer_snapshot is not None:
            optimizer.load_state_dict(deepcopy(optimizer_snapshot))
        gc.collect()
        torch.cuda.empty_cache()
        result = update_once(module, optimizer, rollout, legacy)
        report["updates_executed"] += 1
        result["warmup"] = warmup
        save()
        print(
            json.dumps(
                {"update": report["updates_executed"], "warmup": warmup, **result}
            ),
            flush=True,
        )
        return result

    optimizer = torch.optim.AdamW(
        [p for p in module.policy.parameters() if p.requires_grad],
        lr=5e-6,
        weight_decay=0.01,
    )
    optimizer_snapshot: Any = None
    try:
        module.hparams.update(
            logprob_batch_size=1,
            logprob_token_budget=None,
            sort_candidates_by_length=False,
            trim_logprob_padding=False,
        )
        rows = synthetic_pairs(sources)
        module.phase_timing.enabled = True
        with torch.autocast("cuda", dtype=torch.bfloat16):
            rollout = module._sample_prompts(
                module._format_prompts(
                    rows["source"], rows["source_lang"], rows["target_lang"]
                ),
                module.hparams["group_size"],
            )
        generation = module.phase_timing.summary()["phases"]["translation/generation"][
            "total_seconds"
        ]
        rewards = (
            torch.arange(len(rollout["sequences"]), device=module.device)
            .remainder(module.hparams["group_size"])
            .float()
        )
        rollout["advantages"] = group_advantages(rewards, module.hparams["group_size"])
        report.update(
            generation_seconds=generation,
            candidates=len(rollout["sequences"]),
            completion_tokens=int(rollout["mask"].sum()),
            valid_source_tokens=int(rollout["inputs"]["attention_mask"].sum()),
        )
        save()
        configs = [
            ("legacy_retained_chunk1", True, 1),
            ("batched_chunk8", False, 8),
            ("batched_chunk16", False, 16),
        ]
        for label, legacy, chunk in configs:
            module.hparams.update(
                logprob_batch_size=chunk,
                logprob_token_budget=None if legacy else 3072,
                sort_candidates_by_length=not legacy,
                trim_logprob_padding=not legacy,
            )
            trial: dict[str, Any] = {
                "name": label,
                "chunk_cap": chunk,
                "sources": sources,
                "samples": [],
            }
            report["trials"].append(trial)
            try:
                restore()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    old, reference, old_time, ref_time = score_inference(
                        module, rollout, legacy
                    )
                mask = rollout["mask"]
                trial.update(
                    old_forward_seconds=old_time,
                    reference_forward_seconds=ref_time,
                    old_logps_max_error=float(
                        (old - rollout["old_logps"])[mask].abs().max()
                    ),
                    reference_logps_max_error=float(
                        (reference - rollout["reference_logps"])[mask].abs().max()
                    ),
                )
                # Keep the same detached old/reference scores in every trial.
                trial["warmup"] = invoke(optimizer, rollout, legacy, True)
                if optimizer_snapshot is None:
                    optimizer_snapshot = clone_to_cpu(optimizer.state_dict())
                for _ in range(2):
                    trial["samples"].append(invoke(optimizer, rollout, legacy, False))
                trial["mean_update_seconds"] = statistics.mean(
                    sample["seconds"] for sample in trial["samples"]
                )
                trial["mean_forward_seconds"] = statistics.mean(
                    sample["forward_seconds"] for sample in trial["samples"]
                )
                trial["mean_backward_seconds"] = statistics.mean(
                    sample["backward_seconds"] for sample in trial["samples"]
                )
                trial["amortized_seconds_per_update_excluding_feedback"] = (
                    trial["mean_update_seconds"]
                    + (generation + old_time + ref_time) / 2
                )
                trial["candidates_per_second"] = (
                    len(rollout["sequences"]) / trial["mean_update_seconds"]
                )
                trial["status"] = "passed"
            except torch.cuda.OutOfMemoryError:
                trial["status"] = "oom"
                optimizer.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
            save()
        # One final measurement trades rollout size against graph size, without
        # changing the full-batch comparison above. Already warmed graph shapes.
        if report["updates_executed"] < 10:
            small = {
                key: (
                    {k: v[: len(v) // 2] for k, v in value.items()}
                    if key == "inputs"
                    else value[: len(value) // 2]
                    if isinstance(value, Tensor)
                    else value
                )
                for key, value in rollout.items()
            }
            module.hparams.update(
                logprob_batch_size=16,
                logprob_token_budget=3072,
                sort_candidates_by_length=True,
                trim_logprob_padding=True,
            )
            trial = {
                "name": "batched_half_rollout_chunk16",
                "sources": sources // 2,
                "status": "running",
            }
            report["trials"].append(trial)
            try:
                trial["sample"] = invoke(optimizer, small, False, False)
                trial["candidates_per_second"] = (
                    len(small["sequences"]) / trial["sample"]["seconds"]
                )
                trial["status"] = "passed"
            except torch.cuda.OutOfMemoryError:
                trial["status"] = "oom"
        report["status"] = "completed"
    finally:
        optimizer.zero_grad(set_to_none=True)
        restore()
        report["adapter_parameters_restored"] = adapter_digest(module) == initial_digest
        report["training_checkpoint_saved"] = False
        save()
        del reservation
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--revision", default="a09a35458c702b33eeacc393d103063234e8bc28"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("logs/grpo_batching_benchmark")
    )
    parser.add_argument("--sources", type=int, default=16)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--source-cap", type=int, default=256)
    parser.add_argument("--completion-cap", type=int, default=128)
    parser.add_argument("--reserve-gib", type=float, default=3.0)
    args = parser.parse_args()
    if args.sources < 2 or args.sources % 2 or args.reserve_gib < 0:
        parser.error("sources must be positive/even and reserve-gib nonnegative")
    torch.set_num_threads(8)
    module = GRPOModule(
        model_name_or_path=args.model,
        revision=args.revision,
        load_dtype="bfloat16",
        lora={
            "enabled": True,
            "r": 8,
            "alpha": 16,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        },
        gradient_checkpointing=True,
        group_size=args.group_size,
        max_source_length=args.source_cap,
        max_new_tokens=args.completion_cap,
        prompt_template="Translate {source_lang} to {target_lang}. Return only the translation.\nSource: {source}\nTranslation:",
        diagnostics={"enabled": False},
        profiling={"enabled": True},
    ).to("cuda")
    report = run(module, args.output, args.sources, args.reserve_gib)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
