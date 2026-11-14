"""Measure real forward/backward/Adam peaks at the training token caps on one GPU."""

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import torch

from src.models.sft_module import TranslationSFTModule


def probe(
    module: TranslationSFTModule,
    optimizer: torch.optim.Optimizer,
    stage: str,
    batch: int,
) -> dict[str, Any]:
    optimizer.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    if stage == "sft":
        rows = {
            "source": ["translation " * 600] * batch,
            "reference": ["translation " * 150] * batch,
        }
        inputs = module.supervised_inputs(rows)
        module._prepare_gradient_scoring()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = module.policy(**inputs, use_cache=False).loss
        loss.backward()
        optimizer.step()
    else:
        prompts = module._format_prompts(["translation " * 600] * batch)
        inputs = {
            key: value.repeat_interleave(2, 0)
            for key, value in module._tokenize_prompts(prompts).items()
        }
        module.policy.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            sequences = module.policy.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=128,
                min_new_tokens=128,
                use_cache=True,
                pad_token_id=module.tokenizer.pad_token_id,
            )
            mask = torch.ones((2 * batch, 128), dtype=torch.bool, device="cuda")
            old = module._logps(module.policy, inputs, sequences, mask)
            reference = module._reference_logps(inputs, sequences, mask)
        advantages = torch.tensor([-1.0, 1.0], device="cuda").repeat(batch)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            module._prepare_gradient_scoring()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                module._backward_rollout(
                    {
                        "inputs": inputs,
                        "sequences": sequences,
                        "mask": mask,
                        "old_logps": old,
                        "reference_logps": reference,
                        "advantages": advantages,
                    },
                    backward_fn=torch.autograd.backward,
                )
            optimizer.step()
    torch.cuda.synchronize()
    return {
        "stage": stage,
        "batch_size": batch,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "seconds": time.perf_counter() - started,
        "status": "passed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    module = TranslationSFTModule.load_from_checkpoint(
        args.checkpoint, weights_only=False, map_location="cpu"
    )
    module.to("cuda")
    optimizer = module.configure_optimizers()
    checkpoint = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    optimizer.load_state_dict(checkpoint["optimizer_states"][0])
    del checkpoint
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    report: dict[str, Any] = {
        "total_gib": total,
        "target_fraction": 0.90,
        "token_caps": [512, 128],
        "checkpoint_unchanged": str(args.checkpoint),
        "trials": [],
        "selected": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for stage in ("sft", "rl"):
        lower, upper, candidate = 0, None, 1
        while True:
            try:
                trial = probe(module, optimizer, stage, candidate)
                fits = trial["peak_allocated_gib"] <= total * 0.90
            except torch.cuda.OutOfMemoryError:
                trial = {"stage": stage, "batch_size": candidate, "status": "oom"}
                fits = False
            optimizer.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            report["trials"].append(trial)
            print(json.dumps(trial), flush=True)
            if fits:
                lower = candidate
            else:
                upper = candidate
            if upper is not None and upper - lower <= 1 or lower >= 128:
                break
            candidate = candidate * 2 if upper is None else (lower + upper) // 2
            args.output.write_text(json.dumps(report, indent=2))
        report["selected"][stage] = max(lower, 1)
        args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
