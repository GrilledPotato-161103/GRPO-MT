"""Forward-only audit of padding/batch precision; performs no optimizer updates."""

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from scripts.benchmarks.grpo import legacy_logps, synthetic_pairs
from src.models.grpo_module import GRPOModule


def difference(first: Tensor, second: Tensor, mask: Tensor) -> dict[str, float]:
    delta = (second - first)[mask].float()
    absolute = delta.abs()
    ratio = delta.exp()
    return {
        "max_abs_logp": float(absolute.max()),
        "mean_abs_logp": float(absolute.mean()),
        "p95_abs_logp": float(absolute.quantile(0.95)),
        "ratio_outside_0.8_1.2": float(((ratio < 0.8) | (ratio > 1.2)).float().mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/workspace/benchmark-results/precision.json"),
    )
    args = parser.parse_args()
    torch.set_num_threads(8)
    module = GRPOModule(
        model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
        revision="a09a35458c702b33eeacc393d103063234e8bc28",
        load_dtype="bfloat16",
        lora={
            "enabled": True,
            "r": 8,
            "alpha": 16,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        },
        gradient_checkpointing=True,
        group_size=4,
        max_source_length=256,
        max_new_tokens=128,
        logprob_batch_size=1,
        sort_candidates_by_length=False,
        trim_logprob_padding=False,
        prompt_template="Translate {source_lang} to {target_lang}. Return only the translation.\nSource: {source}\nTranslation:",
        diagnostics={"enabled": False},
    ).to("cuda")
    report: dict[str, Any] = {
        "optimizer_updates": 0,
        "torch": torch.__version__,
        "checks": {},
    }

    def save() -> None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    rows = synthetic_pairs(16)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        rollout = module._sample_prompts(
            module._format_prompts(
                rows["source"], rows["source_lang"], rows["target_lang"]
            ),
            4,
        )
        baseline = rollout["old_logps"]
        for name, trim, sort in (
            ("batch8_untrimmed", False, False),
            ("batch8_trimmed", True, False),
            ("batch8_trimmed_sorted", True, True),
        ):
            module.hparams.update(
                logprob_batch_size=8,
                logprob_token_budget=3072,
                trim_logprob_padding=trim,
                sort_candidates_by_length=sort,
            )
            values = module._logps(
                module.policy, rollout["inputs"], rollout["sequences"], rollout["mask"]
            )
            report["checks"][name] = difference(baseline, values, rollout["mask"])
            if name == "batch8_trimmed_sorted":
                matched_old = values
            save()
        matched_repeat = module._logps(
            module.policy, rollout["inputs"], rollout["sequences"], rollout["mask"]
        )
        report["checks"]["matched_plan_repeat"] = difference(
            matched_old, matched_repeat, rollout["mask"]
        )
    module._prepare_gradient_scoring()
    with torch.enable_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        current = module._logps(
            module.policy, rollout["inputs"], rollout["sequences"], rollout["mask"]
        )
        report["checks"]["matched_plan_old_vs_grad_current"] = difference(
            matched_old, current.detach(), rollout["mask"]
        )
    del current
    module.policy.eval()
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        strict_old = legacy_logps(module, module.policy, rollout)
        strict_new = module._logps(
            module.policy, rollout["inputs"], rollout["sequences"], rollout["mask"]
        )
        report["checks"]["bf16_full_reduction_batch_difference"] = difference(
            strict_old, strict_new, rollout["mask"]
        )
    save()
    indices = torch.tensor([0, 15, 32, 63], device=module.device)
    subset = {
        "inputs": {key: value[indices] for key, value in rollout["inputs"].items()},
        "sequences": rollout["sequences"][indices],
        "mask": rollout["mask"][indices],
    }
    del rollout, matched_old, matched_repeat, strict_old, strict_new, values, baseline
    gc.collect()
    torch.cuda.empty_cache()
    module.policy.float()
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        torch.backends.cuda.matmul.fp32_precision = "ieee"
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
    with torch.no_grad():
        fp32_old = legacy_logps(module, module.policy, subset)
        fp32_new = module._logps(
            module.policy, subset["inputs"], subset["sequences"], subset["mask"]
        )
        report["checks"]["fp32_subset_batch_padding_equivalence"] = difference(
            fp32_old, fp32_new, subset["mask"]
        )
    report["completed"] = True
    save()
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
