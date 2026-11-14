"""Refine SFT token budget between integer full-length batch limits."""

import argparse
import gc
import json
from pathlib import Path

import torch

from scripts.benchmarks.batch_size import probe
from src.models.sft_module import TranslationSFTModule


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    samples = sorted(
        [
            row
            for row in report["trials"]
            if row["stage"] == "sft" and row["status"] == "passed"
        ],
        key=lambda row: row["batch_size"],
    )
    first, last = samples[0], samples[-1]
    slope = (last["peak_allocated_gib"] - first["peak_allocated_gib"]) / (
        last["batch_size"] - first["batch_size"]
    )
    intercept = first["peak_allocated_gib"] - slope * first["batch_size"]
    limit = report["total_gib"] * 0.9
    budget = int(((limit - intercept) / slope * 640) // 128) * 128
    module = TranslationSFTModule.load_from_checkpoint(
        args.checkpoint, weights_only=False, map_location="cpu"
    )
    module.to("cuda")
    optimizer = module.configure_optimizers()
    checkpoint = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    optimizer.load_state_dict(checkpoint["optimizer_states"][0])
    del checkpoint
    while budget >= 1280:
        module.hparams["max_source_length"] = budget // 8 - 128
        try:
            trial = probe(module, optimizer, "sft", 8)
            trial["source_cap"] = budget // 8 - 128
            trial["padded_token_budget"] = budget
            print(json.dumps(trial), flush=True)
            if trial["peak_allocated_gib"] <= limit:
                report["sft_token_budget"] = budget
                report["sft_token_budget_trial"] = trial
                args.report.write_text(json.dumps(report, indent=2))
                return
        except torch.cuda.OutOfMemoryError:
            print(
                json.dumps({"padded_token_budget": budget, "status": "oom"}), flush=True
            )
        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        budget -= 128
    raise RuntimeError("No safe refined SFT token budget found")


if __name__ == "__main__":
    main()
