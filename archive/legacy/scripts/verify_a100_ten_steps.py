"""Exactly ten optimizer updates with Qwen LoRA and frozen COMETKiwi on one GPU."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import lightning as L
import torch
from torch.utils.data import DataLoader

from src.data.translation_datamodule import TranslationDataModule, collate_translation
from src.models.components.translation_reward import TranslationReward
from src.models.grpo_module import GRPOModule
from src.models.sft_module import TranslationSFTModule


class TenUpdatePolicy(TranslationSFTModule):
    def training_step(self, batch, batch_idx):
        assert (
            max(
                len(row)
                for row in self._encode(
                    batch["source"], batch["source_lang"], batch["target_lang"]
                )["input_ids"]
            )
            <= self.hparams["max_source_length"]
        )
        if self.global_step < 2:
            loss = super().training_step(batch, batch_idx)
        else:
            loss = GRPOModule.training_step(self, batch, batch_idx)
        print(f"Verification optimizer updates: {self.global_step}/10", flush=True)
        return loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--comet",
        type=Path,
        default=Path("/workspace/cometkiwi/checkpoints/model.ckpt"),
    )
    parser.add_argument("--output", type=Path, default=Path("logs/a100_verify10"))
    args = parser.parse_args()
    config = json.loads(Path("logs/production/rental_config.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    torch.set_num_threads(8)
    L.seed_everything(42)
    with args.checkpoint.open("rb") as stream:
        original_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    reward = TranslationReward(
        chrf_weight=0,
        bleu_weight=0.3,
        comet_weight=0.7,
        comet_model_name=str(args.comet),
        comet_device="cuda:0",
        comet_batch_size=config["comet_batch_size"],
    )
    assert reward.comet is not None
    reward.comet(["The book has ten pages."], ["Cuốn sách có mười trang."])
    model = TenUpdatePolicy.load_from_checkpoint(
        args.checkpoint,
        weights_only=False,
        map_location="cpu",
        max_source_length=config["source_cap"],
        max_new_tokens=config["target_cap"],
        max_target_length=config["target_cap"],
        group_size=2,
        num_iterations=2,
        reward=reward,
        profiling={"enabled": True},
        logprob_batch_size=1,
    )
    before = {
        name: p.detach().cpu().clone()
        for name, p in model.policy.named_parameters()
        if p.requires_grad
    }
    batches = []
    for stage, count in [("sft", 2), ("rl", 4)]:
        size = config["batch_sizes"][stage]
        data = TranslationDataModule(
            train_file=f"data/phomt_stages/{stage}.csv",
            val_file="data/phomt_prepared/validation.csv",
            bidirectional=True,
            augmentation={"enabled": True, "concat_probability": 1.0},
            batch_size=size,
        )
        data.setup("fit")
        rows = sorted(
            data.splits["train"].rows,
            key=lambda row: len(row["source"]) + len(row["reference"]),
            reverse=True,
        )
        data.splits["train"].rows = rows
        # Both directions of the longest examples exercise the configured caps.
        iterator = iter(
            DataLoader(
                data.splits["train"], batch_size=size, collate_fn=collate_translation
            )
        )
        batches.extend(next(iterator) for _ in range(count))
    validation = collate_translation([data.splits["val"][0], data.splits["val"][1]])
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        max_epochs=1,
        max_steps=10,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        default_root_dir=args.output,
    )
    torch.cuda.reset_peak_memory_stats()
    trainer.fit(model, train_dataloaders=DataLoader(batches, batch_size=None))
    assert trainer.global_step == 10
    # Lightning may move the policy to CPU during fit teardown.
    model.to("cuda:0")
    with trainer.precision_plugin.forward_context():
        predictions = model.translate(
            validation["source"], validation["source_lang"], validation["target_lang"]
        )
    scores = reward.components(
        predictions, validation["reference"], sources=validation["source"]
    )
    assert all(prediction.strip() for prediction in predictions)
    assert set(validation["direction"]) == {"en_vi", "vi_en"}
    assert reward.comet.model is not None
    assert all(
        p.device.type == "cuda" and not p.requires_grad
        for p in reward.comet.model.parameters()
    )
    assert all(p.device.type == "cuda" for p in model.policy.parameters())
    assert any(
        not torch.equal(before[name], p.detach().cpu())
        for name, p in model.policy.named_parameters()
        if name in before
    )
    assert all(
        torch.isfinite(p).all() for p in model.policy.parameters() if p.requires_grad
    )
    with args.checkpoint.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == original_hash
    phases = model.phase_timing.summary()
    assert phases["phases"]["sft/optimizer"]["calls"] == 2
    assert phases["phases"]["translation/optimizer"]["calls"] == 8
    report = {
        "status": "passed",
        "optimizer_steps": 10,
        "sft_updates": 2,
        "grpo_updates": 8,
        "models_on_gpu": ["Qwen-7B LoRA", "Unbabel/wmt22-cometkiwi-da"],
        "source_cap": config["source_cap"],
        "target_cap": config["target_cap"],
        "sft_batch": config["batch_sizes"]["sft"],
        "rl_batch": config["batch_sizes"]["rl"],
        "comet_batch": config["comet_batch_size"],
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "gpu": torch.cuda.get_device_name(),
        "source_checkpoint_unchanged": True,
        "validation": {
            direction: {metric: values[index] for metric, values in scores.items()}
            for index, direction in enumerate(validation["direction"])
        },
        "seconds": time.perf_counter() - started,
        "phases": phases,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "phases"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
