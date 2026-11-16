"""A mid-epoch logger restart must preserve updates and the remaining data."""

import csv

import lightning as L
import pytest
import torch
from torch.utils.data import DataLoader

from src.data.components.resume_sampler import ResumeBatchSampler


class TinyManual(L.LightningModule):
    automatic_optimization = False

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        self.seen = []

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)

    def training_step(self, batch, batch_idx):
        value = batch.float().reshape(-1)[0]
        self.seen.append(int(value))
        optimizer = self.optimizers()
        optimizer.zero_grad()
        loss = (self.weight - value).square()
        self.manual_backward(loss)
        optimizer.step()
        self.log("train/loss", loss)


class InterruptAfterTwo(L.Callback):
    def __init__(self, path):
        self.path = path

    def on_train_batch_start(self, trainer, pl_module, *args):
        if trainer.global_step == 2:
            raise InterruptedError("Restart boundary")

    def on_exception(self, trainer, pl_module, error):
        trainer.save_checkpoint(self.path)


def trainer(tmp_path, **kwargs):
    return L.Trainer(
        accelerator="cpu",
        max_epochs=1,
        max_steps=6,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
        **kwargs,
    )


def test_mid_epoch_resume_preserves_samples_optimizer_and_total_steps(tmp_path):
    batches = [[i] for i in range(6)]
    full = TinyManual()
    trainer(tmp_path / "full").fit(
        full, train_dataloaders=DataLoader(list(range(6)), batch_sampler=batches)
    )
    checkpoint = tmp_path / "interrupted.ckpt"
    interrupted = TinyManual()
    with pytest.raises(InterruptedError):
        trainer(tmp_path / "partial", callbacks=[InterruptAfterTwo(checkpoint)]).fit(
            interrupted,
            train_dataloaders=DataLoader(list(range(6)), batch_sampler=batches),
        )
    saved = torch.load(checkpoint, weights_only=False)
    completed = saved["loops"]["fit_loop"]["epoch_loop.batch_progress"]["current"][
        "completed"
    ]
    assert saved["global_step"] == completed == 2
    resumed = TinyManual()
    live = trainer(tmp_path / "resume")
    sampler = ResumeBatchSampler(batches, completed)
    assert len(sampler) == 6 and list(sampler) == batches[2:]
    live.fit(
        resumed,
        train_dataloaders=DataLoader(list(range(6)), batch_sampler=sampler),
        ckpt_path=checkpoint,
    )
    assert live.global_step == 6
    assert interrupted.seen == [0, 1] and resumed.seen == [2, 3, 4, 5]
    torch.testing.assert_close(resumed.weight, full.weight)
    sampler.set_epoch(1)
    assert list(sampler) == batches


def test_local_test_sample_is_seeded_unique_and_reproducible(tmp_path):
    from scripts.data.sample_test_1000 import sample

    source = tmp_path / "original.csv"
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["en", "vi"])
        writer.writerows((f"English {i}", f"Tiếng Việt {i}") for i in range(20))
        writer.writerow(["English 1", "Tiếng Việt 1"])
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    record_a, record_b = sample(source, a, count=7), sample(source, b, count=7)
    assert record_a["sha256"] == record_b["sha256"]
    assert record_a["pairs"] == 7 and record_a["distinct_normalized_pairs"] == 20
    assert record_a["quality_based_selection"] is False


def test_native_wandb_logger_records_without_rental_network(tmp_path):
    from lightning.pytorch.loggers import WandbLogger

    logger = WandbLogger(
        project="mt-restart-test",
        id="offline-smoke",
        offline=True,
        save_dir=str(tmp_path),
    )
    logger.log_metrics({"train/loss": 1.2}, step=2)
    try:
        assert logger.experiment.settings.mode in {"offline", "dryrun"}
    finally:
        logger.experiment.finish()
