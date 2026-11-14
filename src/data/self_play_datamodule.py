"""Synthetic task-proposal contexts with independent parallel validation/test data."""

from typing import Any

import lightning as L
from torch.utils.data import DataLoader, Dataset

from src.data.translation_datamodule import TranslationDataModule


class TopicDataset(Dataset[dict[str, str]]):
    """Deterministic topic contexts; proposed source sentences come from the policy."""

    def __init__(self, topics: list[str], size: int) -> None:
        self.topics, self.size = topics, size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, str]:
        return {"topic": self.topics[index % len(self.topics)]}


class SelfPlayDataModule(L.LightningDataModule):
    """Generate training contexts without using held-out text for proposals."""

    def __init__(
        self,
        topics: list[str],
        val_file: str,
        test_file: str,
        steps_per_epoch: int = 100,
        batch_size: int = 1,
        source_column: str = "en",
        target_column: str = "vi",
        seed: int = 42,
        bidirectional: bool = False,
        source_lang: str = "English",
        target_lang: str = "Vietnamese",
    ) -> None:
        super().__init__()
        if not topics or any(not isinstance(t, str) or not t.strip() for t in topics):
            raise ValueError("Provide at least one nonempty proposal topic")
        if min(steps_per_epoch, batch_size) < 1:
            raise ValueError("Self-play step count and batch size must be positive")
        self.contexts = TopicDataset(list(topics), steps_per_epoch * batch_size)
        self.batch_size = batch_size
        self.evaluation = TranslationDataModule(
            train_file=val_file,
            val_file=val_file,
            test_file=test_file,
            source_column=source_column,
            target_column=target_column,
            batch_size=batch_size,
            seed=seed,
            bidirectional=bidirectional,
            source_lang=source_lang,
            target_lang=target_lang,
        )

    def setup(self, stage: str | None = None) -> None:
        """Prepare only independent evaluation splits needed by the stage."""
        self.evaluation.setup(stage)

    def train_dataloader(self) -> DataLoader[Any]:
        """Return deterministic topic batches; policy sampling creates the tasks."""
        return DataLoader(self.contexts, batch_size=self.batch_size, num_workers=0)

    def val_dataloader(self) -> DataLoader[Any]:
        """Return external parallel development data."""
        return self.evaluation.val_dataloader()

    def test_dataloader(self) -> DataLoader[Any]:
        """Return external held-out parallel test data."""
        return self.evaluation.test_dataloader()
