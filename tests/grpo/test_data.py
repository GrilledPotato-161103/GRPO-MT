"""Translation split and parsing tests; no network or user datasets required."""

import json
from pathlib import Path

import pytest

from src.data.translation_datamodule import TranslationDataModule


def test_csv_split_is_deterministic_and_test_is_held_out(tmp_path: Path) -> None:
    train = tmp_path / "train.csv"
    train.write_text(
        "en,vi\n" + "\n".join(f"source {i},target {i}" for i in range(10)),
        encoding="utf-8",
    )
    test = tmp_path / "test.jsonl"
    test.write_text(json.dumps({"en": "held out", "vi": "reference"}), encoding="utf-8")
    first = TranslationDataModule(
        train_file=str(train), test_file=str(test), val_fraction=0.2
    )
    second = TranslationDataModule(
        train_file=str(train), test_file=str(test), val_fraction=0.2
    )
    first.setup("fit")
    second.setup("fit")
    assert first.splits["val"].rows == second.splits["val"].rows
    assert len(first.splits["val"]) == 2
    assert len(first.splits["train"]) == 8
    assert {x["source"] for x in first.splits["train"].rows}.isdisjoint(
        x["source"] for x in first.splits["val"].rows
    )
    assert "test" not in first.splits
    assert len(next(iter(first.train_dataloader()))["source"]) == 2
    assert len(next(iter(first.val_dataloader()))["source"]) == 2
    first.setup("test")
    assert next(iter(first.test_dataloader()))["source"] == ["held out"]


def test_nested_json_and_explicit_validation(tmp_path: Path) -> None:
    path = tmp_path / "pairs.jsonl"
    path.write_text(
        json.dumps({"translation": {"en": "hello", "vi": "xin chao"}}), encoding="utf-8"
    )
    data = TranslationDataModule(
        train_file=str(path), val_file=str(path), translation_column="translation"
    )
    data.setup("validate")
    assert data.splits["val"][0] == {"source": "hello", "reference": "xin chao"}


@pytest.mark.parametrize(
    "content,suffix", [("", ".csv"), ('{"en":"","vi":"x"}', ".jsonl"), ("text", ".txt")]
)
def test_invalid_data_fails_clearly(tmp_path: Path, content: str, suffix: str) -> None:
    path = tmp_path / f"bad{suffix}"
    path.write_text(content, encoding="utf-8")
    data = TranslationDataModule(train_file=str(path))
    with pytest.raises(ValueError):
        data.setup("fit")


def test_invalid_configuration() -> None:
    with pytest.raises(ValueError):
        TranslationDataModule()
    with pytest.raises(ValueError):
        TranslationDataModule(train_file="unused", batch_size=0)
