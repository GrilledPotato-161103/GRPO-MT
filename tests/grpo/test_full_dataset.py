"""Full-corpus leakage safeguards and disk-backed logical-record loading."""

import csv
import json
import pickle
from pathlib import Path

import pytest

from scripts.data.prepare_full import prepare_full
from scripts.data.prepare_stages import _normalize
from src.data.components.indexed_translation import IndexedTranslationRows
from src.data.translation_datamodule import TranslationDataModule


def write(path: Path, pairs: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["en", "vi"])
        writer.writerows(pairs)


def test_transitive_groups_test_exclusion_and_determinism(tmp_path: Path) -> None:
    train, test = tmp_path / "train.csv", tmp_path / "test.csv"
    chain = [
        ("same source", "target A"),
        ("same source", "target B"),
        ("other source", "target B"),
    ]
    excluded = [("test source", "overlap A"), ("linked source", "overlap A")]
    write(
        train,
        chain
        + excluded
        + [(" SAME  source ", "TARGET a")]
        + [(f"source {i}", f"target {i}") for i in range(30)]
        + [("", "invalid")],
    )
    write(test, [("test source", "official target")])
    original_train, original_test = train.read_bytes(), test.read_bytes()
    a = prepare_full(train, test, tmp_path / "first", validation_pairs=3)
    prepare_full(train, test, tmp_path / "second", validation_pairs=3)
    assert a["counts"]["duplicate_training_pairs"] == 1
    assert a["counts"]["invalid_training_rows"] == 1
    assert a["counts"]["test_overlapping_training_pairs_removed"] == 2
    assert train.read_bytes() == original_train and test.read_bytes() == original_test
    splits = {}
    for stage in ("sft", "rl", "validation"):
        assert (tmp_path / "first" / f"{stage}.csv").read_bytes() == (
            tmp_path / "second" / f"{stage}.csv"
        ).read_bytes()
        with (tmp_path / "first" / f"{stage}.csv").open(encoding="utf-8") as stream:
            splits[stage] = list(csv.DictReader(stream))
    for field in ("en", "vi"):
        values = [{_normalize(row[field]) for row in rows} for rows in splits.values()]
        assert all(not values[i] & values[j] for i in range(3) for j in range(i))
    assert (
        sum(any(row["en"] == "same source" for row in rows) for rows in splits.values())
        == 1
    )
    assert sum(a["counts"][k] for k in ("sft", "rl", "validation")) == 33
    with pytest.raises(FileExistsError):
        prepare_full(train, test, tmp_path / "first", validation_pairs=3)


def test_indexed_csv_multiline_unicode_pickle_and_stale_index(tmp_path: Path) -> None:
    path = tmp_path / "pairs.csv"
    write(
        path,
        [('a,"quoted"\nline', "Tiếng Việt\nnext"), ("another", "\ufeffliteral BOM")],
    )
    rows = IndexedTranslationRows(path)
    assert rows[0] == {"source": 'a,"quoted"\nline', "reference": "Tiếng Việt\nnext"}
    assert rows[-1]["reference"] == "\ufeffliteral BOM"
    assert len(pickle.dumps(rows)) < 4096
    restored = pickle.loads(pickle.dumps(rows))
    assert list(restored) == list(rows)
    rows.close()
    restored.close()
    write(path, [("replacement", "mới")])
    changed = IndexedTranslationRows(path)
    assert len(changed) == 1 and changed[0]["source"] == "replacement"
    with pytest.raises(IndexError):
        _ = changed[1]
    changed.close()


def test_indexed_jsonl_blanks_nested_and_invalid(tmp_path: Path) -> None:
    path = tmp_path / "pairs.jsonl"
    path.write_text(
        "\n" + json.dumps({"translation": {"en": "hello", "vi": "xin chào"}}) + "\n\n",
        encoding="utf-8",
    )
    rows = IndexedTranslationRows(path, translation_column="translation")
    assert rows[:1] == [{"source": "hello", "reference": "xin chào"}]
    rows.close()
    bad = tmp_path / "bad.csv"
    write(bad, [("", "missing")])
    with pytest.raises(ValueError):
        IndexedTranslationRows(bad)


def test_disk_backed_bidirectional_loader_and_explicit_validation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "pairs.csv"
    write(path, [("hello", "xin chào"), ("good day", "ngày tốt")])
    data = TranslationDataModule(
        train_file=str(path),
        val_file=str(path),
        test_file=str(path),
        disk_backed=True,
        bidirectional=True,
        num_workers=0,
        batch_size=2,
    )
    data.setup("fit")
    assert isinstance(data.splits["train"].rows, IndexedTranslationRows)
    assert len(data.splits["train"]) == 4
    assert data.splits["train"][1]["source"] == "xin chào"
    assert next(iter(data.val_dataloader()))["direction"] == ["en_vi", "vi_en"]
    data.setup("test")
    assert len(data.splits["test"]) == 4
    bad = TranslationDataModule(train_file=str(path), disk_backed=True)
    with pytest.raises(ValueError, match="explicit validation"):
        bad.setup("fit")
