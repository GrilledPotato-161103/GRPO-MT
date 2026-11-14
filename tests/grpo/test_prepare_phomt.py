"""Verify that materialized development files remain disjoint and preserve inputs."""

import csv
from pathlib import Path

import pytest

from scripts.data.prepare_phomt import prepare


def test_prepare_preserves_originals_and_keeps_test_disjoint(tmp_path: Path) -> None:
    train, test = tmp_path / "train.csv", tmp_path / "test.csv"
    train.write_text("en,vi\none,một\ntwo,hai\nthree,ba\nfour,bốn\n", encoding="utf-8")
    test.write_text("en,vi\nfive,năm\n", encoding="utf-8")
    before = train.read_bytes(), test.read_bytes()
    output = tmp_path / "prepared"
    assert prepare(train, test, output, val_fraction=0.25) == {
        "train": 3,
        "validation": 1,
        "test": 1,
        "overlap": 0,
    }
    with (output / "train.csv").open(encoding="utf-8") as stream:
        a = {row["en"] for row in csv.DictReader(stream)}
    with (output / "validation.csv").open(encoding="utf-8") as stream:
        b = {row["en"] for row in csv.DictReader(stream)}
    assert not a & b and len(a | b) == 4
    assert before == (train.read_bytes(), test.read_bytes())
    with pytest.raises(FileExistsError):
        prepare(train, test, output)
    test.write_text("en,vi\none,một\n", encoding="utf-8")
    with pytest.raises(ValueError, match="overlap"):
        prepare(train, test, tmp_path / "overlap")
