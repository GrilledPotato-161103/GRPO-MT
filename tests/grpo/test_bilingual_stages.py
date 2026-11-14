"""Stage isolation, online augmentation, bilingual prompts, and bidirectional evaluation."""

import csv
import json
import random
import unicodedata
from pathlib import Path

import lightning as L
import pytest
from hydra import compose, initialize

from scripts.data.prepare_stages import prepare
from scripts.verify.assets import create_assets
from src.data.components.augmentation import TranslationAugmentation
from src.data.translation_datamodule import TranslationDataModule, TranslationDataset
from src.models.components.translation_reward import TranslationReward
from src.models.grpo_module import GRPOModule


def write_pairs(path: Path, prefix: str, count: int) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["en", "vi"])
        writer.writeheader()
        writer.writerows(
            {
                "en": f"{prefix} English sentence number {index}.",
                "vi": f"{prefix} Câu tiếng Việt số {index}.",
            }
            for index in range(count)
        )


def test_stage_split_precedes_augmentation_and_is_disjoint_in_both_languages(
    tmp_path: Path,
) -> None:
    train, val, test = (tmp_path / f"{name}.csv" for name in ("train", "val", "test"))
    for path, prefix, count in (
        (train, "train", 10),
        (val, "dev", 2),
        (test, "test", 3),
    ):
        write_pairs(path, prefix, count)
    original = {path: path.read_bytes() for path in (train, val, test)}
    report = prepare(train, val, test, tmp_path / "stages")
    assert report["counts"] == {
        "training_pool": 10,
        "sft": 8,
        "rl": 2,
        "validation": 2,
        "test_pairs": 3,
        "test_directional_examples": 6,
    }
    assert all(path.read_bytes() == content for path, content in original.items())
    prepare(train, val, test, tmp_path / "repeat")
    for stage in ("sft", "rl"):
        assert (tmp_path / "stages" / f"{stage}.csv").read_bytes() == (
            tmp_path / "repeat" / f"{stage}.csv"
        ).read_bytes()
    loaded = []
    for stage in ("sft", "rl"):
        data = TranslationDataModule(
            train_file=str(tmp_path / "stages" / f"{stage}.csv"),
            val_file=str(val),
            test_file=str(test),
            bidirectional=True,
            augmentation={"enabled": True, "concat_probability": 1.0},
        )
        data.setup(None)
        assert len(data.splits["test"]) == 6
        assert data.splits["test"].augmentation is None
        assert data.splits["val"].augmentation is None
        assert data.splits["train"][0]["source_lang"] == "English"
        assert data.splits["train"][1]["target_lang"] == "English"
        loaded.append(data.splits["train"].rows)
    for key in ("source", "reference"):
        assert {row[key] for row in loaded[0]}.isdisjoint(row[key] for row in loaded[1])
    with pytest.raises(FileExistsError):
        prepare(train, val, test, tmp_path / "stages")


@pytest.mark.parametrize("column", ["en", "vi"])
def test_overlap_in_either_language_is_rejected(tmp_path: Path, column: str) -> None:
    paths = [tmp_path / f"{name}.csv" for name in ("train", "val", "test")]
    for index, path in enumerate(paths):
        write_pairs(path, str(index), 3)
    with paths[1].open(encoding="utf-8") as stream:
        dev = list(csv.DictReader(stream))
    dev[0][column] = (
        f"0 {'English sentence number' if column == 'en' else 'Câu tiếng Việt số'} 0."
    )
    with paths[1].open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["en", "vi"])
        writer.writeheader()
        writer.writerows(dev)
    with pytest.raises(ValueError, match="overlap"):
        prepare(*paths, tmp_path / "stages")


def test_online_augmentation_preserves_alignment_and_refreshes(tmp_path: Path) -> None:
    pairs = [
        {"source": "Hello 123.", "reference": "Xin chào 123."},
        {"source": "Book 456.", "reference": "Sách 456."},
    ]
    augment = TranslationAugmentation(
        whitespace_probability=1, unicode_probability=1, concat_probability=1
    )
    first = TranslationDataset(pairs, bidirectional=True, augmentation=augment, seed=19)
    second = TranslationDataset(
        pairs, bidirectional=True, augmentation=augment, seed=19
    )
    forward = first[0]
    reverse = first[1]
    assert forward["reference"] == "Xin chào 123. Sách 456."
    assert reverse["reference"] == "Hello 123. Book 456."
    assert reverse["clean_source"] == "Xin chào 123. Sách 456."
    assert (
        " ".join(unicodedata.normalize("NFC", reverse["source"]).split())
        == reverse["clean_source"]
    )
    for index in (0, 1):
        repeated = second[index]
        expected = forward if index == 0 else reverse
        assert {
            key: value
            for key, value in repeated.items()
            if key != "augmentation_seconds"
        } == {
            key: value
            for key, value in expected.items()
            if key != "augmentation_seconds"
        }
    assert len({first[1]["source"] for _ in range(12)}) > 1
    short = TranslationAugmentation(concat_probability=1, max_concat_chars=2).apply(
        "hello", "xin chào", ("book", "sách"), random.Random(1)
    )
    assert short["reference"] == "xin chào"
    with pytest.raises(ValueError):
        TranslationAugmentation(unicode_probability=2)
    with pytest.raises(IndexError):
        _ = first[len(first)]


def test_bilingual_prompts_and_exact_per_direction_corpus_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "test.csv"
    write_pairs(path, "held out", 2)
    with path.open(encoding="utf-8") as stream:
        pairs = list(csv.DictReader(stream))
    mapping = {row["en"]: row["vi"] for row in pairs} | {
        row["vi"]: row["en"] for row in pairs
    }
    module = GRPOModule(
        model_name_or_path=str(create_assets(tmp_path / "model", "seq2seq")),
        save_predictions=True,
        reward=TranslationReward(
            chrf_weight=0,
            bleu_weight=0.3,
            comet_weight=0.7,
            comet_scorer=lambda sources, predictions, references: (
                [0.8] * len(predictions)
            ),
        ),
    )
    seen: list[tuple[str, str]] = []

    def translate(
        sources: list[str], source_langs: list[str], target_langs: list[str]
    ) -> list[str]:
        seen.extend(zip(source_langs, target_langs, strict=True))
        return [mapping[source] for source in sources]

    monkeypatch.setattr(module, "translate", translate)
    prompts = module._format_prompts(
        ["hello", "xin chào"], ["English", "Vietnamese"], ["Vietnamese", "English"]
    )
    assert (
        "English to Vietnamese" in prompts[0] and "Vietnamese to English" in prompts[1]
    )
    data = TranslationDataModule(
        train_file=str(path),
        val_file=str(path),
        test_file=str(path),
        bidirectional=True,
        batch_size=2,
    )
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        default_root_dir=tmp_path,
    )
    trainer.test(module, datamodule=data)
    report = json.loads((tmp_path / "translation_metrics/test.json").read_text())
    assert report["counts"] == {"en_vi": 2, "vi_en": 2}
    assert set(seen) == {("English", "Vietnamese"), ("Vietnamese", "English")}
    for direction in ("en_vi", "vi_en"):
        assert report["directions"][direction]["bleu"] == pytest.approx(100)
        assert report["directions"][direction]["comet"] == pytest.approx(0.8)
    assert float(trainer.callback_metrics["test/chrf_macro"]) == pytest.approx(100)
    prediction_dir = tmp_path / "translation_metrics"
    saved = [
        json.loads(line)
        for line in (prediction_dir / "test_predictions.jsonl").read_text().splitlines()
    ]
    with (prediction_dir / "test_predictions.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        table = list(csv.DictReader(stream))
    assert len(saved) == len(table) == 4
    assert [row["example_index"] for row in saved] == list(range(4))
    assert [row["direction"] for row in saved] == ["en_vi", "vi_en"] * 2
    assert all(row["source"] and row["prediction"] == row["reference"] for row in saved)
    assert [row["prediction"] for row in saved] == [row["prediction"] for row in table]
    trainer.test(module, datamodule=data)
    assert (
        len((prediction_dir / "test_predictions.jsonl").read_text().splitlines()) == 4
    )


def test_new_stage_configurations_are_bilingual_and_profiled() -> None:
    with initialize(version_base="1.3", config_path="../../configs"):
        for config, experiment in (
            ("sft", "phomt_sft"),
            ("grpo", "phomt_rl"),
            ("sft_eval", "phomt_sft"),
            ("grpo_eval", "phomt_rl"),
        ):
            cfg = compose(config_name=config, overrides=[f"experiment={experiment}"])
            assert cfg.data.bidirectional and cfg.data.augmentation.enabled
            assert cfg.model.reward.comet_model_name == "Unbabel/wmt22-cometkiwi-da"
            assert cfg.model.profiling.enabled
            assert (
                cfg.callbacks.model_checkpoint.monitor == "val/chrf_macro"
                if "callbacks" in cfg
                else True
            )
