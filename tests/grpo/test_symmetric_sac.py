"""Paired bilingual rollouts, symmetric endpoint rewards and aligned augmentation."""

import json
from pathlib import Path
from typing import Any

import lightning as L
import pytest
import torch
from lightning.fabric.utilities.data import _set_sampler_epoch

from scripts.main.common import ROOT, compose_stage, load_config
from scripts.verify.assets import create_assets
from src.data.translation_datamodule import TranslationDataModule
from src.models.components.grpo import group_advantages
from src.models.components.reward_diagnostics import reward_diagnostics
from src.models.components.symmetric_alignment import paired_candidate_indices
from src.models.components.translation_reward import TranslationReward
from src.models.grpo_module import GRPOModule


class EndpointScores:
    def evaluate(
        self, sources: list[str], candidates: list[str], src: list[str], tgt: list[str]
    ) -> list[float]:
        assert len(candidates) == 8
        direct = {"vi0": 1.0, "vi1": 0.7, "en0": 0.3, "en1": 1.0}
        endpoint = {"vi0": 1.0, "vi1": 0.3, "en0": 0.7, "en1": 1.0}
        return [
            (endpoint if a == b else direct)[candidate]
            for candidate, a, b in zip(candidates, src, tgt, strict=True)
        ]


def test_penalty_is_symmetric_and_matches_both_original_endpoints() -> None:
    reward = TranslationReward(
        chrf_weight=0,
        comet_weight=1,
        comet_scorer=lambda *args: [0.8] * 4,
        sac={
            "enabled": True,
            "consistency_mode": "symmetric_endpoints",
            "consistency_weight": 0.2,
        },
        sac_scorer=EndpointScores(),
    )
    predictions = ["vi0", "vi1", "en0", "en1"]
    kwargs = {
        "sources": ["original_en"] * 2 + ["original_vi"] * 2,
        "source_langs": ["English"] * 2 + ["Vietnamese"] * 2,
        "target_langs": ["Vietnamese"] * 2 + ["English"] * 2,
        "partner_indices": [2, 3, 0, 1],
    }
    references = ["original_vi"] * 2 + ["original_en"] * 2
    result = reward.components(predictions, references, **kwargs)
    assert result["endpoint_sac"] == [1.0, 0.3, 0.7, 1.0]
    assert result["symmetric_penalty"] == pytest.approx([0.03, 0.07, 0.03, 0.07])
    assert result["reward"] == pytest.approx([0.77, 0.49, 0.21, 0.73])
    assert "round_trip_sac" not in result
    # Swapping language order cannot change a pair's penalty or reward.
    order = [2, 3, 0, 1]
    swapped = reward.components(
        [predictions[i] for i in order],
        [references[i] for i in order],
        **{
            key: value if key == "partner_indices" else [value[i] for i in order]
            for key, value in kwargs.items()
        },
    )
    assert swapped["reward"] == pytest.approx([result["reward"][i] for i in order])
    with pytest.raises(ValueError, match="reversed pairs"):
        reward.components(predictions, ["unrelated"] * 4, **kwargs)
    with pytest.raises(ValueError, match="partner"):
        reward.components(
            predictions, references, **{**kwargs, "partner_indices": [0, 1, 2, 3]}
        )


def test_candidate_pairs_survive_row_reordering_and_group_expansion() -> None:
    batch = {
        "source": ["viB", "enA", "enB", "viA"],
        "reference": ["enB", "viA", "viB", "enA"],
        "source_lang": ["Vietnamese", "English", "English", "Vietnamese"],
        "target_lang": ["English", "Vietnamese", "Vietnamese", "English"],
        "pair_id": ["B", "A", "B", "A"],
    }
    assert paired_candidate_indices(batch, 2) == [4, 5, 6, 7, 0, 1, 2, 3]
    with pytest.raises(ValueError, match="exactly two"):
        paired_candidate_indices({**batch, "pair_id": ["B", "A", "C", "D"]}, 2)
    with pytest.raises(ValueError, match="complete"):
        paired_candidate_indices({k: v for k, v in batch.items() if k != "pair_id"}, 2)


def test_endpoint_penalty_survives_grpo_group_centering() -> None:
    class Score:
        def evaluate(self, *args: Any) -> list[float]:
            return [1.0] * 4 + [1.0, 0.3, 1.0, 0.3]

    reward = TranslationReward(
        chrf_weight=0,
        comet_weight=1,
        comet_scorer=lambda *args: [0.8] * 4,
        sac={
            "enabled": True,
            "consistency_mode": "symmetric_endpoints",
            "consistency_weight": 0.2,
        },
        sac_scorer=Score(),
    )
    scores = reward.components(
        ["v0", "v1", "e0", "e1"],
        ["v", "v", "e", "e"],
        sources=["e", "e", "v", "v"],
        source_langs=["English"] * 2 + ["Vietnamese"] * 2,
        target_langs=["Vietnamese"] * 2 + ["English"] * 2,
        partner_indices=[2, 3, 0, 1],
    )
    advantages = group_advantages(torch.tensor(scores["reward"]), 2)
    assert advantages[0] > 0 > advantages[1]
    assert advantages[2] > 0 > advantages[3]


@pytest.mark.parametrize("length_buckets", [False, True])
@pytest.mark.parametrize("workers", [0, 2])
def test_pair_batches_and_concatenation_remain_aligned(
    tmp_path: Path, length_buckets: bool, workers: int
) -> None:
    asset = create_assets(tmp_path / "base", "causal")
    path = tmp_path / "pairs.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"en": "Hello " * n, "vi": "Xin chao " * n}) for n in range(1, 8)
        ),
        encoding="utf-8",
    )
    data = TranslationDataModule(
        train_file=str(path),
        val_file=str(path),
        test_file=str(path),
        batch_size=4,
        bidirectional=True,
        paired_batches=True,
        num_workers=workers,
        persistent_workers=workers > 0,
        augmentation={
            "enabled": True,
            "concat_probability": 1.0,
            "unicode_probability": 1.0,
            "whitespace_probability": 1.0,
        },
        length_batching={
            "enabled": length_buckets,
            "tokenizer_name_or_path": str(asset),
            "source_cap": 64,
            "target_cap": 32,
            "candidate_multiplier": 2,
            "reserve_target_tokens": 32,
            "max_tokens": 1536,
        },
    )
    data.setup(None)
    loader = data.train_dataloader()
    seen = []
    for batch in loader:
        partners = paired_candidate_indices(batch, 1)
        assert len(partners) % 2 == 0
        assert all("concat" in value for value in batch["augmentation"])
        seen.extend(batch["pair_id"])
    assert sorted(seen) == sorted([str(i) for i in range(7)] * 2)
    first = list(loader.batch_sampler)
    _set_sampler_epoch(loader, 1)
    assert list(loader.batch_sampler) != first
    for split in (data.val_dataloader(), data.test_dataloader()):
        for batch in split:
            assert "augmentation" not in batch
            paired_candidate_indices(batch, 1)


def test_penalty_conflicts_use_the_negative_reward_effect() -> None:
    result = reward_diagnostics(
        torch.tensor([0.9, 0.1]),
        {"comet": [0.9, 0.3], "symmetric_penalty": [0.0, 0.2]},
        2,
        ["good", "bad"],
        ["en_vi"],
    )
    assert result["reward_conflict/comet_symmetric_penalty_disagreement"] == 0
    assert result["direction/en_vi/symmetric_penalty"] == pytest.approx(0.1)


def test_profile_requires_even_complete_pairs() -> None:
    cfg = load_config(ROOT / "configs/runs/rtx4090.yaml", ["wandb.enabled=false"])
    assert (
        cfg["paired_batches"]
        and cfg["sac_reward"]["consistency_mode"] == "symmetric_endpoints"
    )
    assert compose_stage(cfg, "rl", None).data.paired_batches
    for override in (
        "batch_sizes.rl=3",
        "paired_batches=false",
        "evaluation.batch_size=3",
    ):
        with pytest.raises(ValueError, match="paired_batches"):
            load_config(ROOT / "configs/runs/rtx4090.yaml", [override])


def test_real_grpo_has_one_generation_pass_and_no_back_translation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.set_num_threads(1)
    L.seed_everything(42)
    assets = create_assets(tmp_path / "base", "causal")

    class FullMatch:
        def __init__(self) -> None:
            self.last_metrics: dict[str, float] = {}

        def check_ready(self) -> None:
            pass

        def evaluate(
            self,
            sources: list[str],
            candidates: list[str],
            src: list[str],
            tgt: list[str],
        ) -> list[float]:
            return [1.0] * len(candidates)

    reward = TranslationReward(
        chrf_weight=0,
        comet_weight=1,
        comet_scorer=lambda s, p, r: [0.4 + 0.2 * (i % 2) for i in range(len(p))],
        sac={
            "enabled": True,
            "consistency_mode": "symmetric_endpoints",
            "consistency_weight": 0.2,
            "fusion": {"enabled": True, "total_steps": 2},
        },
        sac_scorer=FullMatch(),
    )
    model = GRPOModule(
        model_name_or_path=str(assets),
        reward=reward,
        beta=0,
        max_source_length=40,
        max_new_tokens=4,
        group_size=2,
        num_iterations=2,
        learning_rate=1e-3,
        profiling={"enabled": True},
        save_predictions=True,
    )
    data = TranslationDataModule(
        train_file=str(ROOT / "data/example_train.jsonl"),
        val_file=str(ROOT / "data/example_val.jsonl"),
        test_file=str(ROOT / "data/example_test.jsonl"),
        batch_size=4,
        bidirectional=True,
        paired_batches=True,
        num_workers=0,
    )
    before = {name: p.clone() for name, p in model.policy.named_parameters()}
    generate, translate = model._generate_policy, model.translate
    calls = []

    def counted(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return generate(*args, **kwargs)

    def no_second_pass(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(
            "No reverse translation pass is permitted in symmetric mode"
        )

    monkeypatch.setattr(model, "_generate_policy", counted)
    monkeypatch.setattr(model, "translate", no_second_pass)
    trainer = L.Trainer(
        default_root_dir=tmp_path,
        accelerator="cpu",
        devices=1,
        max_steps=2,
        limit_train_batches=1,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, datamodule=data)
    assert trainer.global_step == 2 and len(calls) == 1
    updates = [
        json.loads(line)
        for line in (tmp_path / "diagnostics/updates.jsonl").read_text().splitlines()
    ]
    assert [
        row["metrics"]["reward/fusion_alpha_mean"] for row in updates
    ] == pytest.approx([0.1, 0.1])
    checkpoint = tmp_path / "fusion.ckpt"
    trainer.save_checkpoint(checkpoint)
    restored = GRPOModule.load_from_checkpoint(
        checkpoint, reward=reward, weights_only=False
    )
    assert restored._reward_progress() == 2
    assert reward.fusion.alpha(restored._reward_progress()) == pytest.approx(0.9)
    restored.hparams.update(cycle_stage="rl", stage_update_offset=100)
    restored._loaded_optimizer_step = 9000
    restored._reward_updates_seen = 120
    assert (
        restored._reward_progress() == 120
    )  # SFT/total counters cannot advance an RL curriculum.
    restored.hparams["cycle_stage"] = "sft"
    assert restored._reward_progress() == 0
    assert any(
        not torch.equal(before[name], p) for name, p in model.policy.named_parameters()
    )
    assert all(
        "round_trip_generation" not in event["phase"]
        for event in model.phase_timing.events
    )
    monkeypatch.setattr(model, "translate", translate)
    evaluator = L.Trainer(
        default_root_dir=tmp_path,
        accelerator="cpu",
        devices=1,
        limit_val_batches=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    evaluator.validate(model, datamodule=data)
    assert len(calls) == 2
    report = json.loads((tmp_path / "translation_metrics/val.json").read_text())
    assert "val/symmetric_sac_macro" in report["metrics"]
    assert report["metrics"]["val/fusion_alpha_macro"] == pytest.approx(0.9)
    assert report["metrics"]["val/fusion_normalized_macro"] == 0
    assert set(report["directions"]) == {"en_vi", "vi_en"}
