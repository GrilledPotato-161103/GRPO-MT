"""Batch packing, augmentation isolation and exact microbatch policy gradients."""

import json
from pathlib import Path
from typing import Any

import lightning as L
import pytest
import torch
from hydra import compose, initialize
from lightning.fabric.utilities.data import _set_sampler_epoch
from transformers import Qwen2Config, Qwen2ForCausalLM

from scripts.verify.assets import create_assets
from src.data.components.length_batching import (
    LengthBucketBatchSampler,
    TranslationCollator,
    padded_length_batches,
)
from src.data.translation_datamodule import TranslationDataModule
from src.models.components.grpo import completion_mask, group_advantages, grpo_loss
from src.models.grpo_module import GRPOModule


def test_buckets_shuffle_batches_cover_once_and_reduce_padding() -> None:
    lengths = [(length, 8) for _ in range(8) for length in (8, 128, 16, 256)]
    sampler = LengthBucketBatchSampler(lengths, 4, bucket_width=8, seed=42)
    first = list(sampler)
    assert sorted(i for batch in first for i in batch) == list(range(len(lengths)))
    assert all(len({lengths[i] for i in batch}) == 1 for batch in first)
    assert len(first) == len(sampler) == 8
    assert list(sampler) == first
    sampler.set_epoch(1)
    assert list(sampler) != first
    assert list(LengthBucketBatchSampler(lengths, 4, bucket_width=8, seed=42)) == first
    padded = sum(
        len(batch) * (max(lengths[i][0] for i in batch) + 8) for batch in first
    )
    naive = sum(
        4 * (max(source for source, _ in lengths[i : i + 4]) + 8)
        for i in range(0, len(lengths), 4)
    )
    assert padded < naive / 2


def test_sampler_budget_includes_candidates_and_reserved_completion() -> None:
    sampler = LengthBucketBatchSampler(
        [(8, 2), (12, 4), (60, 3), (64, 2), (100, 4)],
        4,
        token_budget=512,
        candidate_multiplier=2,
        reserve_target_tokens=64,
    )
    batches = list(sampler)
    assert sorted(i for batch in batches for i in batch) == list(range(5))
    assert all(
        2 * len(batch) * (max(sampler.lengths[i][0] for i in batch) + 64) <= 512
        for batch in batches
    )
    assert any(len(batch) < 4 for batch in batches)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_size": 0},
        {"bucket_width": 0},
        {"candidate_multiplier": 0},
        {"reserve_target_tokens": -1},
        {"token_budget": 5},
    ],
)
def test_sampler_rejects_invalid_bounds(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        LengthBucketBatchSampler([(10, 10)], **{"batch_size": 4, **kwargs})
    with pytest.raises(ValueError):
        padded_length_batches([(0, 1)], 4)


@pytest.mark.parametrize("workers", [0, 2])
def test_collator_and_loader_preserve_directions_and_augmentation_rng(
    tmp_path: Path,
    workers: int,
) -> None:
    asset = create_assets(tmp_path / "tiny", "causal")
    path = tmp_path / "pairs.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"en": "Hello " * n, "vi": "Xin chào " * n})
            for n in range(1, 13)
        ),
        encoding="utf-8",
    )
    data = TranslationDataModule(
        train_file=str(path),
        val_file=str(path),
        test_file=str(path),
        batch_size=4,
        num_workers=workers,
        persistent_workers=True,
        prefetch_factor=2,
        bidirectional=True,
        augmentation={"enabled": True},
        length_batching={
            "enabled": True,
            "tokenizer_name_or_path": str(asset),
            "source_cap": 128,
            "target_cap": 64,
            "bucket_width": 4,
        },
    )
    data.setup("fit")
    dataset = data.splits["train"]
    assert dataset._rng is None
    loader = data.train_dataloader()
    assert dataset._rng is None  # Estimating lengths must not consume augmentation.
    _set_sampler_epoch(loader, 3)
    assert loader.batch_sampler.epoch == 3
    rows = [
        row
        for batch in loader
        for row in zip(
            batch["source_lang"], batch["target_lang"], batch["direction"], strict=True
        )
    ]
    assert len(rows) == 24
    assert rows.count(("English", "Vietnamese", "en_vi")) == 12
    assert rows.count(("Vietnamese", "English", "vi_en")) == 12
    assert (dataset._rng is not None) == (workers == 0)
    # Evaluation keeps dataset order and is never augmented or length shuffled.
    val = next(iter(data.val_dataloader()))
    assert val["direction"] == ["en_vi", "vi_en", "en_vi", "vi_en"]
    assert "augmentation_seconds" not in val
    collator = loader.collate_fn
    assert isinstance(collator, TranslationCollator)
    supplied = [dataset.unaugmented_item(i) for i in (8, 0, 4)]
    collated = collator(supplied)
    assert collated["source"] == sorted([row["source"] for row in supplied], key=len)
    with pytest.raises(ValueError):
        collator([])


@pytest.mark.parametrize(
    "options",
    [
        {"tokenizer_name_or_path": None},
        {"source_cap": 0},
        {"candidate_multiplier": True},
        {"max_tokens": -1},
        {"reserve_target_tokens": -1},
    ],
)
def test_length_loader_config_validation(options: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        TranslationDataModule(
            train_file="unused",
            length_batching={
                "enabled": True,
                "tokenizer_name_or_path": "local",
                "source_cap": 128,
                "target_cap": 64,
                **options,
            },
        )
    with pytest.raises(ValueError):
        TranslationDataModule(
            train_file="unused",
            resume_skip_examples=1,
            length_batching={"enabled": True},
        )


def scoring_example(module: GRPOModule) -> dict[str, Any]:
    inputs = module._encode(
        ["Hello", "Good morning", "Hello " * 10, "Thank you", "Good " * 7, "Hello"]
    )
    targets = torch.tensor(
        [
            [8, 1, 0, 0, 0],
            [9, 10, 1, 0, 0],
            [1, 0, 0, 0, 0],
            [12, 11, 10, 1, 0],
            [13, 14, 15, 17, 1],
            [8, 9, 1, 0, 0],
        ],
        device=module.device,
    )
    if module.hparams["forced_bos_token_id"] is not None:
        targets = torch.cat([torch.full_like(targets[:, :1], 13), targets], dim=1)
    mask = completion_mask(targets, module.generation_config.eos_token_id)
    if module.hparams["forced_bos_token_id"] is not None:
        mask[:, 0] = False
    prefix = (
        torch.zeros_like(targets[:, :1])
        if module.is_encoder_decoder
        else inputs["input_ids"]
    )
    sequences = torch.cat([prefix, targets], dim=1)
    with torch.no_grad():
        scores = module._score_batch(module.policy, inputs, sequences, mask)
    return {
        "inputs": inputs,
        "sequences": sequences,
        "mask": mask,
        "old_logps": scores
        + torch.linspace(-0.5, 0.5, 6, device=module.device)[:, None],
        "reference_logps": scores + 0.1,
        # Groups of three deliberately cross the scoring chunk boundary.
        "advantages": group_advantages(
            torch.tensor([0.0, 0.2, 1.0, 0.0, 0.6, 0.8], device=module.device), 3
        ),
    }


@pytest.mark.parametrize(
    "architecture,forced,lora",
    [
        ("seq2seq", False, False),
        ("seq2seq", True, False),
        ("causal", False, False),
        ("qwen", False, True),
    ],
)
def test_microbatch_loss_gradients_metrics_and_interleaved_graphs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    architecture: str,
    forced: bool,
    lora: bool,
) -> None:
    torch.set_num_threads(1)
    L.seed_everything(42)
    asset = create_assets(
        tmp_path / "tiny", "seq2seq" if architecture == "seq2seq" else "causal"
    )
    if architecture == "qwen":
        Qwen2ForCausalLM(
            Qwen2Config(
                vocab_size=18,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=128,
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=1,
            )
        ).save_pretrained(asset)
    module = GRPOModule(
        model_name_or_path=str(asset),
        temperature=0.7,
        forced_bos_token_id=13 if forced else None,
        logprob_batch_size=4,
        gradient_checkpointing=lora,
        lora={"enabled": True, "r": 2, "target_modules": ["q_proj", "v_proj"]}
        if lora
        else None,
    )
    module._prepare_gradient_scoring()
    rollout = scoring_example(module)
    full = module._score_batch(
        module.policy, rollout["inputs"], rollout["sequences"], rollout["mask"]
    )
    expected_loss, expected_metrics = grpo_loss(
        full,
        rollout["old_logps"],
        rollout["reference_logps"],
        rollout["advantages"],
        rollout["mask"],
        module.hparams["clip_epsilon"],
        module.hparams["beta"],
    )
    expected_loss.backward()
    expected_grads = {
        name: p.grad.clone()
        for name, p in module.policy.named_parameters()
        if p.grad is not None
    }
    module.policy.zero_grad(set_to_none=True)
    calls: list[str] = []
    original = module._score_batch

    def score(*args: Any) -> torch.Tensor:
        calls.append("forward")
        return original(*args)

    def backward(loss: torch.Tensor) -> None:
        calls.append("backward")
        loss.backward()

    monkeypatch.setattr(module, "_score_batch", score)
    monkeypatch.setattr(module, "manual_backward", backward)
    role_weight = 0.3 if lora else 1.0
    actual_loss, metrics = module._backward_rollout(rollout, loss_weight=role_weight)
    assert calls == ["forward", "backward", "forward", "backward"]
    torch.testing.assert_close(
        actual_loss, expected_loss.detach(), atol=2e-6, rtol=2e-5
    )
    for name, expected in expected_metrics.items():
        torch.testing.assert_close(
            metrics[name], expected.detach(), atol=2e-6, rtol=2e-5
        )
    for name, parameter in module.policy.named_parameters():
        if name in expected_grads:
            torch.testing.assert_close(
                parameter.grad, expected_grads[name] * role_weight, atol=3e-6, rtol=3e-4
            )
    assert metrics["scoring_microbatches"] == 2
    assert metrics["scoring_candidates_per_graph_max"] == 4
    with torch.no_grad():
        actual = module._logps(
            module.policy, rollout["inputs"], rollout["sequences"], rollout["mask"]
        )
    torch.testing.assert_close(
        actual[rollout["mask"]], full.detach()[rollout["mask"]], atol=2e-5, rtol=2e-5
    )
    assert (actual[~rollout["mask"]] == 0).all()
    batches = module._candidate_batches(
        rollout["inputs"], rollout["sequences"], rollout["mask"]
    )
    assert sum(batch.padded_tokens for batch in batches) < 6 * (
        rollout["inputs"]["input_ids"].shape[1] + rollout["mask"].shape[1]
    )
    module.hparams["logprob_token_budget"] = max(
        batch.padded_tokens // len(batch.indices) for batch in batches
    )
    budgeted = module._candidate_batches(
        rollout["inputs"], rollout["sequences"], rollout["mask"]
    )
    assert all(
        batch.padded_tokens <= module.hparams["logprob_token_budget"]
        for batch in budgeted
    )
    module.hparams["logprob_token_budget"] = 1
    with pytest.raises(ValueError, match="exceeds"):
        module._candidate_batches(
            rollout["inputs"], rollout["sequences"], rollout["mask"]
        )


def test_throughput_preset_composes() -> None:
    with initialize(version_base="1.3", config_path="../../configs"):
        cfg = compose(config_name="grpo", overrides=["experiment=phomt_rl_throughput"])
    assert cfg.data.batch_size == 16 and cfg.model.group_size == 4
    assert cfg.data.length_batching.candidate_multiplier == 4
    assert cfg.data.length_batching.reserve_target_tokens == 128
    assert cfg.model.logprob_batch_size == 8
    assert cfg.model.logprob_token_budget == 3072


def test_entire_candidate_batch_can_share_one_graph(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    asset = create_assets(tmp_path / "tiny", "causal")
    module = GRPOModule(model_name_or_path=str(asset), logprob_batch_size=None)
    module.policy.eval()
    rollout = scoring_example(module)
    calls = []

    def backward(loss: torch.Tensor) -> None:
        calls.append(1)
        loss.backward()

    loss, metrics = module._backward_rollout(rollout, backward_fn=backward)
    assert len(calls) == 1 and torch.isfinite(loss)
    assert metrics["scoring_candidates_per_graph_max"] == 6
    module.hparams["trim_logprob_padding"] = False
    module.hparams["sort_candidates_by_length"] = False
    batches = module._candidate_batches(
        rollout["inputs"], rollout["sequences"], rollout["mask"]
    )
    assert torch.equal(batches[0].sequences, rollout["sequences"])
    invalid = {**rollout, "advantages": torch.full((6,), float("nan"))}
    with pytest.raises(FloatingPointError, match="Non-finite"):
        module._backward_rollout(invalid, backward_fn=backward)


def test_lightning_buckets_and_microbatches_step_once_per_update(
    tmp_path: Path,
) -> None:
    torch.set_num_threads(1)
    asset = create_assets(tmp_path / "tiny", "causal")
    module = GRPOModule(
        model_name_or_path=str(asset),
        max_new_tokens=4,
        group_size=2,
        num_iterations=2,
        logprob_batch_size=3,
        logprob_token_budget=256,
        reward=lambda p, r: [float(i % 2) for i in range(len(p))],
    )
    data = TranslationDataModule(
        train_file="data/example_train.jsonl",
        val_file="data/example_val.jsonl",
        batch_size=2,
        bidirectional=True,
        length_batching={
            "enabled": True,
            "tokenizer_name_or_path": str(asset),
            "source_cap": 64,
            "target_cap": 4,
        },
    )
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=2,
        limit_train_batches=1,
        limit_val_batches=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path / "run",
    )
    trainer.fit(module, datamodule=data)
    assert trainer.global_step == 4
    records = [
        json.loads(line)
        for line in (tmp_path / "run/diagnostics/updates.jsonl")
        .read_text()
        .splitlines()
    ]
    assert len(records) == 4
    assert all(
        record["metrics"]["grpo/scoring_microbatches"] == 2 for record in records
    )
    assert all(
        record["metrics"]["grpo/scoring_candidates_per_graph_max"] == 3
        for record in records
    )


@pytest.mark.parametrize(
    "kwargs", [{"logprob_batch_size": False}, {"logprob_token_budget": 0}]
)
def test_scoring_bound_validation(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        GRPOModule(**kwargs)
