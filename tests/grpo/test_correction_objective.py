"""Real mixed-direction NLLB training and exact microbatch gradient checks."""

from pathlib import Path

import lightning as L
import pytest
import torch
from torch.utils.data import DataLoader

from src.models.components.correction_objective import (
    CorrectionGroup,
    CorrectionOptions,
    correction_loss,
    quality_scores,
)


def tiny_nllb(path: Path) -> Path:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (
        M2M100Config,
        M2M100ForConditionalGeneration,
        NllbTokenizerFast,
    )

    vocab = {
        word: i
        for i, word in enumerate(
            [
                "<s>",
                "<pad>",
                "</s>",
                "<unk>",
                "Hello",
                "world",
                "Xin",
                "chÃ o",
                "eng_Latn",
                "vie_Latn",
            ]
        )
    }
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = NllbTokenizerFast(
        tokenizer_object=backend, src_lang="eng_Latn", tgt_lang="vie_Latn"
    )
    model = M2M100ForConditionalGeneration(
        M2M100Config(
            vocab_size=len(tokenizer),
            d_model=16,
            encoder_layers=1,
            decoder_layers=1,
            encoder_attention_heads=2,
            decoder_attention_heads=2,
            encoder_ffn_dim=32,
            decoder_ffn_dim=32,
            dropout=0,
            attention_dropout=0,
            activation_dropout=0,
            decoder_start_token_id=2,
            eos_token_id=2,
            pad_token_id=1,
        )
    )
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path


def test_hydra_nllb_length_cache_handles_nested_language_codes(tmp_path: Path) -> None:
    import json
    import os
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    assets = tiny_nllb(tmp_path / "native")
    train = tmp_path / "train.jsonl"
    train.write_text(
        json.dumps({"en": "Hello world", "vi": "Xin"}) + "\n", encoding="utf-8"
    )
    cfg = OmegaConf.create(
        {
            "_target_": "src.data.translation_datamodule.TranslationDataModule",
            "train_file": str(train),
            "val_file": str(train),
            "test_file": str(train),
            "bidirectional": True,
            "paired_batches": True,
            "batch_size": 2,
            "length_batching": {
                "enabled": True,
                "tokenizer_name_or_path": str(assets),
                "source_cap": 258,
                "target_cap": 258,
                "language_codes": {"English": "eng_Latn", "Vietnamese": "vie_Latn"},
                "cache_dir": str(tmp_path / "lengths"),
            },
        }
    )
    data = instantiate(cfg)
    data.setup()
    first = list(data.train_dataloader())
    assert os.environ["TOKENIZERS_PARALLELISM"] == "false"
    assert len(first) == 1 and set(first[0]["direction"]) == {"en_vi", "vi_en"}
    assert len(list((tmp_path / "lengths").glob("*.npy"))) == 1
    assert list(data.train_dataloader()) == first


class Judge:
    def __init__(self, reject_reference: bool = False) -> None:
        self.reject_reference = reject_reference

    def evaluate(
        self,
        sources: list[str],
        predictions: list[str],
        source_langs: list[str],
        target_langs: list[str],
    ) -> list[float]:
        # The functional trial deliberately tests the all-failed fallback route.
        if len(predictions) > 2 or self.reject_reference:
            return [0.3] * len(predictions)
        return [1.0] * len(predictions)

    def check_ready(self) -> None:
        pass


def make_module(tmp_path: Path, **overrides):
    from src.models.anchored_grpo_module import AnchoredGRPOModule
    from src.models.components.translation_reward import TranslationReward

    reward = TranslationReward(
        chrf_weight=0,
        comet_weight=1,
        comet_scorer=lambda sources, predictions, references: [0.8] * len(predictions),
        sac={"enabled": True, "consistency_weight": 0, "fusion": {"enabled": False}},
        sac_scorer=Judge(),
    )
    kwargs = {
        "model_name_or_path": str(tiny_nllb(tmp_path / "model")),
        "reward": reward,
        "beta": 0,
        "temperature": 1,
        "language_codes": {"English": "eng_Latn", "Vietnamese": "vie_Latn"},
        "prompt_template": "{source}",
        "group_size": 2,
        "num_iterations": 1,
        "max_new_tokens": 4,
        "logprob_batch_size": 1,
        "learning_rate": 1e-3,
        "diagnostics": {"enabled": False},
    }
    kwargs.update(overrides)
    return AnchoredGRPOModule(**kwargs)


@pytest.fixture
def batch() -> dict[str, list[str]]:
    return {
        "source": ["Hello world", "Xin chÃ o"],
        "reference": ["Xin chÃ o", "Hello world"],
        "source_lang": ["English", "Vietnamese"],
        "target_lang": ["Vietnamese", "English"],
        "direction": ["en_vi", "vi_en"],
    }


def test_negative_pair_has_a_positive_replacement_and_bounded_weight() -> None:
    scores = torch.tensor([-2.0, -2.0], requires_grad=True)
    group = CorrectionGroup(0, (1,), (0.2,), 0, "training_reference")
    loss, _ = correction_loss(scores, [group], CorrectionOptions(), 1)
    loss.backward()
    assert scores.grad[0] < 0 < scores.grad[1]
    # Near-tied judge scores must not be amplified back to unit weight.
    low = torch.tensor([-2.0, -2.0], requires_grad=True)
    correction_loss(
        low, [CorrectionGroup(0, (1,), (0.02,), 0, "candidate")], CorrectionOptions(), 1
    )[0].backward()
    assert torch.isclose(low.grad[1] * 10, scores.grad[1])
    assert quality_scores([0.9, 0.9], [1.0, 0.3]) == pytest.approx([0.9, 0.27])
    with pytest.raises(ValueError):
        CorrectionOptions(anchor_weight=0)
    with pytest.raises(ValueError):
        quality_scores([float("nan")], [0.3])


def test_native_language_prefix_and_sft_target_mask(tmp_path: Path, batch) -> None:
    from src.models.sft_module import TranslationSFTModule

    module = make_module(tmp_path)
    rollout = module._rollout(batch)
    assert rollout["sequences"][:, 1].tolist() == [9, 9, 8, 8]
    assert not rollout["mask"][:, 0].any()
    assert rollout["advantages"].eq(0).all()
    assert rollout["correction_routes"]["reference_anchors"] == 2
    sft = TranslationSFTModule(
        model_name_or_path=module.hparams["model_name_or_path"],
        beta=0,
        prompt_template="{source}",
        language_codes=dict(module.hparams["language_codes"]),
        max_target_length=6,
    )
    inputs = sft.supervised_inputs(batch)
    assert inputs["labels"][:, 0].tolist() == [-100, -100]
    assert inputs["decoder_input_ids"][:, 1].tolist() == [9, 8]
    assert torch.isfinite(sft.policy(**inputs).loss)


def test_microbatched_correction_matches_one_dense_backward(
    tmp_path: Path, batch
) -> None:
    from src.models.components.grpo import grpo_loss

    torch.set_num_threads(1)
    torch.manual_seed(7)
    module = make_module(tmp_path)
    rollout = module._rollout(batch)
    pool = rollout["correction_pool"]
    scores = module._logps(
        module.policy, pool["inputs"], pool["sequences"], pool["mask"]
    )
    means = (scores * pool["mask"]).sum(-1) / pool["mask"].sum(-1)
    expected, _ = correction_loss(
        means, rollout["correction_groups"], module.correction, 2
    )
    # Exercise the GRPO branch in the same combined graph as corrections.
    rollout["advantages"] = torch.tensor([1.0, -1.0, 0.0, 0.0])
    mapping = rollout["correction_mapping"]
    old = torch.nn.functional.pad(
        rollout["old_logps"], (0, scores.shape[1] - rollout["old_logps"].shape[1])
    )
    pg = grpo_loss(
        scores[mapping],
        old,
        torch.zeros_like(old),
        rollout["advantages"],
        pool["mask"][mapping],
        beta=0,
    )[0]
    (expected + pg).backward()
    gradients = {
        name: p.grad.clone()
        for name, p in module.policy.named_parameters()
        if p.grad is not None
    }
    module.policy.zero_grad()
    actual, _ = module._backward_rollout(
        rollout, backward_fn=lambda loss: loss.backward()
    )
    torch.testing.assert_close(actual, (expected + pg).detach())
    for name, param in module.policy.named_parameters():
        if name in gradients:
            torch.testing.assert_close(
                param.grad, gradients[name], atol=2e-5, rtol=2e-4
            )


def test_train_lora_checkpoint_and_resume_two_directions(tmp_path: Path, batch) -> None:
    from src.models.anchored_grpo_module import AnchoredGRPOModule

    torch.set_num_threads(1)
    module = make_module(
        tmp_path,
        num_iterations=2,
        lora={"enabled": True, "r": 2, "alpha": 4, "target_modules": "all-linear"},
    )
    before = {name: p.detach().clone() for name, p in module.policy.named_parameters()}
    trainer = L.Trainer(
        max_steps=2,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        limit_val_batches=0,
        accelerator="cpu",
        devices=1,
        default_root_dir=tmp_path,
    )
    trainer.fit(module, train_dataloaders=DataLoader([batch, batch], batch_size=None))
    assert trainer.global_step == 2
    assert module._rollout_counter == 1
    assert any(
        not torch.equal(before[name], p)
        for name, p in module.policy.named_parameters()
        if "lora_" in name
    )
    assert all(
        torch.equal(before[name], p)
        for name, p in module.policy.named_parameters()
        if "lora_" not in name
    )
    path = tmp_path / "correction.ckpt"
    trainer.save_checkpoint(path)
    loaded = AnchoredGRPOModule.load_from_checkpoint(
        path, reward=module.reward_fn, weights_only=False
    )
    assert loaded.correction.anchor_weight == 1
    for name, param in loaded.policy.named_parameters():
        torch.testing.assert_close(param, dict(module.policy.named_parameters())[name])
    resumed = L.Trainer(
        max_steps=4,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        limit_val_batches=0,
        accelerator="cpu",
        devices=1,
        default_root_dir=tmp_path / "resume",
    )
    resumed.fit(
        loaded,
        train_dataloaders=DataLoader([batch, batch], batch_size=None),
        ckpt_path=path,
    )
    assert resumed.global_step == 4


def test_unverified_reference_cannot_create_a_training_target(
    tmp_path: Path, batch
) -> None:
    from src.models.anchored_grpo_module import NoUsableFeedback

    module = make_module(tmp_path)
    module.reward_fn.sac = Judge(reject_reference=True)
    with pytest.raises(NoUsableFeedback):
        module._rollout(batch)


def test_correction_profile_composes_without_changing_sft_class() -> None:
    from scripts.main.common import compose_stage, load_config

    cfg = load_config(
        Path("configs/runs/nllb_600m_correction.yaml"), ["wandb.enabled=false"]
    )
    assert compose_stage(cfg, "sft", None).model._target_.endswith(
        "TranslationSFTModule"
    )
    rl = compose_stage(cfg, "rl", None)
    assert rl.model._target_.endswith("AnchoredGRPOModule")
    assert rl.model.correction.anchor_sac_min == 1
    assert rl.model.reward.sac.consistency_weight == 0
    assert rl.data.evaluation_batch_size == 16


def test_mixed_group_routes_to_correction_without_opposing_grpo(
    tmp_path: Path, batch, monkeypatch
) -> None:
    from src.models.components.grpo import group_advantages
    from src.models.grpo_module import GRPOModule

    module = make_module(tmp_path)
    rollout = GRPOModule._rollout(module, batch)
    rollout["predictions"] = ["Xin chÃ o", "world", "Hello", "Hello world"]
    rollout["capped"][:] = False
    components = rollout["reward_components"]
    components["sac"] = [1.0, 0.3, 0.7, 0.7]
    components["comet"] = [0.8, 0.7, 0.8, 0.9]
    components["reward"] = quality_scores(components["comet"], components["sac"])
    rollout["rewards"] = torch.tensor(components["reward"])
    rollout["advantages"] = group_advantages(rollout["rewards"], 2)
    monkeypatch.setattr(GRPOModule, "_rollout", lambda self, batch: rollout)
    actual = module._rollout(batch)
    assert actual["advantages"][:2].eq(0).all()
    assert actual["advantages"][2:].ne(0).all()
    assert actual["correction_routes"]["reference_anchors"] == 0
    assert actual["correction_groups"][0].provenance == "candidate"


def test_resource_estimates_keep_memory_and_compute_separate() -> None:
    from scripts.benchmarks.correction_cost import estimate, forward_flops

    config = {
        "d_model": 16,
        "encoder_layers": 1,
        "decoder_layers": 1,
        "encoder_ffn_dim": 32,
        "decoder_ffn_dim": 32,
        "vocab_size": 211,
    }
    a = forward_flops(config, 24, 24)
    b = forward_flops(config, 24, 24, 4)
    assert b["total"] == 4 * a["total"]
    assert forward_flops(config, 258, 258)["total"] > a["total"]
    bench = {"base_allocated_gib": 1.2, "parameters": 600, "trainable_parameters": 4}
    worst = {
        "cases": [
            {
                "status": "ok",
                "batch_size": 4,
                "registered_operator_flops_forward_backward": 1e12,
                "peak_reserved_gib": 4.8,
                "peak_allocated_gib": 4.2,
            }
        ]
    }
    one = estimate(config, bench, worst, [], iterations=1)
    two = estimate(config, bench, worst, [], iterations=2)
    assert one["api"]["training_requests"] == 2 * two["api"]["training_requests"]
    assert two["memory"]["planning_total_gib"] < 24
    quantized = {**bench, "quantization": "nf4"}
    with pytest.raises(ValueError, match="explicit BF16"):
        estimate(config, quantized, worst, [])
    projected = estimate(config, quantized, worst, [], bf16_base_parameters=1000)
    assert projected["projection_from_quantized_probe"]
    assert projected["model_parameters_including_lora"] == 1004
    with pytest.raises(ValueError):
        forward_flops(config, 0, 1)


def test_causal_policy_can_learn_from_the_same_correction_objective(
    tmp_path: Path, batch
) -> None:
    from scripts.verify.assets import create_assets
    from src.models.anchored_grpo_module import AnchoredGRPOModule
    from src.models.components.translation_reward import TranslationReward

    torch.set_num_threads(1)
    reward = TranslationReward(
        chrf_weight=0,
        comet_weight=1,
        comet_scorer=lambda sources, predictions, references: [0.8] * len(predictions),
        sac={"enabled": True, "consistency_weight": 0, "fusion": {"enabled": False}},
        sac_scorer=Judge(),
    )
    model = AnchoredGRPOModule(
        model_name_or_path=str(create_assets(tmp_path / "causal", "causal")),
        reward=reward,
        beta=0,
        temperature=1,
        group_size=2,
        max_new_tokens=8,
        num_iterations=1,
        logprob_batch_size=1,
    )
    rollout = model._rollout(batch)
    loss, _ = model._backward_rollout(
        rollout, backward_fn=lambda value: value.backward()
    )
    assert torch.isfinite(loss)
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.policy.parameters()
    )


def test_sft_microbatches_preserve_global_token_mean_gradient(
    tmp_path: Path, batch
) -> None:
    from src.models.sft_module import TranslationSFTModule

    torch.set_num_threads(1)
    model = TranslationSFTModule(
        model_name_or_path=str(tiny_nllb(tmp_path / "native")),
        beta=0,
        prompt_template="{source}",
        language_codes={"English": "eng_Latn", "Vietnamese": "vie_Latn"},
        max_target_length=10,
        microbatch_size=1,
    )
    varied = {**batch, "reference": ["Xin chào", "Hello world Hello world"]}
    inputs = model.supervised_inputs(varied)
    expected = model.policy(**inputs, use_cache=False).loss
    expected.backward()
    gradients = {
        k: p.grad.clone()
        for k, p in model.policy.named_parameters()
        if p.grad is not None
    }
    model.policy.zero_grad()
    actual = model._backward_supervised(
        inputs, backward_fn=lambda loss: loss.backward()
    )
    torch.testing.assert_close(actual, expected.detach())
    for name, p in model.policy.named_parameters():
        if name in gradients:
            torch.testing.assert_close(p.grad, gradients[name], atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize("size,microbatch", [("1p3b", 8), ("3p3b", 4)])
def test_larger_nllb_profiles_preserve_global_batches(
    size: str, microbatch: int
) -> None:
    from scripts.main.common import compose_stage, load_config

    cfg = load_config(
        Path(f"configs/runs/nllb_{size}_correction.yaml"),
        ["token_filter_manifest=null", "wandb.enabled=false"],
    )
    assert cfg["batch_sizes"] == {"sft": 16, "rl": 16}
    sft, rl = compose_stage(cfg, "sft", None), compose_stage(cfg, "rl", None)
    assert sft.model.microbatch_size == microbatch
    assert sft.model.max_target_length == 258
    assert rl.model.num_iterations == 2


def test_native_manifest_reuse_checks_tokenizer_and_csv_identity(
    tmp_path: Path,
) -> None:
    import hashlib
    import json

    from transformers import AutoTokenizer

    from scripts.data.filter_token_pairs import reuse_manifest
    from scripts.data.prepare_full import file_hash
    from src.data.components.multilingual_tokens import multilingual_ids

    tokenizer = AutoTokenizer.from_pretrained(tiny_nllb(tmp_path / "tokens"))
    codes = {"English": "eng_Latn", "Vietnamese": "vie_Latn"}
    multilingual_ids(
        tokenizer, ["", ""], ["English", "Vietnamese"], ["Vietnamese", "English"], codes
    )
    fingerprint = hashlib.sha256(
        tokenizer.backend_tokenizer.to_str().encode()
    ).hexdigest()
    csv = tmp_path / "pairs.csv"
    csv.write_text("en,vi\nHello,Xin chào\n", encoding="utf-8")
    manifest = {
        "token_budget": {
            "language_codes": codes,
            "prompt_template": "{source}",
            "use_chat_template": False,
            "tokenizer_state_sha256": fingerprint,
        },
        "outputs": {"sft": {"path": str(csv), "sha256": file_hash(csv)}},
    }
    source = tmp_path / "manifest.json"
    source.write_text(json.dumps(manifest), encoding="utf-8")
    reused = reuse_manifest(
        source, tmp_path / "compatible", tokenizer, "new-model", "pinned-revision"
    )
    assert reused["outputs"] == manifest["outputs"]
    assert reused["token_budget"]["model_name"] == "new-model"
    csv.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="CSV hash"):
        reuse_manifest(
            source, tmp_path / "tampered", tokenizer, "new-model", "revision"
        )
