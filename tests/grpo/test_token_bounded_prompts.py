"""Whole-pair filtering, native role boundaries, and consistent SFT/RL budgets."""

import csv
import json

import pytest
import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

from scripts.data.filter_token_pairs import prepare
from scripts.data.prepare_full import prepare_full
from scripts.main.common import ROOT, compose_stage, load_config
from scripts.verify.assets import create_assets
from src.data.components.length_batching import translation_lengths
from src.data.components.translation_prompt import (
    PRECISE_TRANSLATION_PROMPT,
    prompt_overhead,
    translation_prompts,
)
from src.models.components.adapters import add_lora
from src.models.grpo_module import GRPOModule
from src.models.sft_module import TranslationSFTModule


@pytest.fixture
def chat_asset(tmp_path):
    torch.set_num_threads(1)
    asset = create_assets(tmp_path / "policy", "causal")
    tokenizer = AutoTokenizer.from_pretrained(asset)
    tokenizer.chat_template = (
        "{% for message in messages %}USER {{ message['content'] }} END "
        "{% endfor %}{% if add_generation_prompt %}ASSISTANT {% endif %}"
    )
    tokenizer.save_pretrained(asset)
    return asset, tokenizer


def write(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["en", "vi"])
        writer.writerows(rows)


def read(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_filter_keeps_both_sides_complete_and_original_partitions(tmp_path, chat_asset):
    _, tokenizer = chat_asset
    train, test = tmp_path / "train.csv", tmp_path / "test.csv"
    rows = [(f"Hello en{i}", f"Xin chào vi{i}") for i in range(30)]
    rows += [("Hello " * 20, "short Vietnamese"), ("short English", "Xin " * 20)]
    write(train, rows)
    write(test, [("Good test", "Cảm test"), ("you " * 20, "test long")])
    original_dir, bounded = tmp_path / "grouped", tmp_path / "bounded"
    original = prepare_full(train, test, original_dir, validation_pairs=3)
    source_bytes = {
        p: p.read_bytes() for p in [train, test, *original_dir.glob("*.csv")]
    }
    report = prepare(
        original_dir, test, bounded, tokenizer, "tiny", "local", content_limit=8
    )
    assert (
        report["counts"]["sft"]
        + report["counts"]["rl"]
        + report["counts"]["validation"]
        == 30
    )
    assert report["counts"]["test"] == 1
    assert (
        sum(
            report["filter_statistics"][s]["rejected_pairs"]
            for s in ("sft", "rl", "validation")
        )
        == 2
    )
    assert report["filter_statistics"]["test"]["coverage"] == 0.5
    assert report["token_budget"]["prompt_limit"] == 8 + prompt_overhead(
        tokenizer, PRECISE_TRANSLATION_PROMPT, True, {"enable_thinking": False}
    )
    for stage in ("sft", "rl", "validation"):
        old = {(r["en"], r["vi"]) for r in read(original_dir / f"{stage}.csv")}
        selected = {(r["en"], r["vi"]) for r in read(bounded / f"{stage}.csv")}
        assert selected <= old
        assert report["filter_statistics"][stage]["maximum_retained"]["en_tokens"] <= 8
        assert report["filter_statistics"][stage]["maximum_retained"]["vi_tokens"] <= 8
    assert all(p.read_bytes() == value for p, value in source_bytes.items())
    assert (
        original["normalized_en_vi_overlap"] == report["normalized_en_vi_overlap"] == 0
    )
    with pytest.raises(FileExistsError):
        prepare(
            original_dir, test, bounded, tokenizer, "tiny", "local", content_limit=8
        )


def module_options(asset, tokenizer):
    return {
        "model_name_or_path": str(asset),
        "prompt_template": PRECISE_TRANSLATION_PROMPT,
        "use_chat_template": True,
        "chat_template_kwargs": {"enable_thinking": False},
        "source_content_limit": 4,
        "strict_prompt_length": True,
        "max_source_length": 4
        + prompt_overhead(tokenizer, PRECISE_TRANSLATION_PROMPT, True),
        "max_new_tokens": 5,
        "beta": 0,
    }


def test_native_prompt_survives_encode_and_overflow_fails_before_generation(chat_asset):
    asset, tokenizer = chat_asset
    module = GRPOModule(**module_options(asset, tokenizer))
    sources = ["Hello you", "Xin chào"]
    src, tgt = ["English", "Vietnamese"], ["Vietnamese", "English"]
    prompts = module._format_prompts(sources, src, tgt)
    assert all(p.endswith("ASSISTANT ") for p in prompts)
    assert "English to Vietnamese precisely" in prompts[0]
    assert "Vietnamese to English precisely" in prompts[1]
    inputs = module._encode(sources, src, tgt)
    expected = tokenizer(prompts, add_special_tokens=False)["input_ids"]
    for ids, mask, full in zip(
        inputs["input_ids"], inputs["attention_mask"], expected, strict=True
    ):
        assert ids[mask.bool()].tolist() == full
    with pytest.raises(ValueError, match="source_content_limit"):
        module._encode(["Hello " * 5])
    # Arbitrary self-play prompts also cannot be silently cut mid-role.
    with pytest.raises(ValueError, match="Complete translation prompt"):
        module._tokenize_prompts(["Hello " * 100])
    estimated = translation_lengths(
        [{"source": "Hello you", "reference": "Xin chào"}],
        tokenizer,
        source_cap=module.hparams.max_source_length,
        target_cap=5,
        prompt_template=PRECISE_TRANSLATION_PROMPT,
        use_chat_template=True,
        chat_template_kwargs={"enable_thinking": False},
    )
    assert estimated == [(len(expected[0]), 3)]


def test_sft_keeps_complete_translation_plus_eos_and_masks_instruction(chat_asset):
    asset, tokenizer = chat_asset
    module = TranslationSFTModule(
        **module_options(asset, tokenizer), max_target_length=5
    )
    batch = {"source": ["Hello you"], "reference": ["Xin chào"]}
    inputs = module.supervised_inputs(batch)
    target = tokenizer("Xin chào", add_special_tokens=False)["input_ids"] + [
        tokenizer.eos_token_id
    ]
    assert inputs["labels"][inputs["labels"] != -100].tolist() == target
    assert int((inputs["labels"] == -100).sum()) > 0
    result = module.policy(**inputs)
    assert torch.isfinite(result.loss)
    with pytest.raises(ValueError, match="Complete SFT target"):
        module.supervised_inputs({"source": ["Hello"], "reference": ["Xin " * 5]})


def test_manifest_drives_total_cap_and_mismatched_instruction_is_rejected(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "token_budget": {
                    "model_name": "Qwen/Qwen3-1.7B",
                    "tokenizer_revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
                    "prompt_template": PRECISE_TRANSLATION_PROMPT,
                    "use_chat_template": True,
                    "chat_template_kwargs": {"enable_thinking": False},
                    "content_limit": 256,
                    "prompt_overhead": 57,
                    "prompt_limit": 313,
                    "target_limit": 257,
                }
            }
        ),
        encoding="utf-8",
    )
    profile = ROOT / "configs/runs/qwen3_1p7b_4090_precise.yaml"
    overrides = [f"token_filter_manifest={manifest.as_posix()}"]
    cfg = load_config(profile, overrides)
    assert cfg["source_cap"] == 313 and cfg["target_cap"] == 257
    for stage in ("sft", "rl"):
        composed = compose_stage(cfg, stage, None)
        assert composed.model.max_source_length == 313
        assert composed.data.length_batching.source_cap == 313
        assert composed.model.use_chat_template
        assert composed.data.length_batching.use_chat_template
        assert composed.model.source_content_limit == 256
        assert composed.model.strict_prompt_length
    with pytest.raises(ValueError, match="differs from model/instruction"):
        load_config(profile, [*overrides, "prompt_template=different"])
    with pytest.raises(ValueError, match="disabled online augmentation"):
        load_config(profile, [*overrides, "augmentation.enabled=true"])
    ablation = load_config(
        ROOT / "configs/runs/qwen3_1p7b_4090_precise_all_linear.yaml", overrides
    )
    assert ablation["lora"]["target_modules"] == "all-linear"
    assert ablation["lora"]["r"] == cfg["lora"]["r"] == 8
    assert ablation["learning_rate"] == cfg["learning_rate"]
    assert ablation["output_dir"] != cfg["output_dir"]


def test_chat_mode_requires_native_template(chat_asset):
    _, tokenizer = chat_asset
    tokenizer.chat_template = None
    with pytest.raises(ValueError, match="chat_template"):
        translation_prompts(
            tokenizer,
            ["Hello"],
            ["English"],
            ["Vietnamese"],
            PRECISE_TRANSLATION_PROMPT,
            True,
        )


def test_all_linear_lora_updates_attention_and_feed_forward_without_backbone_changes():
    torch.set_num_threads(1)
    policy = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=18,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
        )
    )
    policy = add_lora(
        policy,
        {"enabled": True, "r": 8, "alpha": 16, "target_modules": "all-linear"},
        False,
    )
    frozen = {
        name: p.detach().clone()
        for name, p in policy.named_parameters()
        if not p.requires_grad
    }
    names = [name for name, p in policy.named_parameters() if p.requires_grad]
    for module in (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ):
        assert any(f".{module}.lora_B." in name for name in names)
    assert all("lm_head" not in name for name in names)
    optim = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad], lr=1e-3
    )
    ids = torch.tensor([[8, 9, 10, 11]])
    policy(input_ids=ids, labels=ids).loss.backward()
    optim.step()
    assert all(
        torch.equal(frozen[name], p)
        for name, p in policy.named_parameters()
        if name in frozen
    )
    for module in ("q_proj", "gate_proj", "up_proj", "down_proj"):
        assert any(
            p.detach().count_nonzero() > 0
            for name, p in policy.named_parameters()
            if f".{module}.lora_B." in name
        )
