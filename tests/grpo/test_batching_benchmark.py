"""A disposable GPU benchmark must respect its update cap and restore adapters."""

from pathlib import Path

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from scripts.benchmarks.grpo import adapter_digest, run
from scripts.verify.assets import create_assets
from src.models.grpo_module import GRPOModule


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA benchmark")
def test_benchmark_restores_parameters_and_caps_updates(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    asset = create_assets(tmp_path / "tiny", "causal")
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
        load_dtype="bfloat16",
        group_size=4,
        max_source_length=64,
        max_new_tokens=6,
        gradient_checkpointing=True,
        lora={"enabled": True, "r": 2, "target_modules": ["q_proj", "v_proj"]},
        profiling={"enabled": True},
        diagnostics={"enabled": False},
    ).to("cuda")
    digest = adapter_digest(module)
    report = run(module, tmp_path / "report", sources=4, reserve_gib=0)
    assert report["status"] == "completed"
    assert report["updates_executed"] == report["max_updates"] == 10
    assert report["adapter_parameters_restored"]
    assert adapter_digest(module) == digest
    assert not report["training_checkpoint_saved"]
    assert not list((tmp_path / "report").glob("*.ckpt"))
    assert all(trial["status"] == "passed" for trial in report["trials"])
    assert report["trials"][0]["samples"][0]["forward_model_calls"] == 16
    assert report["trials"][1]["samples"][0]["forward_model_calls"] == 2
