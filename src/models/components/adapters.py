"""Optional LoRA/QLoRA construction with a frozen shared-base reference."""

from typing import Any, cast

import torch


def lora_enabled(options: dict[str, Any] | None) -> bool:
    """Return whether adapter training was explicitly enabled."""
    return bool(options and options.get("enabled", False))


def validate_lora(options: dict[str, Any] | None) -> None:
    """Validate the supported adapter options before loading model weights."""
    if not lora_enabled(options):
        return
    assert options is not None
    allowed = {
        "enabled",
        "r",
        "alpha",
        "dropout",
        "target_modules",
        "warmup_steps",
        "layers_to_transform",
        "layers_pattern",
    }
    if set(options) - allowed:
        raise ValueError(f"Unsupported LoRA options: {sorted(set(options) - allowed)}")
    for field, default in (("r", 8), ("alpha", 16)):
        value = options.get(field, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("LoRA rank and alpha must be positive integers")
    if not 0 <= float(options.get("dropout", 0.0)) < 1:
        raise ValueError("LoRA dropout must be in [0,1)")
    if int(options.get("warmup_steps", 10)) < 0:
        raise ValueError("LoRA warmup_steps must be nonnegative")
    targets = options.get("target_modules", "all-linear")
    if not targets or not isinstance(targets, (str, list)):
        raise ValueError(
            "LoRA target_modules must be a module-name list or regex string"
        )
    layers = options.get("layers_to_transform")
    if layers is not None:
        indices = [layers] if isinstance(layers, int) else layers
        if (
            not isinstance(indices, list)
            or not indices
            or any(
                isinstance(index, bool) or not isinstance(index, int) or index < 0
                for index in indices
            )
            or len(set(indices)) != len(indices)
            or isinstance(targets, str)
        ):
            raise ValueError(
                "Layer-selective LoRA requires unique nonnegative layer indices and a target-module list"
            )
    pattern = options.get("layers_pattern")
    if pattern is not None and (
        layers is None or not isinstance(pattern, str) or not pattern
    ):
        raise ValueError(
            "layers_pattern requires layers_to_transform and a nonempty string"
        )


def add_lora(model: Any, options: dict[str, Any], gradient_checkpointing: bool) -> Any:
    """Freeze the base and add zero-output adapters; reference is the disabled adapter.

    Bias and arbitrary modules_to_save are intentionally unsupported because they
    would change the shared reference. A pre-existing adapter must be merged into
    a base checkpoint first; Lightning resume restores our own adapter state.
    """
    from peft import (
        LoraConfig,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )

    validate_lora(options)
    if getattr(model, "peft_config", None):
        raise ValueError(
            "Start from a merged/base model, or resume a Lightning checkpoint"
        )
    if getattr(model, "is_loaded_in_4bit", False):
        model = cast(Any, prepare_model_for_kbit_training)(
            model,
            use_gradient_checkpointing=gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    layer_options = {
        key: options[key]
        for key in ("layers_to_transform", "layers_pattern")
        if options.get(key) is not None
    }
    config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM
        if model.config.is_encoder_decoder
        else TaskType.CAUSAL_LM,
        r=int(options.get("r", 8)),
        lora_alpha=int(options.get("alpha", 16)),
        lora_dropout=float(options.get("dropout", 0.0)),
        target_modules=options.get("target_modules", "all-linear"),
        bias="none",
        init_lora_weights=True,
        **layer_options,
    )
    return get_peft_model(model, config)


def disable_dropout(model: Any) -> None:
    """Keep likelihoods deterministic while enabling checkpointed training mode."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
        for name in ("dropout", "attention_dropout", "activation_dropout"):
            value = getattr(module, name, None)
            if isinstance(value, (int, float)):
                setattr(module, name, 0.0)


def quantized_load_options(
    quantization: str | None, load_dtype: str, enabled: bool
) -> dict[str, Any]:
    """Build explicit, single-GPU NF4 loading options without inference dispatch."""
    if quantization is None:
        return {}
    if quantization != "nf4" or not enabled:
        raise ValueError("NF4 quantization requires enabled LoRA training")
    if not torch.cuda.is_available():
        raise ValueError("NF4 training requires a CUDA GPU")
    from transformers import BitsAndBytesConfig

    config = BitsAndBytesConfig(  # type: ignore[no-untyped-call]
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=getattr(torch, load_dtype),
    )
    return {
        "quantization_config": config,
        "device_map": {"": torch.cuda.current_device()},
    }
