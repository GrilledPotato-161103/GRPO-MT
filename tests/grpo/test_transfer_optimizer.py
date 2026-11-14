"""Transfer optimization contracts, drift bounds, and Lightning/PEFT resume."""

from copy import deepcopy
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import Any

import lightning as L
import pytest
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from transformers import Qwen2Config, Qwen2ForCausalLM

from scripts.verify.assets import create_assets
from src.data.translation_datamodule import TranslationDataModule
from src.models.components.optim import TransferAdamW
from src.models.components.optim.factory import resolve_optimizer_factory
from src.models.grpo_module import GRPOModule


@pytest.mark.parametrize("maximize", [False, True])
@pytest.mark.parametrize("foreach", [False, True])
def test_disabled_protection_matches_native_adamw(
    maximize: bool, foreach: bool
) -> None:
    initial = torch.tensor([1.0, -2.0], dtype=torch.float64)
    native_parameter = torch.nn.Parameter(initial.clone())
    transfer_parameter = torch.nn.Parameter(initial.clone())
    kwargs = {
        "lr": 0.03,
        "weight_decay": 0.2,
        "amsgrad": True,
        "maximize": maximize,
        "foreach": foreach,
    }
    native = torch.optim.AdamW([native_parameter], **kwargs)
    transfer = TransferAdamW(
        [transfer_parameter], l2sp_weight=0.0, max_relative_drift=None, **kwargs
    )
    for index in range(4):
        gradient = torch.tensor([0.4 + index, -0.7], dtype=torch.float64)
        native_parameter.grad = gradient.clone()
        transfer_parameter.grad = gradient.clone()
        native.step()
        transfer.step()
        assert torch.equal(native_parameter, transfer_parameter)
        assert torch.equal(transfer_parameter.grad, gradient)
    assert transfer.state_dict()["transfer_state"]["anchors"] == {}
    assert transfer.drift_metrics() == {
        "distance_l2": 0.0,
        "max_relative_drift": 0.0,
        "l2sp_penalty": 0.0,
    }


@pytest.mark.parametrize("maximize", [False, True])
def test_l2sp_matches_regularized_objective_and_closure(maximize: bool) -> None:
    start = torch.tensor([2.0, -1.0], dtype=torch.float64)
    parameter = torch.nn.Parameter(start.clone())
    expected = torch.nn.Parameter(start.clone())
    transfer = TransferAdamW(
        [parameter], lr=0.1, l2sp_weight=0.4, max_relative_drift=None, maximize=maximize
    )
    native = torch.optim.AdamW([expected], lr=0.1, weight_decay=0.0, maximize=maximize)
    calls = 0

    def closure() -> torch.Tensor:
        nonlocal calls
        calls += 1
        transfer.zero_grad()
        loss = (parameter - 4).square().sum()
        loss.backward()
        return loss

    for _ in range(5):
        native.zero_grad()
        task_loss = (expected - 4).square().sum()
        regularization = 0.2 * (expected - start).square().sum()
        (
            task_loss - regularization if maximize else task_loss + regularization
        ).backward()
        native.step()
        result = transfer.step(closure)
        assert isinstance(result, torch.Tensor)
        torch.testing.assert_close(parameter, expected)
    assert calls == 5
    metrics = transfer.drift_metrics()
    assert metrics["distance_l2"] == pytest.approx(
        float((parameter.detach() - start).norm())
    )
    assert metrics["l2sp_penalty"] == pytest.approx(0.2 * metrics["distance_l2"] ** 2)


def test_projection_bounds_drift_without_freezing_zero_initialized_adapters() -> None:
    zero = torch.nn.Parameter(torch.zeros(4))
    nonzero = torch.nn.Parameter(torch.tensor([2.0, -3.0]))
    frozen = torch.nn.Parameter(torch.ones(3), requires_grad=False)
    unused = torch.nn.Parameter(torch.ones(2))
    optimizer = TransferAdamW(
        [zero, nonzero, frozen, unused], lr=1.0, max_relative_drift=0.03
    )
    for _ in range(8):
        zero.grad = torch.full_like(zero, -10)
        nonzero.grad = torch.full_like(nonzero, 10)
        frozen.grad = torch.ones_like(
            frozen
        )  # Even stale gradients must not thaw a frozen base.
        optimizer.step()
        assert float(zero.norm()) <= 0.030001
        assert optimizer.drift_metrics()["max_relative_drift"] <= 0.030001
        assert torch.equal(frozen, torch.ones(3))
        assert torch.equal(frozen.grad, torch.ones(3))
        assert torch.equal(unused, torch.ones(2))
    assert float(zero.norm()) > 0
    assert len(optimizer.state_dict()["transfer_state"]["anchors"]) == 3
    assert frozen not in optimizer.state


def test_zero_radius_and_per_group_override_with_scheduler() -> None:
    protected = torch.nn.Parameter(torch.ones(2))
    new_head = torch.nn.Parameter(torch.zeros(2))
    optimizer = TransferAdamW([protected], lr=0.1, max_relative_drift=0.0)
    optimizer.add_param_group(
        {
            "params": [new_head],
            "lr": 0.2,
            "l2sp_weight": 0.0,
            "max_relative_drift": None,
        }
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    for _ in range(2):
        protected.grad = torch.ones_like(protected)
        new_head.grad = torch.ones_like(new_head)
        optimizer.step()
        scheduler.step()
    assert torch.equal(protected, torch.ones_like(protected))
    assert torch.all(new_head < 0)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.025)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(0.05)
    assert len(optimizer.state_dict()["transfer_state"]["anchors"]) == 1


def test_optimizer_checkpoint_keeps_original_anchor_and_exact_next_update() -> None:
    parameter = torch.nn.Parameter(torch.tensor([2.0, -3.0], dtype=torch.float64))
    optimizer = TransferAdamW(
        [parameter], lr=0.2, l2sp_weight=0.5, max_relative_drift=0.03
    )
    for _ in range(3):
        parameter.grad = torch.tensor([1.0, -2.0], dtype=torch.float64)
        optimizer.step()
    buffer = BytesIO()
    torch.save(optimizer.state_dict(), buffer)
    buffer.seek(0)
    saved = torch.load(buffer, weights_only=False)
    restored_parameter = torch.nn.Parameter(parameter.detach().clone())
    restored = TransferAdamW([restored_parameter], lr=999.0)
    restored.load_state_dict(saved)
    assert torch.equal(
        restored.state_dict()["transfer_state"]["anchors"][0],
        torch.tensor([2.0, -3.0], dtype=torch.float64),
    )
    for _ in range(3):
        parameter.grad = torch.tensor([-3.0, 1.0], dtype=torch.float64)
        restored_parameter.grad = parameter.grad.clone()
        optimizer.step()
        restored.step()
        torch.testing.assert_close(parameter, restored_parameter, rtol=0, atol=0)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_anchor_precision_survives_checkpoint_restore(dtype: torch.dtype) -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.25], dtype=dtype))
    optimizer = TransferAdamW([parameter])
    saved = deepcopy(optimizer.state_dict())
    optimizer.load_state_dict(saved)
    anchor = optimizer.state_dict()["transfer_state"]["anchors"][0]
    assert anchor.dtype == (torch.float64 if dtype == torch.float64 else torch.float32)


def test_temporarily_disabled_controls_keep_the_original_anchor_on_resume() -> None:
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = TransferAdamW([parameter], lr=0.2)
    optimizer.param_groups[0].update(l2sp_weight=0.0, max_relative_drift=None)
    parameter.grad = torch.ones(2)
    optimizer.step()
    restored_parameter = torch.nn.Parameter(parameter.detach().clone())
    restored = TransferAdamW([restored_parameter])
    restored.load_state_dict(deepcopy(optimizer.state_dict()))
    restored.param_groups[0]["max_relative_drift"] = 0.01
    restored_parameter.grad = torch.ones(2)
    restored.step()
    assert restored.drift_metrics()["max_relative_drift"] <= 0.010001
    assert torch.equal(
        restored.state_dict()["transfer_state"]["anchors"][0], torch.ones(2)
    )


@pytest.mark.parametrize(
    "settings",
    [
        {"l2sp_weight": -1},
        {"l2sp_weight": float("nan")},
        {"max_relative_drift": -0.1},
        {"max_relative_drift": float("inf")},
        {"min_reference_norm": 0},
        {"min_reference_norm": True},
    ],
)
def test_invalid_transfer_settings(settings: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        TransferAdamW([torch.nn.Parameter(torch.ones(1))], **settings)


@pytest.mark.parametrize("flag", ["fused", "capturable", "differentiable"])
def test_unsupported_execution_modes_fail_before_training(flag: str) -> None:
    with pytest.raises(ValueError, match="requires non-fused"):
        TransferAdamW([{"params": [torch.nn.Parameter(torch.ones(1))], flag: True}])


@pytest.mark.parametrize(
    "corruption", ["missing", "version", "shape", "extra", "nonfinite"]
)
def test_invalid_anchor_checkpoint_is_rejected(corruption: str) -> None:
    optimizer = TransferAdamW([torch.nn.Parameter(torch.ones(2))])
    saved = deepcopy(optimizer.state_dict())
    if corruption == "missing":
        del saved["transfer_state"]
    elif corruption == "version":
        saved["transfer_state"]["version"] = 999
    elif corruption == "shape":
        saved["transfer_state"]["anchors"][0] = torch.ones(3)
    elif corruption == "extra":
        saved["transfer_state"]["anchors"][99] = torch.ones(2)
    else:
        saved["transfer_state"]["anchors"][0].fill_(float("nan"))
    with pytest.raises(ValueError):
        optimizer.load_state_dict(saved)


def test_sparse_gradient_failure_restores_task_gradients_without_updating() -> None:
    dense = torch.nn.Parameter(torch.ones(2))
    sparse = torch.nn.Parameter(torch.ones(2))
    optimizer = TransferAdamW([dense, sparse])
    dense.grad = torch.ones(2)
    task_gradient = dense.grad
    sparse.grad = torch.sparse_coo_tensor(torch.tensor([[0]]), torch.ones(1), (2,))
    with pytest.raises(RuntimeError, match="sparse"):
        optimizer.step()
    assert dense.grad is task_gradient
    assert torch.equal(dense, torch.ones(2))


def test_missing_anchor_after_unfreezing_requires_explicit_group() -> None:
    parameter = torch.nn.Parameter(torch.ones(2), requires_grad=False)
    optimizer = TransferAdamW([parameter])
    parameter.requires_grad_(True)
    parameter.grad = torch.ones(2)
    with pytest.raises(RuntimeError, match="new parameter group"):
        optimizer.step()


def test_duplicate_parameters_are_rejected() -> None:
    parameter = torch.nn.Parameter(torch.ones(2))
    with pytest.raises(ValueError, match="Duplicate"):
        TransferAdamW([parameter, parameter])


def test_hydra_optimizer_factory_is_applied_to_both_modules() -> None:
    with initialize(version_base="1.3", config_path="../../configs"):
        for name in ("grpo", "self_play"):
            cfg = compose(config_name=name, overrides=["experiment=qwen_7b_qlora"])
            factory = instantiate(cfg.model.optimizer)
            parameter = torch.nn.Parameter(torch.ones(1))
            optimizer = factory([parameter])
            assert isinstance(optimizer, TransferAdamW)
            assert optimizer.param_groups[0]["lr"] == cfg.model.learning_rate
            assert optimizer.param_groups[0]["weight_decay"] == 0
            legacy = compose(config_name=name, overrides=["optimizer=adamw"])
            assert (
                type(instantiate(legacy.model.optimizer)([parameter]))
                is torch.optim.AdamW
            )


def test_factory_configuration_stays_plain_and_validates_inputs() -> None:
    assert resolve_optimizer_factory(None) == (None, None)
    factory, config = resolve_optimizer_factory(
        partial(TransferAdamW, lr=0.2, betas=(0.8, 0.99))
    )
    assert factory is not None and config is not None
    assert config["_target_"].endswith("TransferAdamW")
    restored_factory, restored_config = resolve_optimizer_factory(config)
    assert restored_config == config and restored_factory is not None
    assert isinstance(
        restored_factory([torch.nn.Parameter(torch.ones(2))]), TransferAdamW
    )
    with pytest.raises(ValueError, match="importable"):
        resolve_optimizer_factory(lambda parameters: torch.optim.AdamW(parameters))
    with pytest.raises(ValueError, match="_partial_"):
        resolve_optimizer_factory({"_target_": "torch.optim.AdamW"})
    with pytest.raises(TypeError, match="plain checkpoint"):
        resolve_optimizer_factory(partial(TransferAdamW, unknown=object()))


def _reward(predictions: list[str], references: list[str]) -> list[float]:
    return [float(index % 2) for index in range(len(predictions))]


def _grouped_optimizer(
    parameters: list[torch.Tensor], lr: float, **kwargs: Any
) -> TransferAdamW:
    return TransferAdamW(
        [
            {"params": parameters[:1], "lr": lr / 2},
            {"params": parameters[1:], "lr": lr},
        ],
        **kwargs
    )


@pytest.mark.parametrize("mode", ["full", "lora", "qlora"])
def test_lightning_transfer_training_validation_and_optimizer_resume(
    tmp_path: Path, mode: str
) -> None:
    if mode != "full":
        pytest.importorskip("peft")
    if mode == "qlora":
        pytest.importorskip("bitsandbytes")
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
    torch.set_num_threads(1)
    L.seed_everything(42)
    asset = create_assets(tmp_path / "model", "causal")
    Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=18,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=256,
            bos_token_id=1,
            eos_token_id=1,
            pad_token_id=0,
        )
    ).save_pretrained(asset)
    module = GRPOModule(
        model_name_or_path=str(asset),
        group_size=2,
        max_new_tokens=4,
        num_iterations=2,
        learning_rate=0.01,
        warmup_steps=2,
        reward=_reward,
        optimizer=partial(
            _grouped_optimizer, lr=0.01, l2sp_weight=0.01, max_relative_drift=0.025
        ),
        lora={"enabled": True, "r": 2, "target_modules": ["q_proj", "v_proj"]}
        if mode != "full"
        else None,
        quantization="nf4" if mode == "qlora" else None,
        load_dtype="bfloat16" if mode == "qlora" else "float32",
        gradient_checkpointing=True,
        logprob_batch_size=1,
    )
    frozen = {
        name: p.detach().cpu().clone()
        for name, p in module.named_parameters()
        if not p.requires_grad
    }
    trainable_before = {
        name: p.detach().cpu().clone()
        for name, p in module.named_parameters()
        if p.requires_grad
    }
    root = Path(__file__).resolve().parents[2]
    data = TranslationDataModule(
        train_file=str(root / "data/example_train.jsonl"),
        val_file=str(root / "data/example_val.jsonl"),
        test_file=str(root / "data/example_test.jsonl"),
    )
    options = {
        "accelerator": "gpu" if mode == "qlora" else "cpu",
        "devices": 1,
        "precision": "bf16-mixed" if mode == "qlora" else "32-true",
        "limit_train_batches": 1,
        "limit_val_batches": 1,
        "num_sanity_val_steps": 0,
        "enable_checkpointing": False,
        "logger": False,
        "enable_model_summary": False,
        "enable_progress_bar": False,
        "default_root_dir": tmp_path,
    }
    trainer = L.Trainer(max_epochs=1, **options)
    trainer.fit(module, datamodule=data)
    optimizer = trainer.optimizers[0]
    assert isinstance(optimizer, TransferAdamW)
    assert trainer.global_step == 2
    assert [group["lr"] for group in optimizer.param_groups] == [0.005, 0.01]
    assert "val/chrf" in trainer.callback_metrics
    assert "train/transfer/max_relative_drift" in trainer.callback_metrics
    assert optimizer.drift_metrics()["max_relative_drift"] <= 0.025001
    assert all(
        torch.equal(p, dict(module.named_parameters())[name].detach().cpu())
        for name, p in frozen.items()
    )
    assert any(
        not torch.equal(p, dict(module.named_parameters())[name].detach().cpu())
        for name, p in trainable_before.items()
    )
    anchors_before = deepcopy(optimizer.state_dict()["transfer_state"]["anchors"])
    checkpoint = tmp_path / "transfer.ckpt"
    trainer.save_checkpoint(checkpoint)
    saved = torch.load(checkpoint, weights_only=True)
    assert isinstance(saved["hyper_parameters"]["optimizer"], dict)
    restored = GRPOModule.load_from_checkpoint(
        checkpoint, reward=_reward, weights_only=False
    )
    resumed = L.Trainer(max_epochs=2, **options)
    resumed.fit(restored, datamodule=data, ckpt_path=str(checkpoint))
    assert resumed.global_step == 4
    resumed_optimizer = resumed.optimizers[0]
    assert isinstance(resumed_optimizer, TransferAdamW)
    assert resumed_optimizer.drift_metrics()["max_relative_drift"] <= 0.025001
    anchors_after = resumed_optimizer.state_dict()["transfer_state"]["anchors"]
    assert anchors_before.keys() == anchors_after.keys()
    assert all(
        torch.equal(anchor.cpu(), anchors_after[key].cpu())
        for key, anchor in anchors_before.items()
    )
