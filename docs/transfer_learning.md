# Transfer optimization for machine translation

`src/models/components/optim/transfer.py` provides **`TransferAdamW`**, a subclass of `torch.optim.AdamW`. The project package is `src/models`, so the optimizer lives there alongside the existing GRPO components.

Use LoRA/QLoRA when the foundation weights must stay unchanged. Their frozen base parameters are excluded from the optimizer, and disabling adapters recovers the original base behavior. The transfer optimizer additionally limits adapter drift. During full tuning it limits movement of the trainable base weights; it cannot keep those weights unchanged while also updating them. Neither parameter proximity nor frozen-base adapters guarantee that the adapted model retains every behavioral capability. Evaluate that separately.

## Method

L2-SP adds a penalty toward the starting weights:

```text
L = L_task + (l2sp_weight / 2) * sum(||theta - theta_initial||_2^2)
```

This follows the pretrained-parameter regularization introduced in [Explicit Inductive Bias for Transfer Learning](https://proceedings.mlr.press/v80/li18a.html). The paper studies convolutional transfer learning; its use here is a configurable MT regularizer, not a claim of demonstrated MT retention. Ordinary AdamW weight decay instead shrinks weights toward zero. The implementation delegates moments and updates to [PyTorch AdamW](https://docs.pytorch.org/docs/2.10/generated/torch.optim.AdamW.html).

After the AdamW update, an optional projection constrains each parameter tensor:

```text
radius = max_relative_drift * max(||theta_initial||_2, min_reference_norm)
delta = theta - theta_initial
theta = theta_initial + delta * min(1, radius / ||delta||_2)
```

The bound is subject to floating-point rounding. The positive norm floor lets zero-initialized LoRA B matrices learn; `max_relative_drift=0` holds protected tensors at their anchors. This is a per-tensor constraint, not a global functional/KL guarantee. Existing GRPO reference-KL regularization remains in the task loss.

## Hydra and Lightning

The `grpo`, `self_play`, and corresponding evaluation configurations select `optimizer=transfer` by default. This uses a normal partial optimizer factory:

```yaml
# configs/optimizer/transfer.yaml, packaged as model.optimizer
_target_: src.models.components.optim.transfer.TransferAdamW
_partial_: true
lr: ${model.learning_rate}
weight_decay: 0.0
l2sp_weight: 1.0e-3
max_relative_drift: 0.1
min_reference_norm: 1.0
```

These are conservative pilot controls, not tuned translation hyperparameters. For Qwen 7B with the existing frozen-base QLoRA preset:

```powershell
python -m src.train --config-name self_play experiment=qwen_7b_qlora model.optimizer.l2sp_weight=0.001 model.optimizer.max_relative_drift=0.1 data.steps_per_epoch=5
```

Use `--config-name grpo` for parallel-data GRPO. API feedback configuration stays the same. Follow the [LoRA guide](lora_qwen7b.md) for local data paths and environment setup.

`model.learning_rate` supplies the configured optimizer LR. Transfer mode's ordinary weight decay defaults to zero; override `model.optimizer.weight_decay` explicitly if desired. The legacy `model.weight_decay` still supplies `optimizer=adamw`. Per-group rates survive manual warmup and checkpoint resume. `model.max_grad_norm` clips task gradients before the optimizer adds the L2-SP contribution; the weight projection follows the AdamW update.

Epoch logs include `train/transfer/distance_l2`, `train/transfer/max_relative_drift`, and `train/transfer/l2sp_penalty`. The displayed GRPO `train/loss` excludes the optimizer-added L2-SP term. Translation validation still reports corpus BLEU/chrF++; drift is a diagnostic, not a translation or retention score.

Select ordinary AdamW for a control experiment or an older AdamW checkpoint:

```powershell
python -m src.train --config-name self_play experiment=qwen_7b_lora optimizer=adamw
```

With `l2sp_weight=0`, `max_relative_drift=null`, and identical AdamW hyperparameters, `TransferAdamW` follows native AdamW exactly and stores no anchors. Direct Python module construction with `optimizer=None` preserves the prior AdamW behavior; Hydra explicitly supplies the new default factory.

## Native PyTorch usage

```python
from src.models.components.optim import TransferAdamW

optimizer = TransferAdamW(
    (parameter for parameter in model.parameters() if parameter.requires_grad),
    lr=1e-5,
    l2sp_weight=1e-3,
    max_relative_drift=0.1,
)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

Normal parameter groups can override learning rate, regularization, and projection. For example, use `l2sp_weight=0.0, max_relative_drift=None` on a newly initialized head while protecting pretrained tensors. Gradual unfreezing uses `add_param_group` after enabling gradients on the new tensors. Pass only currently trainable parameters at construction so those tensors are not already present in another group.

Closures, `maximize`, AMSGrad, foreach updates, `zero_grad`, learning-rate schedulers, and optimizer `state_dict`/`load_state_dict` work through the PyTorch interface. Sparse gradients and fused/capturable/differentiable steps are unsupported. Conventional non-fused AMP is supported; tiny-model NF4 BF16 mixed-precision integration is tested locally.

To use the factory directly in either GRPO Lightning module:

```python
from functools import partial
from src.models.grpo_module import GRPOModule
from src.models.components.optim import TransferAdamW

module = GRPOModule(
    model_name_or_path='path/to/translation-sft',
    optimizer=partial(TransferAdamW, lr=1e-5, l2sp_weight=1e-3),
)
```

Optimizer factories must be importable functions/classes or `functools.partial` objects with plain-data/tensor arguments. Their configuration is stored as data rather than pickled callables, keeping new checkpoints compatible with restricted PyTorch loading. Custom factory modules must remain importable on resume. Lambdas, local functions, and arbitrary object arguments fail early.

## Checkpoints and memory

The optimizer snapshots only protected trainable tensors, in FP32 (FP64 for FP64 parameters). Adam state and the original anchors are restored together. Resume never resets anchors to the already-adapted policy. Keep the same optimizer type, parameter groups, base revision, and LoRA configuration when resuming. A protected optimizer rejects a checkpoint lacking its anchors instead of silently choosing a new starting point. Use `optimizer=adamw` when resuming historical AdamW runs.

The rank-8 Qwen2.5-7B attention preset has 5,046,272 trainable parameters, so FP32 anchors add an estimated **19.25 MiB** of persistent storage, plus approximately another trainable-gradient-sized temporary allocation when L2-SP is active. Full tuning instead adds one FP32 anchor per trainable base weight—about **28.37 GiB extra for Qwen 7B**—and remains unsuitable for a single 4090 in this framework. The prior [4090 capacity results](../archive/reports/vast_lora_results_2026-11-08.md) used ordinary AdamW and are historical evidence, not new measurements of this optimizer. No rental was used for this extension.

## Retention validation

Before a research run, record a baseline on disjoint MT development/test data and approved general-language or multilingual retention tasks. Run a short adapter pilot, inspect translations, compare MT quality and retention against the baseline, and monitor drift. Reduce LR, rank, drift radius, or training duration if behavior deteriorates. Select checkpoints using both MT and retention results; a better proxy reward or chrF alone does not establish preservation of foundation capabilities. Automatic general-capability gating and rehearsal/distillation datasets are not implemented here.
