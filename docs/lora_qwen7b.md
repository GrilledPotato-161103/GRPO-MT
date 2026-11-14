# LoRA / QLoRA and Qwen 7B on an RTX 4090

This optional/reference guide retains earlier experiments and measured results.
Use [the execution manual](execution.md) and `configs/runs/a100.yaml` for the
current COMET/SAC pipeline. Archived shell launchers describe historical runs.

This optional/reference guide retains earlier experiments and measured results.
Use [the execution manual](execution.md) and `configs/runs/a100.yaml` for the
current COMET/SAC pipeline. Archived shell launchers describe historical runs.

The supplied presets target **`Qwen/Qwen2.5-7B-Instruct`**. Other Qwen 7B releases can have different shapes and memory requirements. Full tuning remains the default unless adapters are explicitly enabled.

## Installation and training

Use the existing conda environment at `E:/IAI/Lab/Research/venv` (`../venv` from this checkout):

```powershell
conda activate E:/IAI/Lab/Research/venv
python -m pip install -e '.[lora]'
# For 4-bit NF4 training, install this extra instead:
python -m pip install -e '.[qlora]'
```

Self-play with a BF16 base and LoRA:

```powershell
python -m src.train --config-name self_play experiment=qwen_7b_lora data.steps_per_epoch=100 data.val_file=data/phomt_prepared/validation.csv data.test_file=data/phomt_test_selected.csv
```

For NF4 QLoRA, replace `experiment=qwen_7b_lora` with `experiment=qwen_7b_qlora`. Feedback remains the configured OpenRouter API by default, using `CLAUDE_AUTH_KEY` and the existing `deepseek_v4.1_flash_1017` model mapping. The feedback model stays outside the 4090 in API mode. A self-hosted judge needs its own memory budget; these presets do not reserve room for one.

For parallel-data GRPO with the same adapter settings:

```powershell
python -m src.train --config-name grpo experiment=qwen_7b_qlora data.train_file=data/phomt_prepared/train.csv data.val_file=data/phomt_prepared/validation.csv data.test_file=data/phomt_test_selected.csv
```

Prepared PhoMT paths require running the existing preparation procedure first. The repository's tiny bundled examples exercise the pipeline; use the held-out development split to select checkpoints and the original test set for final evaluation. No translation-quality improvement is implied by enabling adapters.

Both 7B presets use one GPU, BF16 mixed precision, rank 8 / alpha 16 adapters on `q_proj`, `k_proj`, `v_proj`, and `o_proj`, batch size 1, group size 2, two updates per rollout, source cap 512, completion cap 64, gradient checkpointing, log-probability microbatches of one, and a ten-optimizer-step linear warmup. Self-play has additional proposer/solver task-count limits in `configs/model/self_play.yaml`; inspect those when changing group sizes.

Hydra now supplies the [transfer optimizer](transfer_learning.md) by default to anchor the trainable adapters and bound their drift. Use `optimizer=adamw` to reproduce the historical 4090 optimizer settings. The new optimizer adds estimated anchor/temporary-gradient memory; the historical 7B peaks below have not been remeasured with transfer protection.

Options can also be applied to another model:

```text
model.lora.enabled=true
model.lora.r=8
model.lora.alpha=16
model.lora.target_modules='[q_proj,k_proj,v_proj,o_proj]'
model.lora.warmup_steps=10
model.load_dtype=bfloat16
model.gradient_checkpointing=true
model.logprob_batch_size=1
model.quantization=nf4             optional; CUDA and enabled LoRA required
```

`target_modules=all-linear` is supported, with a larger adapter/optimizer budget. Target names must match the chosen architecture. `model.lora.warmup_steps=0` disables adapter warmup; positive `model.warmup_steps` overrides the adapter setting. Dropout is disabled for GRPO scoring, including LoRA dropout during checkpointed training.

QLoRA uses bitsandbytes NF4, nested quantization, and BF16 compute with `prepare_model_for_kbit_training`. This is 4-bit base-weight storage plus trainable floating-point adapters, following the [PEFT quantization procedure](https://huggingface.co/docs/peft/developer_guides/quantization). It is restricted to a single explicitly selected CUDA device; no automatic CPU/off-device dispatch is used.

## Frozen reference and checkpoints

Adapter mode freezes every base parameter and initializes the adapter output to zero. Reference likelihoods run through that same base with adapters disabled under `no_grad`, preserving the initial KL anchor without a second base copy. Bias training and arbitrary `modules_to_save` are intentionally unsupported because they could modify that anchor. Merge an existing adapter into a base checkpoint before starting a new run, or resume a framework Lightning checkpoint.

For checkpointed scoring, the model enters training mode with dropout disabled. Causal models exposing `logits_to_keep` compute only the required completion-position logits. `logprob_batch_size` limits the number of rows in each scoring forward; graphs for all rows still contribute to the joint loss.

LoRA Lightning checkpoints contain adapter tensors, optimizer state, hyperparameters, and self-play task memory where applicable. The frozen base is reconstructed from the saved model ID and resolved Hub commit. Use the same experiment configuration when resuming:

```powershell
python -m src.train --config-name self_play experiment=qwen_7b_qlora ckpt_path=logs/path/checkpoints/last.ckpt data.steps_per_epoch=100
```

Retain access to the original tokenizer and base assets. Local model directories must remain unchanged because they have no immutable Hub revision. Full-tuning checkpoints continue to store both policy and reference weights.

The Python `load_from_checkpoint` path reads the saved base revision automatically. The Hydra CLI constructs the model before restoring state; if a Hub branch has advanced, pass `model.base_revision=COMMIT_FROM_CHECKPOINT` for resume/evaluation. A mismatched base revision fails explicitly rather than silently changing the KL anchor.

`module.export_pretrained('exports/translator')` exports a PEFT adapter and tokenizer, not a merged standalone model. Load it with the matching base and quantization settings, for example:

```python
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, BitsAndBytesConfig

tokenizer = AutoTokenizer.from_pretrained('exports/translator')
model = AutoPeftModelForCausalLM.from_pretrained(
    'exports/translator',
    torch_dtype=torch.bfloat16,
    device_map={'': 0},
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    ),
)
```

Omit `quantization_config` for a BF16-base adapter export. Export/reload is for inference; Lightning resume also restores optimizer and task-memory state.

## Capacity calculation

The [official Qwen2.5-7B-Instruct configuration](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/raw/main/config.json) specifies 28 layers, hidden size 3,584, intermediate size 18,944, vocabulary 152,064, four KV heads, and untied input/output embeddings. Counting those tensors gives **7,615,616,512 base parameters**. Rank-8 attention adapters add **5,046,272 trainable parameters**.

| Storage component | Calculated memory (GiB) |
|---|---:|
| One BF16 base | 14.185 |
| Two BF16 bases, or one FP32 base | 28.370 |
| Attention adapters + gradients + two Adam moments, all FP32 | 0.075 |
| NF4 linear weights plus FP32 nonquantized weights in k-bit preparation | About 7.100 before quantization metadata |

These are tensor-storage calculations, not total training peaks. The large untied vocabulary embeddings and output head remain unquantized; k-bit preparation promotes nonquantized floating-point weights to FP32 in the tested PEFT implementation. Consequently, dividing the entire parameter count by two bytes is not a realistic NF4 training estimate. Activations, logits, KV cache, quantization metadata, and CUDA workspaces need additional memory.

Full-parameter GRPO with this framework's separate reference cannot fit in 24 GiB: even the two BF16 bases exceed capacity. Shared-base LoRA and NF4 QLoRA avoid that duplication. Capacity still depends on prompt/output lengths, number of solver tasks, group size, target modules, and concurrent processes.

## Measured RTX 4090 result

**Both BF16 LoRA and NF4 QLoRA fit the tested workload.** Instance 59599517 reported 23.648 GiB usable GPU memory. The probe used rank-8 attention adapters, 512 prompt tokens, 64 completion tokens, two proposer candidates, two solver sources with two candidates each, one-row log-probability microbatches, gradient checkpointing, and two optimizer updates. Generation separately exercised four candidates at the full token caps.

| Measurement (GiB) | BF16 LoRA | NF4 QLoRA |
|---|---:|---:|
| Loaded model allocation | 14.246 | 7.227 |
| Generation peak allocation | 14.642 | 8.001 |
| Joint training peak allocation | **15.338** | **15.005** |
| Joint training peak reserved | 15.643 | 15.377 |
| Adapter parameters updated | Yes | Yes |

The peak includes the actual backward pass and Adam optimizer state. There is no separate reference model and no resident feedback model. QLoRA reduced loaded-weight memory substantially, but **did not proportionally reduce training peak** in this implementation. Choose BF16 LoRA for the tested short-context workload or NF4 when reducing resident/generation memory is useful; reprofile before increasing context, role/task counts, or adding a local judge.

The full 63-test suite also passed on this 4090 with 95.14% coverage, including tiny-model NF4 optimizer resume. See [rental evidence, timings, and cleanup](../archive/reports/vast_lora_results_2026-11-08.md). These capacity measurements use synthetic rewards and do not establish translation quality or live-feedback throughput.

## Reproducing the memory probe

```powershell
python -m scripts.benchmarks.memory --mode lora --output logs/qwen7b_lora
python -m scripts.benchmarks.memory --mode qlora --output logs/qwen7b_qlora
```

Each command runs in a fresh process. It exercises a generation batch of four at 512 prompt tokens and 64 forced new tokens, then a real joint proposer/solver backward/update with two proposer candidates and two solver sources with two candidates each. Training uses synthetic teacher-forced completions and fixed nonconstant rewards to stress the full token shapes. It checks that adapters change, records Adam-inclusive peak allocated/reserved memory and time in `memory_profile.json`, and never calls the feedback API. This is a capacity test, not an on-policy learning or translation-quality benchmark.

`archive/legacy/scripts/run_lora_verification.sh` first checks CUDA initialization, then installs the adapter extras, runs the regression suite with coverage, and profiles both modes. Its `COMPLETE` marker means both profiles were attempted; inspect each JSON status and exit file for success.
