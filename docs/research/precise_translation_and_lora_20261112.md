# Complete-pair token limits, instruction prompting and LoRA audit

## Prepared experiment

The new profile is `configs/runs/qwen3_1p7b_4090_precise.yaml`. It keeps only original aligned pairs whose **English and Vietnamese texts both fit 256 content tokens**, checks the complete native chat prompt in both directions, and reserves the measured instruction/chat overhead in addition to those 256 tokens. No source text is shortened to make a pair pass.

The instruction is:

```text
Translate {source_lang} to {target_lang} precisely. Preserve the meaning, names, numbers, and negation. Return only the translation.
Source: {source}
```

The tokenizer renders the user turn and assistant generation prefix using `apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)`. Native templates receive `add_special_tokens=False` during subsequent tokenization. No model-specific think block is manually inserted.

`source_cap` in the new profile is populated from the dataset manifest: **256 + maximum empty-source prompt overhead across EN→VI and VI→EN**. The filter also checks actual complete prompts, covering tokenization effects at text boundaries. The target cap becomes **257**, allowing 256 content tokens plus EOS. SFT no longer prepends an extra space to native-chat targets. Strict encoding rejects overlength sources, complete prompts and SFT targets before generation/reward requests; it does not cut role delimiters or silently shorten supervised targets.

The original `data/full_prepared` files already contain the grouped, deduplicated and test-isolated split. The new filter preserves their stage membership; it does not reshuffle SFT examples into RL or validation. The retained SFT fraction can differ slightly from 80% because length filtering acts independently on existing partitions. The manifest records the actual fraction.

The completed local preparation measured **38 overhead tokens**, so this Qwen3 recipe uses **294 total input tokens** and **257 generated/target tokens**. It retained:

| Partition | Original pairs | Retained pairs | Removed |
| --- | ---: | ---: | ---: |
| SFT | 2,373,127 | 2,372,323 | 804 |
| RL | 593,282 | 593,115 | 167 |
| Validation | 2,000 | 2,000 | 0 |
| Bounded test subset | 19,151 | 19,149 | 2 |

The retained training pool is 79.9991% SFT. Output hashes were verified against `data/token256_qwen3/manifest.json`; the streaming pass took 888.7 seconds on the local CPU, with no rental or foundation download. The largest retained complete prompt is exactly 294 tokens, so it fits without truncation.

For this original-pair experiment, online augmentation is disabled: the current concatenation and Unicode perturbations are not token bounded. Enabling them would invalidate the guarantee, so this profile rejects that configuration. Existing run profiles retain their previous behavior. A future augmentation ablation must recheck both sides after augmentation and fall back as a whole pair.

Validation is filtered within the existing validation partition. A separate `test_bounded.csv` is written with coverage reported. **It is a length-selected subset, not the complete official test.** The original official test remains unchanged, and scores on the new subset must not be compared directly with previous full-test scores.

The profiles retain the existing **200-batch validation/test cap**. Evaluating every retained test pair requires overriding `trainer.limit_test_batches=1.0`. Comparisons should use the same selected examples and decoding settings.

## Qwen3 is already instruction capable

`Qwen/Qwen3-1.7B` is the post-trained chat model, distinct from `Qwen3-1.7B-Base`. Its official card documents the non-thinking switch. An empty think block in a rendered native template is therefore not evidence of an unsupported instruction model; manually copying that block to unrelated models would be inappropriate. Use each tokenizer's own template. [Qwen3-1.7B model card](https://huggingface.co/Qwen/Qwen3-1.7B).

| Model | Use in this project | Qualification |
| --- | --- | --- |
| `Qwen/Qwen3-1.7B` | Keep for the first controlled data/prompt correction | Already post-trained; native `enable_thinking=False` is supported. |
| `Qwen/Qwen2.5-1.5B-Instruct` | Small, explicitly instruction-tuned alternative near the current parameter budget | Official card lists Vietnamese; use its native template without adding think tags. |
| `Qwen/Qwen3-4B-Instruct-2507` | Larger instruction-only alternative for a later quality comparison | Officially non-thinking-only; no think block is required. More VRAM and compute need an actual batch benchmark. |

The first two model facts above come from the [Qwen3 card](https://huggingface.co/Qwen/Qwen3-1.7B) and [Qwen2.5-1.5B-Instruct card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct); the third comes from the [Qwen3-4B-Instruct-2507 card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507). These are compatible causal-model candidates, not demonstrated EN–VI quality rankings. Changing the tokenizer requires rebuilding the bounded dataset and its manifest. Do not load adapters from another foundation into a replacement model.

## What the LoRA evidence shows

The active-run snapshot has **3,211,264 trainable parameters (0.186% including adapters)**, spread over Q/K/V/O projections in **all 28 layers**, at rank 8 and alpha 16. No layer-selection freeze is active. The feed-forward gate/up/down projections, embeddings, LM head and backbone remain frozen.

The saved SFT step-2000 and RL step-1092 checkpoints each contain 112 adapted projections / 224 adapter tensors. All **112 effective B×A updates are nonzero** in both checkpoints, and all Adam states report the expected completed update count. During the later 450-update snapshot, gradient norms range from **0.207 to 5.243**, with mean **1.152**; **55.1%** of updates are gradient clipped. The observed LR is 2e-5 throughout. This rules out a completely disconnected or never-updated adapter, not a quality or optimization limitation.

I computed the nonzero singular values of each B×A update using QR factors and an r×r SVD, avoiding dense foundation weights. Effective rank here is exp(entropy(normalized squared singular values)):

| Checkpoint | Median energy-effective rank | Median rank for 99% energy | Rank allowed |
| --- | ---: | ---: | ---: |
| SFT 2000 | 2.347 | 6 | 8 |
| RL 1092 | 1.815 | 6 | 8 |

Some projections use all eight dimensions, but these values **do not establish widespread rank saturation**. Effective rank is descriptive and does not prove that a larger adapter would be useless. The archived RL adapter is trained on the merged SFT base; its matrices should not be subtracted from the separate SFT adapter as if both shared the same foundation.

Attention-only placement may limit adaptation of lexical/semantic transformations in the feed-forward layers. That is a hypothesis requiring a matched ablation. The prepared `qwen3_1p7b_4090_precise_all_linear.yaml` changes only placement to `all-linear`, retaining rank 8, alpha 16, LR, data, instruction and schedule. PEFT's `all-linear` option covers transformer linear projections and excludes the output layer for supported pretrained models. [PEFT LoRA documentation](https://huggingface.co/docs/peft/en/developer_guides/lora).

Training exposure is another confounder: the archived SFT configuration uses 12 directional examples per update and stops at 2,000 updates. With paired EN→VI/VI→EN views, that is at most 12,000 distinct parallel pairs, roughly **0.51%** of the original 2,373,127-pair SFT partition. A training-loss plateau on this sampled stream does not demonstrate that the full corpus was learned, or that LoRA capacity was exhausted. A new model, additional layer freezing or a higher rank should be compared on the same fixed validation examples.

For Qwen3-1.7B, the parameter counts are **3.211M** for attention-only r8, **8.716M** for all-linear r8, and **17.433M** for all-linear r16. The larger adapters add modest parameter/optimizer storage relative to the foundation, but activation and wall-time effects still need GPU measurement. Try placement at fixed rank before changing rank and learning rate together. Freezing additional layers would reduce adaptation capacity further and is not justified by the current evidence.

## Commands

From `GRPO-MT`, using the existing Conda environment on Windows:

```powershell
# Uses the saved tokenizer locally; no foundation weights are downloaded.
..\venv\python.exe -m scripts.data.filter_token_pairs --tokenizer ../runs/qwen3-60296212-20261112/tokenizer

# Resolve measured token limits and inspect both stage configurations.
..\venv\python.exe -m scripts.main.train --config configs/runs/qwen3_1p7b_4090_precise.yaml --dry-run

# Start a NEW controlled experiment when ready; this task does not launch it.
..\venv\python.exe -m scripts.main.train --config configs/runs/qwen3_1p7b_4090_precise.yaml

# Matched placement ablation: new output directory, same data/prompt/rank/LR.
..\venv\python.exe -m scripts.main.train --config configs/runs/qwen3_1p7b_4090_precise_all_linear.yaml
```

On a fresh Linux machine, replace `..\venv\python.exe` with the environment's `python`, and use `--tokenizer Qwen/Qwen3-1.7B` for preparation. The filter pins revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` by default. It downloads tokenizer assets only. Runtime credentials remain environment variables as in the existing manual.

The full-data filter makes a single streaming pass through each existing partition, retains CSV text unchanged, and records file hashes and per-language length histograms. A new output directory is required; source data and old checkpoints are not overwritten. Model/instruction/revision mismatches are rejected when loading the bounded profile.

LoRA audit reproduction: `..\venv\python.exe diagnostics/lora_prompt_audit_20261112/analyze_lora.py`. Detailed numeric evidence is in `diagnostics/lora_prompt_audit_20261112/lora_analysis.json`.

Validation: a 62-test regression run passed, followed by six focused tests covering the final filtering/native-prompt implementation and the added all-linear adapter update check (63 distinct tests). Ruff and mypy passed for the changed Python implementation. Both complete-data profiles were composed for SFT and RL with the measured 294/257 limits and 200-batch evaluation cap. No large-model training, API reward requests or rental deployment were started by this change; the longer-output batch plan still needs a GPU benchmark before a production run.
