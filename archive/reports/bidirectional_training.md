# Bilingual SFT → GRPO with COMETKiwi and BLEU

The new paired-data workflow trains one model in both **EN→VI and VI→EN** directions. It first performs supervised fine-tuning on 80% of the prepared training pool, exports that policy as the RL starting point, and applies GRPO on the remaining 20%.

## Data partitions

The existing 250-pair development split is retained. The 4,750 remaining training pairs split deterministically into **3,800 SFT pairs and 950 RL pairs**. The original 3,500-pair test CSV is unchanged. Normalized English and Vietnamese text is checked for overlap across SFT/RL/development/test; duplicates or overlap fail instead of silently moving or discarding examples.

```powershell
# Only needed on a fresh checkout; existing prepared files are not overwritten.
python -m scripts.prepare_phomt
python -m scripts.prepare_training_stages
```

The second command writes `data/phomt_stages/sft.csv`, `rl.csv`, and a manifest containing counts, seed, and input/output hashes. The split occurs **before** direction expansion or augmentation. Training therefore sees 7,600 directional SFT examples and 1,900 directional RL examples per complete epoch. Development contains 500 directional examples; final testing contains **7,000 translations: 3,500 each way**.

## Environment and gated model access

Use the existing conda environment:

```powershell
conda activate E:/IAI/Lab/Research/venv
python -m pip install -e '.[qlora,comet]'
```

The requested evaluator is [Unbabel/wmt22-cometkiwi-da](https://huggingface.co/Unbabel/wmt22-cometkiwi-da), a reference-free quality estimator using source and candidate translation. Accept its repository conditions and supply access through **`HUGGING_FACE_API_KEY`**. Values are read at download time, never stored in Hydra configs or checkpoints. On Windows, the code also reads that named user-level environment variable when a long-running shell has not inherited it. A local COMET checkpoint can be supplied directly for offline/credential-free workers.

The official `unbabel-comet==2.2.7` dependency family uses NumPy 1.x, TorchMetrics 0.10.x, protobuf 4.x, and legacy `pkg_resources`; the extra pins compatible setuptools. The validated existing environment uses NumPy 1.26.4, TorchMetrics 0.10.3, pandas 3.0.6, protobuf 4.25.9, and setuptools below 81. Its optional logging packages were aligned to W&B 0.24.0, Google common protos 1.69.0, and OpenTelemetry 1.27.0; the unused newer telemetry transport packages were removed. `pip check` passes. These logging packages are not required for the default CSV logger.

## Training and stage handoff

The supplied presets use Qwen2.5-7B-Instruct, rank-8 attention LoRA, BF16 mixed precision, gradient checkpointing, and TransferAdamW. They are conservative pilot configurations. Inspect output truncation and tune source/target caps for the real corpus.

```powershell
python -m src.train --config-name sft experiment=phomt_sft
python -m scripts.export_sft --checkpoint logs/SFT_RUN/checkpoints/last.ckpt --output exports/phomt_sft
python -m src.train --config-name grpo experiment=phomt_rl
```

SFT uses teacher-forced cross-entropy. Causal prompt and padding tokens are excluded from the loss, while target EOS remains supervised even when EOS and PAD share an ID. Encoder–decoder models use target labels. SFT allocates no separate reference model.

`export_sft` loads a trusted checkpoint on CPU and merges adapters into an unquantized dense policy, which becomes the RL reference base. Existing export directories are refused. For QLoRA SFT, reload adapters onto the unquantized base before merging; reevaluate the exported policy because quantized-to-dense merging can change outputs. Add `model.quantization=nf4` to a stage command to use QLoRA. RL creates fresh adapters on the exported SFT policy, keeping that base frozen.

The RL reward defaults to:

```text
reward = 0.7 × clip(COMETKiwi(source, candidate), 0, 1)
       + 0.3 × sentence_BLEU(candidate, reference) / 100
```

Empty candidates receive zero. Raw COMETKiwi scores are retained for evaluation; clipping applies only to the reward. Sentence BLEU uses SacreBLEU's effective-order smoothing. Weights are configurable through `model.reward.comet_weight`, `bleu_weight`, and `chrf_weight`. COMETKiwi stays frozen, runs without gradients or caller AMP, and defaults to CPU; set `model.reward.comet_device=cuda:0` only after budgeting its GPU memory.

COMET scoring uses clean aligned sources even when training inputs receive formatting noise. The existing API-judged self-play mode remains available for generated tasks; its training data has no gold references for BLEU. The paired-data RL preset uses the 20% partition and its gold references.

## Online augmentation

`data.bidirectional=true` creates both language directions from each pair. Training-only augmentation can vary whitespace, use equivalent NFC/NFD Unicode forms, and concatenate another aligned pair from the **same stage partition**. Concatenation keeps the same sentence order on both sides and obeys a character limit. Word deletion, accent removal, number changes, and unverified paraphrases are excluded.

Control probabilities with `data.augmentation.whitespace_probability`, `unicode_probability`, `concat_probability`, and `max_concat_chars`. Set `data.augmentation.enabled=false` for an ablation. Development and test views are always clean and deterministic. Augmentation is seeded for a fixed worker configuration; exact worker augmentation streams are not restored by checkpoints.

## Evaluation in both directions

```powershell
python -m src.eval --config-name grpo_eval experiment=phomt_rl ckpt_path=logs/RL_RUN/checkpoints/last.ckpt
```

Use the training model, tokenizer, adapter, and decode settings. The loader reverses each test pair and changes the prompt's language direction. Models requiring a fixed source-language tokenizer setting or fixed forced target-language BOS reject a conflicting reverse direction; use a prompt-conditioned model for the supplied bilingual presets.

Metrics include `test/en_vi/bleu`, `test/en_vi/chrf`, `test/en_vi/comet`, and corresponding `test/vi_en/*` values, plus equal-direction macro averages. BLEU/chrF use corpus scores on the 0–100 scale; COMET is the mean raw segment score. JSON reports are written under `translation_metrics/test.json`. Checkpoint selection uses `val/chrf_macro`.

## Computation and propagation timings

The new presets enable `model.profiling.enabled=true`. `phase_timing/events.jsonl` records generation, stored-old-policy likelihoods, frozen-reference likelihoods, current-policy forward passes, backward propagation, optimizer steps, and whole-batch latency. SFT and shared proposer/solver roles have separate phase names; joint self-play backward is recorded once because both roles share a combined loss. Frozen COMETKiwi has forward/tokenization/load times and no backward pass.

`phase_timing/summary.json` includes call counts, mean/median/p95 latency, input/output token throughput, and samples per second. CUDA is synchronized around measured phases. Backward includes gradient-checkpoint recomputation. Augmentation and reward components are recorded separately. Parent batch totals overlap child phases and must not be summed with them. Profiling adds synchronization and logging overhead, so disable it for an uninstrumented throughput comparison.

The rental runner tests full tuning, LoRA, and QLoRA with a small Qwen2.5-0.5B policy and synthetic data. Its shared-policy role pilot uses explicitly mocked judge responses; live OpenRouter/server compute is not measured. Real COMETKiwi inference is independently verified. `scripts.summarize_stage_profiles` flattens every mode/stage/phase into `policy_speed.csv`. These short-shape measurements establish infrastructure and pilot throughput, not translation-quality improvement or Qwen 7B runtime.

The [completed 4090 report](vast_bilingual_results_2026-11-08.md) records 120 passing tests, 95.21% coverage, all three tuning modes, real GPU COMETKiwi scoring, 138 timing rows, time/cost scenarios, and verified rental cleanup. CSV rows label policy, COMETKiwi, lexical rewards, augmentation, and synthetic mock feedback separately. Per-role forward passes are measured independently; shared-policy backward is joint.
