# Training procedure and handoff

## Current deployment: 12 November 2026

**Qwen/Qwen3-1.7B is training on RTX4090 instance60341320**, at
`root@142.252.100.103:57220`, using the user's existing `id_ed25519` identity.
Rental cost: **$0.364444/hour** including64GB disk. All uploads and pinned model
weights were verified; CUDA BF16 compute passed. COMET downloaded in107.37s
at20.08MiB/s. Dataset and temporary HF/OpenRouter/W&B keys were explicitly
authorized for this destination; keys were delivered through encrypted SSH stdin
and a RAM FIFO, outside files and logs.

The run resumes the preserved RL1092 checkpoint after reconstructing its SFT2000
merged foundation. The backbone remains frozen. **15 real updates,1093?1107,
were verified at2e-5**, and the native checkpoint at1100 was loaded on CPU:
224 Adam states retain step1100, and ReduceLROnPlateau state is serialized.
The local scheduler/config/resume suite passed18 tests, Ruff and mypy checks.
Next full validation is at cumulative RL update2000, capped at200 batches.

Optimizer: **AdamW**. Scheduler: native **ReduceLROnPlateau** on
`val/comet_macro`, mode max, factor0.5, patience1 (two consecutive completed
non-improving validations), threshold0.001 absolute, minimum1e-6. Baseline best
COMET is0.8216457132 from the prior matched RL1000 evaluation. The target is
10000 cumulative RL updates. SFT is complete and will not be repeated.

[W&B live run](https://wandb.ai/25025069-vietnam-national-university-hanoi/Machine+Translation/runs/qwen3-1p7b-60341320-plateau-rl)
uses the corrected display name `Qwen3-1.7B-LoRA-rl`, project `Machine Translation`.

```powershell
ssh -i C:\Users\Admin\.ssh\id_ed25519 -p 57220 -t root@142.252.100.103 "tmux attach -t mt-train"
```

Detach with Ctrl+B then D. Startup log:`/workspace/train-60341320.log`.
Output:`/workspace/grpo-mt/logs/qwen3-1p7b-60341320/`. The job runs independently;
there is **no ongoing agent monitoring, watchdog, automatic stop or destruction**.
Evidence:`logs/rtx4090_60341320/deployment_verified.json`. Python:
`/venv/main/bin/python`. Resume profile:`configs/runs/qwen3_1p7b_4090_plateau.yaml`.

## Latest closure: 12 November 2026

Old4090 rental **60296212 was destroyed at the user's request**, after local
SHA256 verification of all29 exported files. The saved Qwen3-1.7B checkpoints
are **SFT2000** and **RL1092**, with224 adapter tensors and224 Adam states each.
They are in `E:/IAI/Lab/Research/runs/qwen3-60296212-20261112/`, alongside configs,
validation, tokenizer and metric logs. Native RL requires its SFT-merged base.

The requested AdamW **2e-5 + ReduceLROnPlateau** profile is implemented locally
at `configs/runs/qwen3_1p7b_4090_plateau.yaml` and passed18 targeted tests, Ruff
and type checks. It was **not deployed** before the download/destroy instruction;
the saved RL checkpoint still has its original5e-6 rate. It overrides a restored
rate only when initializing a new schedule, and preserves existing plateau
reductions on subsequent resumes.

The account shows the user's new4090 **60341320 running**. This task has not
uploaded data/keys or changed that instance. The sections below are historical.

## Archived Qwen3 repair: 12 November 2026

RTX 4090 instance **60296212**, Netherlands, costs **$0.471111/hour** including
48 GB disk. The same rental, dataset and pinned Qwen3/COMET caches are in use.
The original run ran out of GPU memory in SFT backward on batch 1,059, after
**1,058 completed optimizer updates**. The step-1,000 validation had BLEU 29.542
and chrF 48.976. A long batch of 24 requested another 5.12 GiB with 2.57 GiB
free; COMET remains resident after validation.

The repaired profile keeps the 256-token source and 128-token target caps,
uses SFT batch **12**, RL batch **16**, scoring microbatches **8** with a
**3,072-token** budget, and `PYTORCH_ALLOC_CONF=expandable_segments:True`.
Manual optimization retains one optimizer update per SFT batch and two per
RL rollout; Lightning gradient accumulation remains at its default of one.

An isolated maximum-length probe passed **five SFT and five RL optimizer
updates**, with COMET loaded on the GPU before backward. Every source was
256 tokens; SFT targets and all 64 RL candidates per rollout were 128 tokens.
Real SAC API scoring produced 208 cache records. Finite BLEU, chrF and real
COMET metrics were checked in both directions, along with the frozen backbone,
checkpoint saving, SFT export loading, and RL adapter/Adam state restoration.
Peak allocated/reserved memory was **13.71/16.37 GiB SFT** and
**8.23/8.56 GiB RL**. These artificial probes verify execution and memory;
their translation scores do not measure production translation quality.
Local batching/resume and profile checks: **25 passed**.

The original checkpoint is preserved locally at
`logs/rtx4090_60296212/interrupted.ckpt` (SHA-256
`41f85e68556cf13a3436d6d61678755baea0951883750f3780d49ec87393119f`). The resume copy changes only loop
accounting; all adapter and Adam tensors were verified identical. Its explicit
batch plan skips the **25,392 already consumed directional examples** and keeps
the remaining bilingual pair order. The production weights are independent of
the probe.

**Production SFT was observed at step 1419** at `2026-11-12T06:34:48.021075+00:00`;
`mt-train` was alive. It resumes toward **2,000 SFT updates**, exports those
weights, then automatically starts **10,000 SAC-GRPO updates**. Production RL
had not started at this observation; RL was validated in the isolated probe.
Rich progress remains enabled. W&B project: `Machine Translation`, group:
`qwen3-1p7b-60296212`. Validation/test are capped at 200 batches.

```powershell
ssh -i C:\Users\Admin\.ssh\id_ed25519 -p 41656 -t root@38.117.87.41 "tmux attach -t mt-train"
```

Detach with `Ctrl+B`, then `D`. Log: `/workspace/repair-60296212.log`.
Outputs: `/workspace/grpo-mt/logs/qwen3-1p7b-60296212/`.
Repair state: `repair_status.json`; stage state: `state.json`;
optimizer progress: `sft/progress.json` or `rl/progress.json` under that output.
Local evidence summary: `logs/rtx4090_60296212/repair_summary.json`.
Credentials remain in process RAM; no credential files were written.
The user controls rental spending. No watchdog or automatic rental stop or
destruction is configured. No agent monitors the run after handoff.

## Previous A100 procedure

Snapshot: 11 November 2026, Asia/Saigon. Configuration files are authoritative;
the rental observations below are historical observations, not a live status feed.


Rental closure: **60228333 was destroyed at the user's request on 12 November
2026**. Official account inventory confirms **no remaining instances**. Checkpoints,
38,302 held-out predictions and evaluation metrics are preserved in
`E:/IAI/Lab/Research/runs/a100-60228333-20261112/`. Rental details below are historical.

## Latest instructions

- Start the newly rented A100 from the Hugging Face foundation, without resuming
  an old model or optimizer checkpoint.
- Run headlessly in tmux and leave the job running independently.
- Do not watch downloads, poll training, or wait for five steps before handing
  back control. Check progress only when the user asks.
- The user controls rental spending. No budget watchdog, automatic stop,
  destruction, replacement rental, or rental overlap is authorized by this note.
- Use the existing user SSH identity. Never generate a replacement key.

The current rental is **60228333**, an A100 PCIe 40 GiB in Japan, host
**458442**, at **$0.620296/hour including 64 GB disk**. Connect directly to
`root@202.122.49.242:15352` with the existing user `id_ed25519` identity.
It uses `vastai/pytorch:cuda-13.2.1-auto`; `/venv/main/bin/python` contains
PyTorch. The measured public Hugging Face throughput was **69.51, 99.77 and
97.63 MiB/s**, across three independent 128 MiB ranges from pinned Qwen weights.
The acceptance threshold was 20 MiB/s in every range test. This is a short
qualification test; sustained training and long-term network rates are not yet
verified.

Previous rental **60201904** was stopped before creating this replacement.
Its disk is preserved; its GPU is no longer running. It had completed no
verified training steps. No model checkpoint transfer is needed.

The user authorized dataset transfer and temporary service keys for this rental.
The 211 MB prepared dataset is uploaded; bootstrap verified the source and data
archive SHA-256 values. Dependency setup is ready. All four pinned Qwen shards
were downloaded and SHA-256 verified in **133.4 seconds**. The preinstalled
PyTorch **2.14.1+cu132** and A100 passed an actual BF16 CUDA computation check.

The fresh pipeline is queued in detached **`mt-train`**. All three temporary
service keys were delivered through encrypted SSH stdin and a RAM FIFO; no key
files were created. The worker completes COMET setup, verifies all four dataset
files, and starts **2,000 SFT ? 10,000 GRPO** updates without old checkpoints.
Optimizer steps have not been observed or awaited. Rich progress is enabled with
`force_terminal: true`. W&B project: **Machine Translation**.

```sh
ssh -i ~/.ssh/id_ed25519 -p 15352 -t root@202.122.49.242 "tmux attach -t mt-train"
```

Detach with `Ctrl+B`, then `D`. Startup log: `/workspace/train-60228333.log`.
Startup state: `/workspace/grpo-mt/logs/sac-a100-60228333/startup.json`.
Training outputs: `/workspace/grpo-mt/logs/sac-a100-60228333/`.
The dataset transfer is complete; the local computer can disconnect. No agent
monitors downloads or training. No automatic rental stop or destruction is set.

## Project and entry points

Local checkout: `E:/IAI/Lab/Research/GRPO-MT`; Conda environment: `../venv`.
Remote working directory: `/workspace/grpo-mt`; Python environment used on the
previous rental: `/venv/main`.

| Path | Purpose |
|---|---|
| `scripts/main/train.py`, `eval.py` | Stable training/evaluation commands |
| `configs/runs/a100.yaml` | Current 40 GiB A100 profile |
| `configs/runs/rtx4090.yaml` | Smaller 24 GiB profile |
| `configs/runs/sac.yaml` | Current objective and SFT/RL schedule |
| `configs/runs/full.yaml` | Shared defaults; SAC overrides its legacy rewards |
| `scripts/rental/setup.sh`, `start.sh` | Environment setup and detached tmux launch |
| `scripts/data/`, `benchmarks/`, `verify/`, `tools/` | Reusable maintenance commands |
| `archive/` | Superseded launchers, reports and the MNIST template |

The cleanup moved 100 files, preserved 62 archived files with original checksums,
and removed 21 disposable caches. Data, checkpoints and experiment logs were
preserved. See [migration](migration.md) and [development](development.md).

## Model and optimization

| Setting | Effective A100 value |
|---|---|
| Foundation | `Qwen/Qwen2.5-7B-Instruct` |
| Foundation revision | `a09a35458c702b33eeacc393d103063234e8bc28` |
| Schedule | 2,000 SFT optimizer updates, then 10,000 GRPO updates |
| SFT/RL learning rates | `1e-4` / `5e-6` |
| SFT/RL warmup updates | 100 / 8 |
| Training | BF16, gradient checkpointing, single GPU |
| LoRA | Rank 8, alpha 16, dropout 0, q/k/v/o attention projections |
| Backbone | Frozen; adapters apply to all matching layers |
| Optimizer | Native AdamW; transfer projection/L2-SP disabled |
| KL | Beta 0; no reference-model forward pass |
| SFT/RL batch sizes | 24 / 16 directional examples |
| GRPO sampling | Four candidates/source; two updates/rollout |
| Log-probability graphs | At most eight candidates and 3,072 padded tokens |
| Source/output token caps | 256 / 128 |
| Data loading | Disk-backed, 12 persistent workers, prefetch factor 2 |
| Batching | Length buckets, shuffled batches, paired directions retained |
| COMET placement/batch | GPU / 4 |
| Evaluation batch | 16 directional examples |

SFT exports its final tuned foundation. RL starts from that export with its own
adapters. A fresh production run never starts from the synthetic probe's tuned
weights. Reusing cached **original** Hugging Face files does not resume training.

## Data and augmentation

| Partition | Parallel pairs | Directional views |
|---|---:|---:|
| SFT | 2,373,127 | 4,746,254 |
| RL | 593,282 | 1,186,564 |
| Validation | 2,000 | 4,000 |
| Official test | 19,151 | 38,302 |

Prepared files are `data/full_prepared/{sft,rl,validation}.csv` and
`data/full/PhoMT_test.csv`, with `data/full_prepared/manifest.json`.
Splitting happens before bidirectional expansion and augmentation. Normalized
English/Vietnamese overlap groups stay together; test-overlapping training
groups are removed. The official test remains unchanged. All four hashes were
verified locally and after upload to instance 60187785.

Training augmentation: whitespace noise 20%, Unicode normalization 20%, aligned
pair concatenation 10%, with at most 768 characters per side. Concatenation is
shared by both directional views; input formatting noise is independent. Clean
originals remain available for scoring. Validation/test are not augmented.
LLM paraphrasing and sequential back-translation augmentation are not enabled.

See [data preparation](data.md). Reuse existing splits with
`python -m scripts.main.prepare --index-only`; do not recreate or overwrite them
merely to launch another rental.

## Reward and feedback

Quality scorer: `Unbabel/wmt22-cometkiwi-da`, revision
`1ad785194e391eebc6c53e2d0776cada8f83179a`. This is reference-free COMETKiwi.
Feedback scorer: OpenRouter `openai/gpt-6-luna`, OpenAI-compatible API at
`https://openrouter.ai/api/v1`.

For an aligned English/Vietnamese pair `(e, v)`, decode both directions in one
batch. Candidate index `k` is matched across the two directions:

```text
v_hat[k] = policy(e, EN -> VI)
e_hat[k] = policy(v, VI -> EN)
S_symmetric[k] = (SAC(e, e_hat[k]) + SAC(v, v_hat[k])) / 2
R_COMET = clip(COMET(source, candidate), 0, 1)
R_SAC = 0.5 * direct_SAC(source, candidate) + 0.5 * S_symmetric[k]
alpha_t = 0.1 + 0.8 * min(completed_RL_updates / 10000, 1)
r_fused = sigmoid((1 - alpha_t) * R_COMET + alpha_t * R_SAC)
```

Both streams reward better quality; no mismatch penalty is subtracted afterward.
Training standardizes each stream within its source's candidate group before
fusion. Set `sac_reward.fusion.normalize_groups=false` for the literal raw mix.
Greedy evaluation uses raw scores, because singleton normalization is constant.
Alpha follows RL optimizer updates, survives resume, and stays fixed while a
rollout is reused. A changing alpha can move fused reward without improving
translation; compare raw validation components too.

SAC tiers: Full match 1.0, High Partial 0.7, Low Partial 0.3, No match 0.1.
API defaults: concurrency 8, max output 32 tokens, timeout 60 seconds, two
retries, disk score cache. A self-hosted Transformers judge is supported by
replacing the backend mapping; see [reward configuration](training.md).
The old DeepSeek conditional harness is disabled in this profile. BLEU/chrF
training weights are zero. This is a COMET/SAC adaptation of M2PO fusion,
not an implementation of M2PO's complete multi-pair CPO algorithm.

RL batch 16 produces 64 candidates and up to 128 judge comparisons per rollout
before caching; the rollout is reused for two updates. Do not count repeated
per-update API counters as new calls. Cache hits, retries and validation affect
actual request counts and API cost.

## Credentials and logging

| Variable | Use |
|---|---|
| `ANTHROPIC_AUTH_TOKEN` | OpenRouter API key, despite the variable name |
| `HUGGING_FACE_API_KEY` | Gated COMET model access; conditions must be accepted |
| `WANDB_API_KEY` | W&B project `Machine Translation` |
| `CLAUDE_AUTH_KEY` | Historical feedback profiles; unused by current SAC |

Keys exist in Windows user environment variables. The operational SSH helper
reads refreshed user-registry values first and strips surrounding whitespace.
Normal launch commands use their process environment; refresh an old shell
after changing keys. Do not put key values in YAML, CLI arguments, logs or Git.

For instance 60187785 the user explicitly approved the 211 MB PhoMT archive and
temporary use of the three current keys over encrypted SSH stdin. Delivery used
a RAM FIFO; no key file was created. This approval record names that instance;
the replacement is now identified as 60201904. Source/data/key transfers to
that replacement have not been attempted as part of this handoff.

W&B entity used previously: `25025069-vietnam-national-university-hanoi`.
Change `wandb.entity` when needed. Rich console progress is retained in tmux.
Metrics include raw/fused rewards, alpha, both directions, policy clipping,
gradient norm, API/cache/token usage, and CUDA-synchronized phase timings.
Backward timing includes gradient-checkpoint recomputation. Sum scoring/backward
micrographs per update; do not sum nested phase totals.

## Fresh, headless launch

Use an existing PyTorch image when possible. The previous rental used a bare
CUDA image and required installing PyTorch/CUDA packages first.

From `/workspace/grpo-mt`, after source/data and environment keys are ready:

```bash
source /venv/main/bin/activate
bash scripts/rental/setup.sh
python -m scripts.main.train --config configs/runs/a100.yaml --dry-run
bash scripts/rental/start.sh --config configs/runs/a100.yaml --stage pipeline \
  --set output_dir=logs/sac-a100-60201904
```

The launcher queues downloads and training in `mt-train`, then returns. The
output-directory override also changes the derived SFT export/cache paths.
Choose a new directory for another fresh run. Do not pass `--resume`.
An existing `mt-train` session or trained output is refused, not overwritten.
An optional five-step synthetic probe is documented in [execution](execution.md);
the latest instruction does not require waiting for it or watching the run.

On-demand commands, only when the user requests a check:

```bash
tmux attach -t mt-train
tail -n 1 logs/sac-a100-60201904/sft/progress.json
nvidia-smi
```

Detach with Ctrl-b, d. Progress JSON is periodic, so phase timing optimizer
events can show a newer update count. Logs are under `logs/sessions/` and the
run's `{sft,rl}/` directories. The agent is not a persistent monitoring service.

## Validation, export and controlled stop

Evaluate every 1,000 optimizer updates and at stage completion. Current code
can repeat final validation when a stage ends exactly on an evaluation boundary.
Validation covers both directions with corpus BLEU/chrF++ (0-100), raw COMET,
SAC, and fused reward. Test evaluation is explicit, not automatic.

```bash
python -m scripts.main.eval --config configs/runs/a100.yaml --stage rl \
  --checkpoint logs/sac-a100-60201904/rl/checkpoints/handoff.ckpt \
  --split test --trust-checkpoint --save-predictions --output logs/final-test
```

Keep the SFT checkpoint and merged SFT export with RL adapters. An RL adapter
is not a standalone adapter against the untouched original foundation.
For a graceful manual stop, create `STOP` inside the run directory; inspect
`checkpoints/interrupted.ckpt` before resuming and remove `STOP`. Resume one
stage explicitly with `--resume ... --trust-checkpoint`. A stopped job does not
stop rental billing. No automatic instance deletion or billing supervisor runs.

## Last verified runtime and remaining work

Previous target: instance **60187785**, host **513108**, California, A100 PCIe
40 GiB, 64 GB disk, recorded total rate **$0.423704/hour**. Direct SSH was
`root@38.65.91.164:20054`; proxy `root@ssh9.vast.ai:38770`. Existing user key:
`C:/Users/Admin/.ssh/id_ed25519`, registered ID 1657702. These are historical
connection details, not the replacement rental's endpoint.

Confirmed on that target:

- Source and full dataset uploaded; all four dataset hashes matched.
- Torch `2.15.0+cu132`, CUDA 13.2 available; Transformers 4.57.6, PEFT 0.18.1,
  Lightning 2.6.6, W&B 0.24.0, COMET 2.2.7 installed.
- Live SAC sanity check returned `[1.0, 0.1]` for match/contradiction.
- COMET checkpoint downloaded and SHA-256 verified: 2,260,734,705 bytes.
- Xet transfers stalled. The worker switched to 16 parallel HTTPS byte ranges
  with per-file pinned SHA-256 verification. The last sub-agent observation
  was Qwen shard 1 at **3.05/3.95 GB**, approximately **9 MiB/s**.
- No optimizer update or production completion was verified at that observation.

The previous `/workspace/launch.py` worker is independent of the agent. Its
queued sequence is downloads -> five SFT/five RL synthetic updates -> fresh
production launch. That describes the previous deployment, not the replacement.
Instance 60187785 was absent from the latest account listing; do not assume its
worker or files exist on 60201904. Do not copy its old endpoint or launch state
onto the replacement.

The delegated agent exited because **workspace credits were exhausted**. That
is separate from Vast.ai balance and does not stop the remote worker or billing.
No continuing sub-agent monitoring is active.

Operational records are in `logs/a100_execution_20261111/`; `state.json` and
`logs/training_active.json` are local records, not live truth. Their older phase
fields can lag the latest observation. `access.py`, `launch.py`,
`download-models.py`, and `observe.sh` describe the previous deployment.

Most recent offline validation: **249 tests passed, 90.04% coverage**, with
Ruff, mypy and shell syntax checks. The later download-acceleration default in
`scripts/main/common.py` received lint/dry-run checks, not another full suite.
The legacy 4090 round-trip pipeline passed a real-model 5+5 pilot; the new
symmetric-fusion GPU probe and full production start remain unverified.

Next execution action: prepare source, data and temporary-key transport on
60201904, then queue a new foundation run under `logs/sac-a100-60201904`.
Return the tmux command after launch acknowledgement; do not poll downloads,
wait for optimizer steps, or stop another rental automatically. After launch,
the replacement's known proxy command will be:

```bash
ssh -i ~/.ssh/id_ed25519 -p 27888 -t root@ssh1.vast.ai "tmux attach -t mt-train"
```

That command is a prepared handoff command, not evidence that the session exists.

Previous 4090 timing estimates are not a benchmark of this A100 configuration:
the batch sizes, sequence lengths, cache hits and API latency differ. A reliable
training ETA is still unmeasured. Recorded rental rates exclude separate API
charges; spending remains under user control.
