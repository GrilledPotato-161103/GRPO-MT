# SAC variability investigation — Qwen3-1.7B, 12 November 2026

The current logs show two different effects: effectively zero numerical noise in the mean normalized SAC metric, and substantial raw-score variation across small, changing batches. The largest raw-score collapse coincides with a truncation failure. These logs do not establish that the judge is inconsistent on identical inputs.

## Evidence and scope

Read-only snapshot from instance **60341320**, optimizer updates **1093–1542**. The 450 update rows contain **225 independently scored rollouts**: each score is reused for two updates. Each rollout contains eight parallel pairs, sixteen directional sources and four candidates per source, for 14,400 candidate translations. Opposite directions and sibling candidates are correlated; they are not 64 independent test examples.

The copied archive SHA-256 is `a6c93b9c7b69147e6f201a2d4aa7b4c4b457191c6276f091713f902dbd8c2ab3`. Analysis verifies the archive, rollout/update alignment, repeated reward equality and group centering. No paid judge calls, training changes, restart or instance lifecycle actions were performed.

## Findings

| Measurement | Observed value |
| --- | ---: |
| Mean direct/raw SAC | 0.74894 |
| Batch standard deviation | 0.08106 |
| Batch 10th–90th percentile | 0.65844–0.84687 |
| Mean absolute change between successive rollouts | 0.08265 |
| Range of 25-rollout moving mean | 0.71381–0.79444 |
| Mean direct SAC, EN→VI / VI→EN | 0.68206 / 0.81583 |
| Mean reference-endpoint SAC | 0.54172 |
| Mean blended SAC reward | 0.64533 |
| Maximum absolute directional normalized mean | 2.98 × 10⁻⁸ |

### 1. `fused_sac_score` is not an absolute quality score

`DynamicRewardFusion` standardizes the blended SAC reward within each source's four candidates. Consequently, the group mean is zero. Each direction contains complete groups, so its mean is also zero, apart from floating-point reduction error. All directional means in this snapshot remain within ±2.98 × 10⁻⁸. A chart with an automatic vertical scale can make this numerical noise look dramatic.

Use `reward/sac_mean`, `direction/en_vi/sac`, `direction/vi_en/sac` and `reward/sac_reward_mean` for absolute batch scores. The normalized metric is useful for candidate discrimination through its group standard deviation, not its mean. The final sigmoid reward also has a structurally constrained mean near 0.5 and is unsuitable as the sole learning-progress metric.

An additional diagnostics caveat exists: empty-output masking is reapplied to normalized components in `reward_diagnostics`, which could disturb zero-centering. **No candidate was empty here**, so that caveat does not explain this snapshot.

### 2. Small batches and changing difficulty explain substantial visible variation

Length bucketing deliberately groups similar-length examples and shuffles completed batches. Candidate sampling uses temperature 1.0, with four independently sampled translations per directional source. Batch scores therefore change with source difficulty and sampled outputs, even without a meaningful change in model quality.

Across all candidate scores, **69.1% of direct-SAC variance lies between source-group means**, with the remaining 30.9% within groups. This is a descriptive variance decomposition, not an estimate of judge noise. The bottom raw-SAC decile has mean prompt length **88.72 tokens** and COMET **0.7715**; the top decile has **40.32 tokens** and COMET **0.8276**. After excluding every batch with prompt truncation, raw SAC still correlates with prompt length at **r = −0.400**, and with COMET at **r = 0.530**. These associations support changing difficulty as an explanation but do not prove causality.

The first versus last 50 rollout means are **0.75384 vs 0.74088**. They contain different examples, augmentation and sampled outputs; that difference cannot establish model regression. No completed new fixed-set validation was present in the snapshot.

### 3. Step 1425 exposes a real prompt-truncation defect

At the first update consuming rollout 713 (optimizer step **1425**):

- Average uncapped prompt length: **531.19 tokens**, against a **256-token whole-prompt cap**.
- **100% of prompts truncated** and **100% of the 64 outputs hit the 128-token generation cap**.
- Direct SAC: **0.146875**; endpoint SAC: **0.171875**; COMET: **0.458689**.

The code truncates the entire formatted chat prompt. The saved Qwen3 tokenizer defaults to **right truncation**; `GRPOModule` overrides padding side but not truncation side. An offline synthetic long-prompt reproduction confirms that right truncation can remove both the user closing delimiter and the assistant prefix. The policy then receives an incomplete user turn, while SAC compares the result with the full, clean source/reference. This makes the low score understandable without assuming an unstable judge. The actual token IDs for that rollout were not logged, so delimiter loss for those exact examples is inferred from the code and tokenizer reproduction.

This extreme batch explains the largest isolated collapse, not every fluctuation. One additional batch had a 6.25% prompt-truncation fraction. Concatenation is limited in characters rather than post-template tokens, and Unicode normalization can change token length; the snapshot lacks per-example augmentation records needed to attribute the offending batch to an operation.

### 4. Four-tier scoring and fusion amplify changes in candidate ranking

The judge emits only **0.1, 0.3, 0.7 or 1.0**. If all four candidates for one of eight directional sources change from 1.0 to 0.1, that direction's batch mean changes by **0.1125**. Even a 1.0→0.7 group shift changes the directional mean by 0.0375.

**42.6%** of direct-SAC groups and **48.6%** of endpoint-SAC groups contain identical tiers. Fusion blends direct SAC with a symmetric average of the two endpoint scores. Thus a candidate's blended reward also depends on its opposite-direction partner. This is the configured symmetric objective, not a request-ordering failure.

The mean direct and endpoint scores differ by **0.2072**, because one compares source meaning across languages and the other compares prediction with the supplied reference in the target language. Their candidate rankings disagree on **12.5% of comparable pairs**, weighted by the number of comparable pairs. COMET and direct SAC disagree on **31.0%**. These metrics evaluate different criteria; disagreement is not by itself evidence of corrupted feedback.

After blending, **25.25% of all source groups** have a nonzero SAC range no larger than 0.1. Group standardization gives nearly unit standard deviation to any nonconstant group, whether its raw spread is small or large. This can amplify small tier changes in the optimization signal. Curriculum alpha also changes with optimizer updates, so the final fused reward is not a stationary absolute metric.

### 5. API/parser failures are too rare to explain the general pattern

The 225 scored rollouts recorded **24,434 API requests**, **4,368 cache/deduplication hits**, **one parse failure successfully retried**, **one transport retry**, **zero exhausted parses** and **zero cache errors**. Invalid replies are retried rather than converted to a fabricated low score. The one parse retry occurred on the long-prompt outlier; that coincidence alone does not explain its 64 low-scored candidates.

Exact source/candidate/language/model/rubric inputs reuse the cached label. We did not bypass that cache to ask the judge repeatedly about identical inputs. The API temperature is omitted, and the logs lack provider identity and repeated blind judgments, so intrinsic judge variance remains **unmeasured**. OpenRouter documents that omitted sampling parameters use upstream defaults; omission alone provides no determinism guarantee. [OpenRouter parameter documentation](https://openrouter.ai/docs/api_reference/parameters).

## Recommended changes, in order

1. **Preserve complete chat framing and aligned content.** Budget source tokens before formatting the prompt; reject or split oversized aligned pairs rather than silently cutting an entire role prompt. Apply token-aware bounds after augmentation, accounting for both directions and target length. Keep truncation/cap rates visible.
2. **Separate training feedback from progress measurement.** Plot raw directional SAC and the blended SAC reward with a 25–50-rollout moving mean. Evaluate a fixed, length-stratified clean probe set using greedy decoding, unchanged scorer settings and separate EN→VI/VI→EN results. Do not judge convergence from mean normalized SAC or mean fused reward.
3. **Measure judge repeatability before changing the judge.** In a small, separately budgeted check, score identical source/candidate pairs several times with cache bypass; record provider/model identity and tier agreement. Include manually reviewed anchor examples to distinguish inconsistency from consistent rubric bias.
4. **Ablate normalization confidence.** Compare the current normalization against a dispersion floor or downweighting of low-margin SAC groups, retaining COMET discrimination. Report fixed-set quality and ranking agreement before adopting the change. Keep direct and endpoint scores separate in the dashboard.

Training was left untouched. These are investigation results and proposed changes, not deployed fixes.

## Reproduction and artifacts

From the project root, using the existing Conda environment:

```powershell
..\venv\python.exe diagnostics/sac_variance_20261112/analyze.py
..\venv\python.exe diagnostics/sac_variance_20261112/probe_prompt_truncation.py
```

Inputs: `logs/sac_variance_20261112/snapshot/rl/diagnostics/{updates,rollouts}.jsonl`. Outputs: `diagnostics/sac_variance_20261112/{analysis.json,rollout_summary.csv,sac_variance.png,prompt_truncation_probe.json}`. The probe uses the saved tokenizer under `../runs/qwen3-60296212-20261112/tokenizer`; it downloads nothing.

Source locations: `src/models/components/reward_fusion.py`, `src/models/components/reward_diagnostics.py`, `src/models/components/translation_reward.py`, `src/models/grpo_module.py`, `src/feedback/sac.py`, `src/data/components/length_batching.py` and `src/data/components/augmentation.py`.
