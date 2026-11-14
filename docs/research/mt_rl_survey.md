# Policies, rewards, and penalties for translation RL

Follow-up, 9 November 2026: [conditional reward policies and reinforcement strategies](mt_policy_strategy_survey_20261109.md) connects the literature to the completed A100 run, separates observed conflicts from hypotheses, and proposes a calibrated reward harness and staged experiments. It supersedes the starting-policy recommendation below for the current COMETKiwi/BLEU LoRA experiment.

Implementation update: a separate [self-play GRPO path](../self_play.md) now supports a source-aware fixed LLM judge through OpenRouter or locally hosted Transformers. The implementation-boundary notes below describe the original scalar-reward GRPO path; the new self-play path adds proposer learnability rewards, validated task memory, and external feedback.

Survey date: 7 November 2026. This is a targeted research survey, not an exhaustive systematic review. Recommendations below are proposed experiments for English–Vietnamese PhoMT; they are not measured quality gains from this repository.

## Recommended starting point

Start from a supervised translation checkpoint, freeze a copy as the KL reference, and optimize sentence chrF++ with original GRPO. Keep the PhoMT test CSV untouched until final evaluation. Compare against the initial policy using corpus BLEU, chrF++, a held-out semantic evaluator, and bilingual human review. Add learned rewards and behavioral penalties only after the baseline is stable.

The default `google/mt5-small` is a compatible loading example, not a translation-finetuned policy. Its model card explicitly requires downstream fine-tuning. Replace it with your English–Vietnamese SFT checkpoint for meaningful training. [mT5 model card](https://huggingface.co/google/mt5-small)

## Policy choices

| Choice | Motivation and tradeoff | Repository support |
|---|---|---|
| Translation-trained encoder–decoder | Direct source conditioning and no instruction-format ambiguity. Language codes, decoder start, and target language tokens must match the checkpoint. | AutoConfig detection + AutoModelForSeq2SeqLM; optional tokenizer source language and forced target BOS |
| Instruction-tuned causal model | Flexible prompts and broader context; needs explicit translation-only output and enough context budget for source plus completion. | AutoModelForCausalLM; configurable text template; left padding and position-aware scoring |
| Original GRPO | Several candidates per source provide a relative baseline without a learned critic. Clipping constrains each update and a frozen reference anchors the policy. | Implemented; population-standard-deviation normalization; per-completion loss reduction |
| REINFORCE / minimum risk training | Useful older sequence-level baselines that directly optimize nondifferentiable MT metrics. | Survey only |
| PPO | A value model may provide a richer baseline, at the cost of extra fitting and memory. | Survey only |
| DPO | An alternative when preference pairs are available; it changes the data and optimization setting. | Survey only |
| Dr. GRPO / DAPO | Investigate normalization and exploration choices when response length or reward variance behaves poorly. Reasoning-benchmark results are not evidence of MT improvement. | Survey only; not aliases for the implemented loss |

GRPO originates in [DeepSeekMath](https://arxiv.org/abs/2402.03300). MT-specific sequence-risk optimization predates GRPO: [Shen et al., ACL 2016](https://aclanthology.org/P16-1159/) train NMT against sequence-level evaluation criteria. This motivates including an MRT baseline rather than attributing every improvement to a newer optimizer.

[Dr. GRPO](https://arxiv.org/abs/2503.20783) identifies optimization biases in GRPO, including response-length effects. [DAPO](https://arxiv.org/abs/2503.14476) studies scalable policy optimization and exploration changes. Our inference is to monitor translation length and zero-variance reward groups and test alternative reductions separately; neither paper establishes the best reduction for PhoMT.

Sampling policy matters as much as optimizer choice. The implementation samples with full-vocabulary temperature scaling and recomputes old/current/reference likelihoods at the same temperature. Dropout is disabled for all these computations. Top-k, nucleus filtering, beam search, repetition processors, and inherited forced EOS are excluded from training sampling because they would require corresponding likelihood or action-mask treatment. A forced target-language BOS is excluded from the loss while remaining visible in the decoder context. See the official [Transformers generation API](https://huggingface.co/docs/transformers/main_classes/text_generation).

## Reward families

| Reward | Inputs | Strength | Limitation and recommended role |
|---|---|---|---|
| chrF++ | Candidate + reference | Cheap character and word n-gram overlap; useful dense sentence-level feedback | Surface similarity cannot establish semantic adequacy. Implemented default, normalized by 100 |
| Smoothed sentence BLEU | Candidate + reference | Cheap, familiar lexical signal | Short segments and valid paraphrases make it a noisy training signal. Optional implemented mixture component; report corpus BLEU separately |
| Reference-based COMET | Source + candidate + reference | Learned semantic evaluation related to human judgments | Check language/domain validity; extra model memory and possible overoptimization. Proposed extension |
| COMET-QE / reference-free ensembles | Source + candidate | Allows monolingual-source training | Frozen `Unbabel/wmt22-cometkiwi-da` is implemented; ensembles remain an extension. Reward-model blind spots need independent evaluation |
| xCOMET error severity | Source + candidate, depending on checkpoint/reference mode | Error spans can make feedback more diagnostic | Token-level credit requires alignment between evaluator spans and policy tokens. Proposed extension |
| Self-judge / LLM judge | Source + candidate + judging prompt | Can reduce dependence on labeled references | Evaluator bias, scoring cost, and self-reinforcement need separate controls. Research option |
| Human preference / MQM | Source + candidate(s) + human judgment | Direct assessment of adequacy, fluency, and errors | More expensive; use a stratified audit and preference-data collection |

The original [chrF paper](https://aclanthology.org/W15-3049/) defines character n-gram evaluation; [chrF++](https://aclanthology.org/W17-4770/) adds word n-grams. [BLEU](https://aclanthology.org/P02-1040/) is based on n-gram overlap and a brevity penalty. An average of smoothed sentence BLEU rewards is not corpus BLEU; this repository computes the latter over the complete evaluation corpus.

[COMET](https://aclanthology.org/2020.emnlp-main.213/) uses multilingual learned representations and human judgments for MT evaluation. This supports considering semantic reward signals, but it does not make a COMET checkpoint an infallible training objective. Keep its weights fixed during policy training and use a separate evaluator for checkpoint selection when practical.

[Fine-Grained Reward Optimization](https://aclanthology.org/2026.tacl-1.33/) reports benefits from xCOMET-based token-level error/severity feedback across encoder–decoder and LLM translation experiments. Our framework currently uses one sequence reward broadcast across generated tokens; implementing that paper requires changing the advantage interface and aligning spans, not simply replacing a scalar scorer.

Recent reference-free directions include [MiLMMT post-training](https://arxiv.org/abs/2608.10812), which combines two QE rewards, language-identification gating, and SFT/RL checkpoint interpolation. [SSR-Zero](https://aclanthology.org/2026.findings-acl.300/) studies self-judging rewards and complements them with COMET. Their reported settings motivate experiments; transfer to English–Vietnamese should be evaluated rather than assumed.

## Penalties and constraints

| Mechanism | Suggested use | Failure mode to check | Implemented? |
|---|---|---|---|
| KL to frozen SFT policy | Restrain drift from the initial translator | Excessive KL strength prevents learning; weak strength permits reward exploitation | Yes: sampled k3 estimator and configurable beta |
| PPO-style ratio clipping | Limit update size relative to the policy that produced the rollout | Reusing stale rollouts too often increases clipping and off-policy error | Yes: configurable epsilon and iterations |
| Wrong-language gate | Reduce reward for confidently off-target output | Language ID is unreliable for very short text, names, and code switching | Survey only |
| Repeated n-gram penalty | Discourage loops and duplicated clauses | Legitimate repetition, lists, and terminology can be penalized | Survey only |
| Length-ratio / omission penalty | Flag unusually short or long translations | A universal 1:1 ratio is inappropriate across languages and tokenizers | Survey only |
| EOS / truncation treatment | Detect length-cap terminations; consider exclusion or a calibrated soft penalty | Punishing every long sentence teaches under-translation | EOS masking implemented; truncation penalty is not |
| Entity/number preservation | Detect dropped or altered numbers, dates, placeholders, and names | Localization changes formats; exact string equality over-penalizes valid translations | Survey only |
| Copying / commentary / format penalty | Discourage source copying or explanations when translation-only output is required | Borrowed words, quotes, and identical named entities are legitimate | Survey only |
| SFT auxiliary loss or checkpoint interpolation | Retain translation competence while optimizing a proxy | Auxiliary weights can dominate RL or obscure attribution | Survey only |

[He et al., NAACL 2024](https://aclanthology.org/2024.naacl-long.451/) document rising QE reward alongside falling translation quality, and mitigate incorrect translations with heuristic reward penalties. [Mending the Holes](https://aclanthology.org/2026.findings-acl.1682/) similarly studies QE weaknesses amplified by multilingual RL. These motivate language, adequacy, and length checks; they do not validate a universal set of penalty thresholds.

A possible future objective is `r = w_c * chrF++/100 + w_q * calibrated_QE - lambda_rep * repetition - lambda_len * length_error - lambda_lang * wrong_language`. This is a proposed design, not implemented behavior. Fit scaling and thresholds on development data. Language-gated quality rewards may be preferable to additive penalties if invalid candidates retain high quality scores. Do not assume COMET or QE scores lie in [0,1] or mix raw scales without calibration.

## PhoMT experiment plan

1. Deduplicate and inspect train/development/test overlaps before quality experiments; the current loader makes a deterministic row split, not a deduplicated or document-aware split. Group documents before splitting if document identifiers are available.
2. Evaluate the SFT checkpoint with the exact tokenizer, source/target language direction, prompt, decode cap, and corpus metric settings used later.
3. Train chrF++ GRPO. Proposed pilot grid: G in {4,8}, temperature in {0.7,1.0}, beta in {0.01,0.04,0.1}, and learning rate in {5e-7,1e-6}. These are starting values, not tuned recommendations. Use a staged search rather than the full Cartesian product initially.
4. Ablate chrF++ vs chrF++/BLEU mixtures, then add a calibrated semantic reward. Evaluate each addition independently before combining penalties.
5. Track reward, group standard deviation, zero-variance group rate, sampled KL, clipping fraction, completion length, corpus BLEU/chrF++, off-target rate, truncation rate, repetition, and entity/number errors. Some require additional evaluators beyond the implemented logs.
6. Compare fixed seeds and equal sampled-token/optimizer budgets. `num_iterations=2` means two optimizer steps per rollout batch; Lightning `global_step` counts those optimizer steps.
7. Select on development data, then evaluate once on test and perform blinded bilingual review stratified by length, names/numbers, domain, and difficult examples.

## Implementation boundary

Implemented: single-device GRPO, two Transformers architectures, grouped rewards, reference KL, clipping, EOS masks, optional forced language BOS, CSV/JSONL/Hub loading, corpus BLEU/chrF++, checkpoint resume, and HF export. Source-aware rewards receive aligned source text; legacy two-argument callables remain supported. The new [bilingual SFT/RL workflow](../../archive/reports/bidirectional_training.md) adds a disjoint 80/20 training split, online augmentation, COMETKiwi + BLEU rewards, and per-direction corpus evaluation.

LoRA and NF4 QLoRA are supported for both parallel-data and self-play GRPO; adapter mode shares the frozen base with the KL reference. See [configuration and Qwen 7B memory analysis](../lora_qwen7b.md).

The [transfer optimizer](../transfer_learning.md) adds an L2-SP parameter anchor and optional per-tensor drift projection through the PyTorch optimizer interface. Hydra selects it by default; `optimizer=adamw` is the control. Its parameter-space controls complement reference KL; broader foundation-capability retention still needs independent evaluation.

Not implemented: PPO, DPO, MRT, Dr. GRPO/DAPO loss variants, sharded/distributed rollouts, vLLM serving, reference-based learned semantic evaluators in the supplied presets, token-level error rewards, or behavioral penalties beyond the KL term. The smoke tests establish software execution and gradient behavior with tiny random models. They establish no translation-quality improvement.
