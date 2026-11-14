# Decoder-only versus encoder–decoder: transfer and translation

Reviewed 13 November 2026. This is a targeted literature review, not a PRISMA systematic review or a new experiment. It prioritizes controlled architecture studies, then modern adaptation experiments and surveys. Published versions were preferred; preprint limitations are identified below.

## Conclusion

**For supervised translation under a limited compute budget, the literature gives substantial support to an encoder–decoder inductive advantage. Architecture should have been a first-class hypothesis in our diagnosis.** The evidence does not establish that causal decoder-only models are universally worse at transferring to new tasks. Rankings depend on training objective, pretraining, scale, adaptation and the meaning of generalization.

The supported mechanism is source conditioning: bidirectional input representations, explicit encoder–decoder attention and the separation of input processing from generation. Describing knowledge as intrinsically stored does not distinguish the architectures: both have learned parameters and input-dependent hidden states. A decoder-only target position can attend to the entire preceding source, even though individual source-token states are causally restricted. The distinction is the computation available to those source representations, not the existence of a latent representation.

My engineering recommendation for this project is to make a translation-pretrained encoder–decoder a required baseline before further decoder-only GRPO tuning. That is a decision informed by the literature; it is not a demonstrated EN–VI result or a claim that architecture caused the stopped run's plateau.

## Controlled studies and modern evidence

| Study | Relevant evidence | Scope and limitation |
| --- | --- | --- |
| [Raffel et al., T5, JMLR 2020](https://jmlr.org/papers/volume21/20-074/20-074.pdf), §3.2, Table 2 | With the same reported P-parameter/M-compute budget and denoising objective, shared-parameter encoder–decoder achieved **26.72 En–De BLEU**, versus **25.09** for a causal language model. Shared encoder–decoder also outperformed PrefixLM. | Supports conditional-generation and transfer advantages beyond simply adding parameters. Historical training regime; it does not directly rank modern Qwen models. |
| [Zhang et al., ICML 2022](https://proceedings.mlr.press/v162/zhang22h.html), §6–7 | Supervised multilingual transfer favored encoder–decoder, with larger gaps on lower-resource directions. Causal source masking hurt translation. Bidirectional-source PrefixLM narrowed the gap and could provide stronger zero-shot transfer. | Direct MT architecture evidence, with scaling and data-regime comparisons. Many experiments trained NMT models rather than adapting today's general pretrained LLMs. |
| [Wang et al., ICML 2022](https://proceedings.mlr.press/v162/wang22u.html), §3 and §6 | Causal autoregressive models performed best at zero-shot tasks after self-supervised pretraining alone. After multitask prompted fine-tuning, denoising encoder–decoder models performed best. | A counterexample to a universal ranking. Compute was approximately matched, not parameter count: encoder–decoder models had about 11B parameters versus 4.8B, with a residual compute difference around 10%. |
| [Qu et al., MRL 2025](https://aclanthology.org/2025.mrl-main.13/), §4 and §8 | Identifies inadequate target-language conditioning of source representations in decoder-only multilingual NMT. Two-stage decoding and instruction-level contrastive learning improved transfer on TED-19 and OPUS-100. | Mechanistically close to the user's concern. The authors explicitly state that extension to pretrained general-purpose LLMs remains untested. Improved decoder variants also demonstrate that the limitation can be mitigated. |
| [Zhang et al., Encoder–Decoder Gemma, 2025 preprint](https://arxiv.org/html/2504.06225), §5–6 | A modern Gemma adaptation study: keeping the adapted encoder causal reduced aggregate instruction-tuning score from **46.4 to 41.7**. Overall post-training performance favored encoder–decoder adaptation. | Comparisons emphasize inference budget; encoder–decoder variants have more parameters and undergo adaptation. It is not a universal MT win: Table 3 reports WMT23 BLEURT **72.0 for Gemma 2 9B versus 71.1 for 9B–9B**. |
| [Luo et al., ACL Findings 2025](https://aclanthology.org/2025.findings-acl.490/), §5–6 and Limitations | LaMaTE uses a pretrained LLM as encoder with a smaller translation decoder. It reports strong translation-related task generalization and **2.4–6.5× decoding speedups** against its baselines. | Supports a hybrid route. Its first stage uses 40M parallel pairs; this is not evidence that a cheap architectural conversion would automatically improve our run. |
| [T5Gemma 2, December 2025 preprint](https://arxiv.org/html/2512.14856), §3–4 | Extends decoder-to-encoder–decoder adaptation to Gemma 3, multilingual inputs and long contexts; reports improved post-training performance with lightweight SFT. | Evidence that encoder–decoder research remains relevant with modern backbones. Comparisons differ in parameters/adaptation; post-training results are not a matched EN–VI translation experiment. |
| [Manna et al., March 2026 preprint](https://arxiv.org/html/2603.17952), §4–7 | Decoder-only MT models did not consistently surpass encoder–decoder systems on gender-context disambiguation, and instruction tuning improved some contextual behavior. | Diagnostic evidence about one translation capability, not proof of overall translation superiority; automatic gender analysis introduces noise. |

These results are empirical evidence, not a mathematical theorem ordering architectures for every task. Source conditioning, supervised language-pair transfer, unseen-task prompting and long-context generalization are distinct questions.

## What surveys contribute

1. [Li et al., *Pre-trained Language Models for Text Generation*](https://arxiv.org/pdf/2201.05273), §4–5: describes causal input restrictions, PrefixLM and explicit cross-attention as different ways to model conditional generation. The inspected version is the May 2022 preprint, preceding later LLM developments; its architectural discussion is useful, but its performance generalizations should not be treated as current universal rankings. The later journal publication is ACM Computing Surveys 2024.
2. [Ataman et al., *Machine Translation in the Era of Large Language Models*, 2025](https://www.mdpi.com/2078-2489/16/9/723), §5–6: emphasizes that dedicated MT systems remain competitive and economical on specialized translation tasks, while LLMs add prompting and broader capabilities. It also discusses data imbalance and generalization across languages. This is synthesis rather than a controlled architecture intervention.
3. [Gain et al., *Bridging the Linguistic Divide*, January 2026 revision](https://arxiv.org/html/2504.01919v4), §2 and §5: reviews adaptation, language-resource dependence and encoder–decoder/LLM comparisons. It explicitly avoids treating heterogeneous cross-paper scores as directly comparable experiments. The reviewed version is a preprint.

Surveys identify recurring patterns and relevant experiments. The controlled studies carry more weight for architecture causality; model leaderboard comparisons combine architecture with data, scale and training differences.

## A precise version of the representation argument

For either architecture, translation is modeled as:

\[
P_\theta(y\mid x)=\prod_t P_\theta(y_t\mid x,y_{<t}).
\]

The factorization itself does not decide how the model should represent the source.

- In a causal decoder-only stack, source position \(i\) has access to preceding source tokens, but not later source positions. Generated target positions can access the complete preceding source and previous target tokens.
- A conventional encoder–decoder constructs bidirectional source representations, then exposes them through cross-attention during target generation. It can allocate different capacity to source processing and target generation.
- A PrefixLM retains one stack but allows bidirectional attention within the source. A textual prompt prefix alone does **not** implement this attention pattern.

These distinctions are described in the T5 architecture comparison and Li et al.'s survey. They provide a plausible explanation for translation-specific efficiency and transfer differences. They do not imply a causal decoder is unable to learn source–target mappings.

In our implementation, LoRA changes projection weights while the attention mask and module topology remain fixed. Increasing rank or adding feed-forward projections does not create bidirectional source attention or a new cross-attention module. Changing the adjective in the translation instruction also does not change those structural properties. Such changes may help adaptation while leaving an architectural constraint intact.

## Implications for the stopped Qwen3 run

The available corpus contains nearly three million original parallel pairs. Our restriction was short training exposure and rental/feedback cost, not an absence of parallel data. We should therefore distinguish a compute-limited adaptation problem from the extremely low-resource settings used in some surveyed studies.

The final run confirms active gradients and adapter updates; it does not compare architectures. Its inconsistent validation sample counts, coarse reward rankings and occasional truncation remain independently established problems. None identifies the architecture's contribution to quality. Consequently, ranking architecture below rewards as an explanation was not supported by a causal experiment.

Architecture changes may also improve GPU efficiency without proportionally improving total GRPO speed: SAC feedback consumed 59.2% of the stopped run's measured batch wall time. The API bottleneck needs separate treatment. Published decoding speedups should not be applied to our entire training pipeline.

## The experiment that would settle the practical decision

This is a proposed protocol, not an executed benchmark or a rental request.

1. Start with SFT-only adaptation of a translation-pretrained encoder–decoder and the current Qwen3 baseline, using the same original pair IDs and complete-pair token filtering. Use each model's correct native formatting and target-language controls.
2. Fix validation at the same 1,600 paired examples in both directions. Decouple evaluation batch size from training batch size; do not allow a batch cap to change the evaluated set.
3. Treat equal wall-time/VRAM budget and equal parameter-count comparisons as separate questions. Report actual trainable parameters, tokens processed, throughput and quality per GPU-hour.
4. Measure COMET, BLEU, chrF, omissions, entity/number/negation errors and a small blinded human adequacy sample. Keep checkpoints/examples and evaluation protocol identical across measurement points.
5. Only after this SFT comparison, apply a matched RL block with unchanged reward settings. Otherwise reward noise and architecture remain confounded.

Different public pretrained models test the practical choice of model package, not architecture alone. A causal architecture study would additionally match tokenizer, data, pretraining and initialization as closely as feasible, or use an attention-mask/adaptation ablation like the modern Gemma study.

**Decision:** prioritize an encoder–decoder baseline for this translation framework. The user's translation-specific concern is supported. The unrestricted claim that decoder-only models necessarily transfer poorly to all tasks, or that intrinsic storage of knowledge is the distinguishing cause, is not supported.
