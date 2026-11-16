# Research notes

Start with the [consolidated optimization and training scheme](translation_training_scheme_20261116.md)
for the current design and implementation status. These notes retain the analysis
behind dated experiments; historical run-profile references may be absent from
this checkout. The new MQM scheme is not an executable default.

Latest requested execution: [10k-pair SFT → 10k-update FP16 NLLB run](nllb_10k_fp16_run_20261116.md).
Instance 60824538 is running the detached startup pipeline; current observations
and the on-demand status command are recorded in that run note.

- `GRPO_MT_BACKLOG.md`: implementation and experiment backlog.
- `reward_to_action_20261113.md`: reward-to-gradient audit and proposed use of failed translations as anchored preference examples.
- `feedback_learning_investigation_20261111.md`: why prior feedback failed to improve rewards.
- `mt_policy_strategy_survey_20261109.md`: policy criteria, conflicts and relaxed strategies.
- `REWARD_AUDIT_20261109.md`: component-level reward audit.
- `mt_rl_survey.md`: earlier machine-translation reinforcement-learning survey.
- [Ten SAC grading engines and prompts](sac_grading_engines_20261115.md):
  source-grounded grading, coverage/entailment, pairwise comparison, naturalness,
  context, invariant checks, adjudication, calibration and verified repair.
  [Reusable JSON prompt pack](sac_grading_prompts_20261115.json) contains all 16
  stage prompts; these are research templates requiring new runtime adapters.
- [MQM observations to criterion-specific gradients](mqm_vector_optimization_20261116.md):
  selected engine 1; differentiable local repair contrasts, explicit token-to-model
  chain rule, adequacy constraints, integration plan and numerical verification.
- [Translation penalty audit and priorities](translation_penalties_audit_20261116.md):
  active versus disabled controls, lexical checkpoint-selection dependence,
  local negative supervision, retention and source-grounded error budgets.
