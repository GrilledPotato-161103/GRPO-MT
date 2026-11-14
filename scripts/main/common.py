"""Configuration and preflight shared by the lightweight manual CLIs."""

import json
import math
import os
import time
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/runs/a100.yaml"


def load_config(
    path: Path, overrides: list[str], updates: dict[str, Any] | None = None
) -> dict[str, Any]:
    from omegaconf import DictConfig, OmegaConf

    def read_profile(profile: Path, parents: set[Path]) -> Any:
        profile = profile.resolve()
        if profile in parents:
            raise ValueError("Recursive base_config inheritance")
        value = OmegaConf.load(profile)
        if not isinstance(value, DictConfig):
            raise TypeError("Configuration must be a YAML mapping")
        base = value.pop("base_config", None)
        if base:
            value = OmegaConf.merge(
                read_profile(profile.parent / base, parents | {profile}), value
            )
        return value

    cfg = read_profile(path, set())
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    if updates:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(updates))
    result = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    if not isinstance(result, dict):
        raise TypeError("Configuration must be a YAML mapping")
    if result.get("token_filter_manifest"):
        result["token_filter_manifest"] = absolute(
            result["token_filter_manifest"]
        ).as_posix()
        manifest = json.loads(
            Path(result["token_filter_manifest"]).read_text(encoding="utf-8")
        )
        budget = manifest["token_budget"]
        if (
            budget["prompt_limit"]
            != budget["content_limit"] + budget["prompt_overhead"]
            or budget["target_limit"] != budget["content_limit"] + 1
        ):
            raise ValueError("Invalid complete-pair token budgets in filter manifest")
        expected = {
            "model_name": result["model_name"],
            "tokenizer_revision": result["policy_revision"],
            "prompt_template": result["prompt_template"],
            "use_chat_template": result["use_chat_template"],
            "chat_template_kwargs": result.get("chat_template_kwargs"),
            "content_limit": result["source_content_limit"],
            "language_codes": result.get("language_codes"),
        }
        if any(budget.get(key) != value for key, value in expected.items()):
            raise ValueError(
                "Token-filter manifest differs from model/instruction; re-filter with this tokenizer"
            )
        if not result.get("strict_prompt_length") or result.get("augmentation", {}).get(
            "enabled"
        ):
            raise ValueError(
                "Original-pair bounded profiles require strict lengths and disabled online augmentation"
            )
        result["source_cap"] = budget["prompt_limit"]
        result["target_cap"] = budget["target_limit"]
    text = json.dumps(result)
    for key in ("WANDB_API_KEY", "HUGGING_FACE_API_KEY", '"api_key"', '"token"'):
        if key in text:
            raise ValueError(
                "Keep credentials in environment variables, not configuration"
            )
    interval, iterations = (
        result["evaluation_every_optimizer_steps"],
        result["num_iterations"],
    )
    for name, value in (
        ("evaluation_every_optimizer_steps", interval),
        ("num_iterations", iterations),
        ("group_size", result["group_size"]),
        ("logprob_batch_size", result["logprob_batch_size"]),
        *result["batch_sizes"].items(),
    ):
        if value is None and name == "logprob_batch_size":
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if result["group_size"] < 2 or interval % iterations:
        raise ValueError(
            "group_size must be >=2 and evaluation interval divisible by num_iterations"
        )
    if result.get("budget_control", True) and (
        not math.isfinite(float(result["max_hours"])) or result["max_hours"] <= 0
    ):
        raise ValueError("max_hours must be positive and finite")
    if result["wandb"]["mode"] not in {"online", "offline"}:
        raise ValueError("wandb.mode must be online or offline")
    for name in ("source_cap", "target_cap", "comet_batch_size"):
        value = result[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < (2 if name == "target_cap" else 1)
        ):
            raise ValueError(f"{name} is outside its valid token/batch range")
    if result["trainer"]["devices"] != 1:
        raise ValueError("Manual GRPO currently supports one device")
    loader = result.get("data_loader", {})
    microbatch = result.get("sft_microbatch_size")
    if microbatch is not None and (
        isinstance(microbatch, bool)
        or not isinstance(microbatch, int)
        or microbatch < 1
    ):
        raise ValueError("sft_microbatch_size must be a positive integer")
    for name, minimum in (("num_workers", 0), ("prefetch_factor", 1)):
        value = loader.get(name, minimum)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"data_loader.{name} must be an integer >= {minimum}")
    weights = list(result["reward"].values())
    if (
        any(
            not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
            for value in weights
        )
        or sum(weights) <= 0
    ):
        raise ValueError(
            "Reward weights must be finite/nonnegative with a positive sum"
        )
    from src.models.components.kl_control import KLController

    KLController(result.get("beta", 0.04), **result.get("kl_control", {}))
    if result.get("optimizer", "transfer") not in {"adamw", "transfer"}:
        raise ValueError("optimizer must be adamw or transfer")
    from src.models.components.optim.plateau import plateau_options

    for stage, options in result.get("lr_scheduler", {}).items():
        if stage not in {"sft", "rl"}:
            raise ValueError("lr_scheduler stages must be sft or rl")
        plateau_options(options)
    for stage, steps in result.get("max_optimizer_steps", {}).items():
        if (
            stage not in {"sft", "rl"}
            or isinstance(steps, bool)
            or not isinstance(steps, int)
            or steps < 1
            or (stage == "rl" and steps % iterations)
        ):
            raise ValueError(
                "max_optimizer_steps must be positive; RL must divide by num_iterations"
            )
    sac = result.get("sac_reward", {})
    if result.get("correction") is not None:
        from src.models.components.correction_objective import CorrectionOptions

        CorrectionOptions(**result["correction"])
        if (
            not sac.get("enabled")
            or sac.get("fusion", {}).get("enabled")
            or sac.get("consistency_weight", 0.2) != 0
            or sac.get("polarity", {}).get("enabled")
            or result.get("beta", 0.04) != 0
            or result.get("temperature", 1) != 1
        ):
            raise ValueError(
                "Correction training requires direct SAC, no fusion/consistency/polarity/KL, and temperature=1"
            )
    if sac.get("fusion", {}).get("enabled") and not sac.get("enabled"):
        raise ValueError("Dynamic fusion requires SAC enabled")
    if sac.get("enabled"):
        from src.models.components.reward_fusion import DynamicRewardFusion

        parse_retries = sac.get("max_parse_retries", 2)
        if (
            isinstance(parse_retries, bool)
            or not isinstance(parse_retries, int)
            or parse_retries < 0
        ):
            raise ValueError(
                "sac_reward.max_parse_retries must be a nonnegative integer"
            )
        DynamicRewardFusion(**sac.get("fusion", {}))
        from src.models.components.signed_sac import SignedSAC

        polarity = SignedSAC(**sac.get("polarity", {}))
        if polarity.enabled and (
            not sac.get("fusion", {}).get("enabled")
            or sac.get("fusion", {}).get("sac_direct_weight", 0.5) != 1
        ):
            raise ValueError(
                "Signed SAC requires enabled direct-only categorical fusion"
            )
        if sac.get("label_schema", "legacy") not in {"legacy", "polarity"}:
            raise ValueError("Unknown SAC label schema")
        if (
            result["reward"]["bleu_weight"]
            or result["reward"]["chrf_weight"]
            or result["reward"]["comet_weight"] <= 0
        ):
            raise ValueError("SAC requires COMET and zero lexical reward weights")
        if result.get("reward_harness", {}).get("enabled"):
            raise ValueError("Disable reward_harness for the SAC experiment")
        mode = sac.get("consistency_mode", "round_trip")
        if mode not in {"round_trip", "symmetric_endpoints"}:
            raise ValueError("Unknown SAC consistency_mode")
        if mode == "symmetric_endpoints" and (
            not result.get("paired_batches")
            or any(
                size < 2 or size % 2
                for size in (
                    *result["batch_sizes"].values(),
                    result["evaluation"]["batch_size"],
                )
            )
        ):
            raise ValueError(
                "Symmetric SAC requires paired_batches=true and even training/evaluation batch sizes"
            )
        penalty = sac.get("consistency_weight", sac.get("round_trip_weight", 0.2))
        batch_size = sac.get("round_trip_batch_size", 8)
        if (
            isinstance(penalty, bool)
            or not isinstance(penalty, (int, float))
            or not math.isfinite(penalty)
            or penalty < 0
        ):
            raise ValueError("SAC consistency weight must be finite and nonnegative")
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size < 1
        ):
            raise ValueError("SAC round_trip_batch_size must be positive")
        tokens = sac.get("round_trip_max_new_tokens", 256)
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 1:
            raise ValueError("SAC round_trip_max_new_tokens must be positive")
    if sac.get("cache_dir"):
        sac["cache_dir"] = absolute(sac["cache_dir"]).as_posix()
    if result.get("reward_harness", {}).get("cache_dir"):
        result["reward_harness"]["cache_dir"] = absolute(
            result["reward_harness"]["cache_dir"]
        ).as_posix()
    if result.get("length_cache_dir"):
        result["length_cache_dir"] = absolute(result["length_cache_dir"]).as_posix()
    for name in (
        "output_dir",
        "sft_export_dir",
        "hf_home",
        "comet_checkpoint",
        "dataset_manifest",
    ):
        result[name] = absolute(result[name]).as_posix()
    result["data_files"] = {
        name: absolute(value).as_posix() for name, value in result["data_files"].items()
    }
    result["evaluation"]["output_dir"] = absolute(
        result["evaluation"]["output_dir"]
    ).as_posix()
    if (
        Path(result["model_name"]).is_absolute()
        or (ROOT / result["model_name"]).is_dir()
    ):
        result["model_name"] = absolute(result["model_name"]).as_posix()
    result["wandb_offline"] = result["wandb"]["mode"] == "offline"
    return cast(dict[str, Any], result)


def absolute(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def environment(cfg: dict[str, Any]) -> None:
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    os.environ.update(
        PROJECT_ROOT=str(ROOT),
        HF_HOME=cfg["hf_home"],
        COMET_CHECKPOINT=cfg["comet_checkpoint"],
        TOKENIZERS_PARALLELISM="false",
        WANDB_DISABLE_CODE="true",
        WANDB_DISABLE_GIT="true",
    )


def deadline(cfg: dict[str, Any], output: Path) -> float | None:
    if not cfg.get("budget_control", True):
        return None
    value = time.time() + float(cfg["max_hours"]) * 3600
    previous = output / "rental_config.json"
    limits = [cfg.get("deadline_unix")]
    if previous.is_file():
        limits.append(json.loads(previous.read_text()).get("graceful_deadline_unix"))
    for limit in limits:
        if limit is not None:
            value = min(value, float(limit))
    if not math.isfinite(value) or value <= time.time():
        raise ValueError(
            "Authorized deadline has passed; the launcher will not extend it"
        )
    return value


def ensure_comet(cfg: dict[str, Any]) -> None:
    if cfg["reward"]["comet_weight"] <= 0:
        return
    checkpoint = Path(cfg["comet_checkpoint"])
    if checkpoint.is_file() and checkpoint.stat().st_size:
        return
    from huggingface_hub import snapshot_download

    from src.utils.credentials import environment_secret

    token = environment_secret("HUGGING_FACE_API_KEY")
    if not token:
        raise ValueError(
            "Set HUGGING_FACE_API_KEY and accept Unbabel/wmt22-cometkiwi-da conditions"
        )
    snapshot_download(
        "Unbabel/wmt22-cometkiwi-da",
        revision=cfg["comet_revision"],
        token=token,
        allow_patterns=["checkpoints/model.ckpt", "hparams.yaml"],
        local_dir=str(checkpoint.parent.parent),
    )
    del token
    if not checkpoint.is_file():
        raise ValueError("COMET checkpoint path must end in checkpoints/model.ckpt")


def check_logging(cfg: dict[str, Any]) -> None:
    if cfg["wandb"]["enabled"] and not cfg["wandb_offline"]:
        from src.utils.credentials import environment_secret

        key = environment_secret("WANDB_API_KEY")
        if not key:
            raise ValueError(
                "Set WANDB_API_KEY, wandb.mode=offline, or wandb.enabled=false"
            )
        os.environ["WANDB_API_KEY"] = key


def check_feedback(cfg: dict[str, Any]) -> None:
    """Read API credentials only for explicitly enabled feedback consumers."""
    from src.utils.credentials import environment_secret

    for consumer in (cfg.get("reward_harness", {}), cfg.get("sac_reward", {})):
        backend = consumer.get("backend", {})
        if (
            not consumer.get("enabled")
            or consumer.get("cache_only")
            or backend.get("kind", "api") != "api"
        ):
            continue
        name = backend.get("api_key_env", "CLAUDE_AUTH_KEY")
        key = environment_secret(name)
        if not key:
            raise ValueError(f"Set {name} for the enabled feedback API")
        os.environ[name] = key


def compose_stage(cfg: dict[str, Any], stage: str, cutoff: float | None) -> Any:
    from hydra import compose, initialize_config_dir

    from scripts.main.pipeline import stage_overrides

    output = Path(cfg["output_dir"])
    environment(cfg)
    os.environ["TRAIN_RUN_GROUP"] = cfg.get("run_group") or "manual-preview"
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        composed = compose(
            config_name="sft" if stage == "sft" else "grpo",
            overrides=stage_overrides(
                stage, output, cutoff, cfg["policy_revision"], runtime=cfg
            ),
        )
    composed.paths.output_dir = str(output / stage)
    composed.paths.work_dir = str(ROOT)
    return composed
