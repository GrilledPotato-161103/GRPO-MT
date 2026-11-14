"""Reproducible NLLB memory, matrix-FLOP and rental/API planning estimates.

Prices are a dated CLI snapshot. Latency ranges are explicit scenarios, not
measurements of rented GPUs. GPU peak rates use dense BF16 with FP32 accumulation.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any


def forward_flops(
    config: dict[str, Any], source: int, target: int, batch: int = 1
) -> dict[str, float]:
    """Count multiply-add as 2 FLOPs; embeddings are lookups, tied LM head is GEMM."""
    if min(source, target, batch) < 1:
        raise ValueError("Token lengths and batch must be positive")
    d = config["d_model"]
    encoder = config["encoder_layers"] * (
        8 * source * d * d
        + 4 * source * d * config["encoder_ffn_dim"]
        + 4 * source * source * d
    )
    decoder = config["decoder_layers"] * (
        12 * target * d * d
        + 4 * source * d * d
        + 4 * target * d * config["decoder_ffn_dim"]
        + 4 * target * target * d
        + 4 * source * target * d
    )
    head = 2 * target * d * config["vocab_size"]
    return {
        "transformer": float(batch * (encoder + decoder)),
        "lm_head": float(batch * head),
        "total": float(batch * (encoder + decoder + head)),
    }


def estimate(
    config: dict[str, Any],
    benchmark: dict[str, Any],
    worst: dict[str, Any],
    offers: list[dict[str, Any]],
    sft_steps: int = 2000,
    rl_steps: int = 10000,
    iterations: int = 2,
    sft_probe: dict[str, Any] | None = None,
    token_budget: int = 4096,
    sft_microbatch_size: int = 16,
    bf16_base_parameters: int | None = None,
) -> dict[str, Any]:
    if min(sft_steps, rl_steps, iterations) < 1 or rl_steps % iterations:
        raise ValueError("Need positive steps and an integral number of fresh rollouts")
    # Per-source means are short in the bounded PhoMT manifest. Round upward.
    average = forward_flops(config, 24, 24)
    capped = forward_flops(config, 258, 258)
    projected = benchmark.get("quantization") is not None
    if projected and bf16_base_parameters is None:
        raise ValueError(
            "A quantized probe needs an explicit BF16 parameter count for projection"
        )
    # Empirical matrix-op profiler includes LoRA and checkpoint recomputation.
    full = [
        x
        for x in worst["cases"]
        if x["status"] == "ok"
        and x.get("source_positions", 258) == 258
        and x.get("decoder_positions", 258) == 258
    ]
    if not full or token_budget < 516 or sft_microbatch_size < 1:
        raise ValueError(
            "Need a successful full-cap probe, token budget >=516 and positive SFT microbatch"
        )
    worst_case = max(full, key=lambda x: x["batch_size"])
    profiled = next(
        x for x in worst["cases"] if x.get("registered_operator_flops_forward_backward")
    )
    profile_forward = forward_flops(
        config,
        profiled.get("source_positions", 258),
        profiled.get("decoder_positions", 258),
    )
    measured_grad = (
        profiled["registered_operator_flops_forward_backward"] / profiled["batch_size"]
    )
    calibration = measured_grad / (
        3 * profile_forward["transformer"] + 2 * profile_forward["lm_head"]
    )
    if projected:
        calibration = 1.0  # NF4 custom kernels are not fully counted by the profiler.
    grad_average = calibration * (3 * average["transformer"] + 2 * average["lm_head"])
    grad_capped = calibration * (3 * capped["transformer"] + 2 * capped["lm_head"])
    sources, candidates, fallback = 16, 64, 4
    pool = candidates + fallback
    # One generation + rollout scoring; external anchors also need scoring.
    # Subsequent updates refresh coefficients but share the same judge labels.
    rollout_flops = (2 * candidates + fallback + (iterations - 1) * pool) * average[
        "total"
    ] + iterations * pool * grad_average
    cap_rollout = (2 * candidates + fallback + (iterations - 1) * pool) * capped[
        "total"
    ] + iterations * pool * grad_capped
    # A conservative linear extrapolation from a measured 4-row full-cap graph.
    measured_base = benchmark["base_allocated_gib"]
    base = (
        (2 * bf16_base_parameters + 12 * benchmark["trainable_parameters"]) / 2**30
        if projected and bf16_base_parameters is not None
        else measured_base
    )
    max_rows = token_budget // (258 + 258)
    policy_reserved = base + max_rows / worst_case["batch_size"] * (
        worst_case["peak_reserved_gib"] - measured_base
    )
    comet_weights = 560e6 * 4 / 2**30  # Frozen XLM-R-large COMET, FP32 approximation.
    planned_vram = policy_reserved + comet_weights + 0.75 + 2.0
    planned_sft = None
    if sft_probe is not None:
        sft_reserved = base + sft_microbatch_size / sft_probe["batch_size"] * (
            sft_probe["peak_reserved_gib"] - sft_probe["base_gib"]
        )
        planned_sft = sft_reserved + comet_weights + 0.75 + 2.0
    else:
        planned_sft = (
            base
            + sft_microbatch_size
            / worst_case["batch_size"]
            * (worst_case["peak_reserved_gib"] - measured_base)
            + comet_weights
            + 0.75
            + 2
        )
    reference_config = {
        "d_model": 1024,
        "encoder_layers": 12,
        "decoder_layers": 12,
        "encoder_ffn_dim": 4096,
        "decoder_ffn_dim": 4096,
        "vocab_size": 256206,
    }
    workload_scale = average["total"] / forward_flops(reference_config, 24, 24)["total"]
    rollouts = rl_steps // iterations
    cache_fraction = 0.20
    requests = rollouts * (candidates + fallback) * (1 - cache_fraction)
    evaluations = math.ceil(sft_steps / 1000) + math.ceil(rl_steps / 1000)
    eval_requests = evaluations * 200 * 16 * (1 - cache_fraction)
    # OpenRouter published GPT-6 Luna: $0.10/M input, $0.50/M output (2026-11-14).
    api_range = [
        (requests + eval_requests) * (inp * 0.10 + out * 0.50) / 1e6
        for inp, out in ((200, 4), (400, 8))
    ]
    hardware: dict[str, dict[str, Any]] = {
        "RTX 3090": {
            "vram_gib": 24,
            "bf16_fp32_accum_tflops": 71.2,
            "gpu_seconds_per_update": [2.0, 3.5],
        },
        "RTX 4090": {
            "vram_gib": 24,
            "bf16_fp32_accum_tflops": 165.2,
            "gpu_seconds_per_update": [1.0, 2.0],
        },
        "L40S": {
            "vram_gib": 48,
            "bf16_fp32_accum_tflops": 362.05,
            "gpu_seconds_per_update": [1.0, 2.0],
        },
        "RTX PRO 4500": {
            "vram_gib": 32,
            "bf16_fp32_accum_tflops": None,
            "gpu_seconds_per_update": [1.0, 2.0],
            "compatibility": "Requires CUDA 12.8+ / sm_120-capable PyTorch; untested on this hardware",
        },
    }
    plans = []
    for name, spec in hardware.items():
        candidates_for_gpu = [
            x
            for x in offers
            if x["gpu_name"] == name
            and x["cpu_ram"] >= 32000
            and x["cpu_cores_effective"] >= 12
            and x["reliability"] > 0.99
            and x["inet_down"] >= 300
            and "CN" not in x.get("geolocation", "")
            and x.get("cuda_max_good", 12.6) >= 12.6
            and (name != "RTX 4090" or x["gpu_ram"] < 25000)
        ]
        if not candidates_for_gpu:
            continue
        offer = min(candidates_for_gpu, key=lambda x: x["dph_total"])
        price = offer["dph_total"]
        # API wait is independent of GPU. Two updates share each judge round.
        feedback_seconds = [3.0, 6.0]
        seconds = [
            a * workload_scale + b
            for a, b in zip(
                spec["gpu_seconds_per_update"], feedback_seconds, strict=True
            )
        ]
        # Explicit extra allowance: setup 10m, SFT 2000 steps at .5s/step,
        # 12 evaluations at 5m each. Actual evaluation includes SAC requests.
        overhead_hours = (
            10 * 60
            + sft_steps * 0.5 * workload_scale
            + evaluations * 5 * 60 * workload_scale
        ) / 3600
        hours = [rl_steps * value / 3600 + overhead_hours for value in seconds]
        plans.append(
            {
                "gpu": name,
                **spec,
                "offer_id": offer["id"],
                "location": offer["geolocation"],
                "hourly_usd_including_storage": price,
                "priced_storage_gib": round(
                    (price - offer["dph_base"]) * 720 / offer["storage_cost"], 2
                )
                if offer.get("storage_cost", 0) > 0 and "dph_base" in offer
                else None,
                "reliability": offer["reliability"],
                "advertised_download_mbps": offer["inet_down"],
                "dlperf_proxy": offer["dlperf"],
                "vram_times_bf16_tflops": spec["vram_gib"]
                * spec["bf16_fp32_accum_tflops"]
                if spec["bf16_fp32_accum_tflops"] is not None
                else None,
                "vram_times_bf16_tflops_per_hourly_dollar": spec["vram_gib"]
                * spec["bf16_fp32_accum_tflops"]
                / price
                if spec["bf16_fp32_accum_tflops"] is not None
                else None,
                "seconds_per_rl_update_SCENARIO": seconds,
                "total_hours_SCENARIO": hours,
                "gpu_usd_SCENARIO": [h * price for h in hours],
                "api_usd_SCENARIO": api_range,
                "total_usd_SCENARIO": [
                    h * price + api for h, api in zip(hours, api_range, strict=True)
                ],
            }
        )
    return {
        "as_of": "2026-11-14 Asia/Saigon",
        "scope": f"{benchmark.get('model', 'NLLB')} all-linear r8 LoRA, native EN/VI, GPU COMET + OpenRouter SAC",
        "gpu_work_time_scale_assumption": workload_scale,
        "model_parameters_including_lora": (
            bf16_base_parameters + benchmark["trainable_parameters"]
        )
        if projected and bf16_base_parameters is not None
        else benchmark["parameters"],
        "projection_from_quantized_probe": projected,
        "trainable_parameters": benchmark["trainable_parameters"],
        "schedule": {
            "sft_updates": sft_steps,
            "rl_updates": rl_steps,
            "updates_per_fresh_rollout": iterations,
            "fresh_rollouts": rollouts,
            "directional_sources_per_rollout": sources,
        },
        "flops": {
            "multiply_add_counts_as": 2,
            "average_source_target_positions": [24, 24],
            "forward_per_candidate": average,
            "policy_per_rl_update_average_tflop": rollout_flops / iterations / 1e12,
            "policy_per_rl_update_full_cap_tflop": cap_rollout / iterations / 1e12,
            "excludes": [
                "COMET forward (~1-2 additional TFLOP/update)",
                "API inference",
                "softmax/normalization",
                "sampling and CPU kernels",
            ],
            "scope": "Matrix-op estimate; generation approximated by one cached-decoder forward",
        },
        "memory": {
            "policy_base_gib": base,
            "measured_probe_base_gib": measured_base,
            "measured_probe_full_cap_batch_size": worst_case["batch_size"],
            "measured_probe_full_cap_peak_allocated_gib": worst_case[
                "peak_allocated_gib"
            ],
            "logprob_token_budget": token_budget,
            "max_full_cap_graph_rows": max_rows,
            "sft_microbatch_size": sft_microbatch_size,
            "estimated_policy_reserved_gib": policy_reserved,
            "estimated_comet_fp32_weights_gib": comet_weights,
            "comet_activations_allowance_gib": 0.75,
            "headroom_gib": 2,
            "rl_planning_total_gib": planned_vram,
            "sft_planning_total_gib": planned_sft,
            "sft_memory_method": "teacher-forced probe"
            if sft_probe is not None
            else "same-model GRPO activation envelope (conservative estimate)",
            "planning_total_gib": max(planned_vram, planned_sft or 0.0),
            "scope": "BF16 weight bytes plus NF4 workspace envelope: conservative projection, not measured BF16"
            if projected
            else "COMET memory not measured; linear graph extrapolation is a planning bound, not a fitted-host guarantee",
        },
        "api": {
            "model": "openai/gpt-6-luna",
            "input_usd_per_million": 0.10,
            "output_usd_per_million": 0.50,
            "assumed_prompt_tokens": [200, 400],
            "assumed_output_tokens": [4, 8],
            "assumed_cache_hit_fraction": cache_fraction,
            "training_requests": requests,
            "validation_requests": eval_requests,
            "usd_range": api_range,
            "excludes": [
                "retries",
                "reasoning tokens",
                "provider markups",
                "teacher-generated corrections (disabled)",
            ],
        },
        "plans": plans,
        "warning": "Neither peak TFLOPs nor VRAM*TFLOPs predicts wall time. Rented-GPU and live-judge latency have not been measured. No instance was rented.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offers", type=Path, default=Path("diagnostics/offers_20261114.json")
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("diagnostics/correction_20261114/gpu.json"),
    )
    parser.add_argument(
        "--worst", type=Path, default=Path("diagnostics/correction_20261114/worst.json")
    )
    parser.add_argument(
        "--config", type=Path, default=Path("logs/assets/nllb-600m/config.json")
    )
    parser.add_argument(
        "--sft-probe",
        type=Path,
        default=Path("diagnostics/correction_20261114/sft.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("diagnostics/correction_20261114/cost.json")
    )
    parser.add_argument(
        "--run-config",
        type=Path,
        help="Use model-size-specific microbatches and schedule",
    )
    parser.add_argument(
        "--bf16-base-parameters",
        type=int,
        help="Logical parameter count when projecting BF16 from NF4",
    )
    args = parser.parse_args()

    def read(path: Path) -> Any:
        return json.loads(path.read_text(encoding="utf-8-sig"))

    run_options: dict[str, Any] = {}
    benchmark = read(args.benchmark)
    if args.run_config:
        from scripts.main.common import load_config

        runtime = load_config(args.run_config, [])
        run_options = {
            "token_budget": runtime["logprob_token_budget"],
            "sft_microbatch_size": runtime.get(
                "sft_microbatch_size", runtime["batch_sizes"]["sft"]
            ),
            "sft_steps": runtime["max_optimizer_steps"]["sft"],
            "rl_steps": runtime["max_optimizer_steps"]["rl"],
            "iterations": runtime["num_iterations"],
        }
    sft = benchmark.get("sft_probe")
    if not sft and not args.run_config and args.sft_probe.is_file():
        sft = read(args.sft_probe)
    if sft and sft.get("status") == "oom":
        sft = None
    result = estimate(
        read(args.config),
        benchmark,
        read(args.worst),
        read(args.offers),
        sft_probe=sft,
        bf16_base_parameters=args.bf16_base_parameters,
        **run_options,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "memory": result["memory"],
                "flops": result["flops"],
                "plans": result["plans"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
