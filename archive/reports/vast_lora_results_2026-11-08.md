# Qwen 7B LoRA / QLoRA RTX 4090 verification

The test ran across midnight on 7–8 November 2026 in Asia/Saigon (7 November UTC), using instance **59599517** and `Qwen/Qwen2.5-7B-Instruct`. Only source code and synthetic examples were uploaded; no PhoMT CSVs or API credentials were transferred. Feedback was not called.

All three test instances were destroyed. At **17:10:40 UTC / 00:10:40 Asia/Saigon**, the official CLI reported **zero active instances**. The cleanup watchdog was marked finished, task SSH registration 1817421 was removed, and its local private/public key files were deleted. The pre-existing user SSH key was retained.

## Results

| Check | Result |
|---|---|
| Regression suite | **63 passed**, **95.14%** configured framework coverage |
| BF16 LoRA | Fits; 15.338 GiB training peak allocated, 15.643 GiB reserved |
| NF4 QLoRA | Fits; 15.005 GiB training peak allocated, 15.377 GiB reserved |
| Trainable parameters | 5,046,272 in each mode |
| Independent reference parameters | Zero; frozen base scored with adapters disabled |
| Optimizer steps | Two in each mode; adapter changes verified |
| GPU | NVIDIA GeForce RTX 4090; PyTorch reports 23.648 GiB total |

Both profiles used BF16 mixed precision, rank 8 / alpha 16 attention adapters, gradient checkpointing, log-probability microbatches of one, 512-token prompts and 64-token completions. The synthetic joint batch contained two proposer candidates and two solver sources with two candidates each. Generation separately used a batch of four forced to produce 64 tokens. The joint teacher-forced training timer was **8.510 seconds for BF16 LoRA** and **6.996 seconds for NF4 QLoRA**, including rollout scoring and two optimizer updates. These timers exclude model loading, the separate generation probe, feedback requests, and validation, so they are not end-to-end self-play throughput estimates.

Loaded-model allocation was 14.246 / 7.227 GiB and generation peak was 14.642 / 8.001 GiB for LoRA / QLoRA respectively. NF4 therefore saved substantial resident memory, while the measured joint-training peaks were close. The memory result applies to the tested shapes and implementation; longer outputs, more tasks/candidates, different adapter targets, or a resident judge require another measurement. No translation-quality gain was tested.

## Environment and host recovery

The host used Python 3.12.15, PyTorch 2.14.1+cu126, CUDA runtime 12.6, Transformers 4.57.6, PEFT 0.18.1, bitsandbytes 0.50.2 and Lightning 2.6.6. Initial CUDA initialization failed with error 804 because the container selected an incompatible forward-compatibility driver library. Preloading the host's native `/usr/lib/x86_64-linux-gnu/libcuda.so.1` resolved initialization and the full suite passed. This override was confined to the test process environment; it is not a project-wide CUDA setting. NVIDIA documents the relevant [minor-version compatibility limits](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).

Two earlier test instances, 59598126 and 59598627, were destroyed after their logs identified incorrect permissions on `/root/.ssh/authorized_keys`. No test bundle reached those hosts. The successful final instance used an explicit startup permission repair. An unavailable offer returned no instance and incurred no running-instance usage.

## Cost evidence

The accepted final offer was **$0.434963/hour including 40 GB storage**, with quoted transfer charges of $0.00260417/GB each way. Account credit before this verification was $4.575644904419974 and the post-cleanup snapshot was $4.347996857659979: an observed decrease of **$0.227648**, including the unsuccessful host starts. This is a balance observation, not an itemized final invoice; settlements can lag. The itemized-billing CLI query remains deferred per the user's earlier instruction and was not retried.

## Evidence

Downloaded artifacts are under `logs/lora_vast_59599517/lora-results/`, including hardware metadata, passing pytest output and coverage, the retained initial CUDA-failure log, and both `memory_profile.json` files. The archive is `logs/lora_vast_results.tar.gz`. These generated files are ignored by Git.

The final local source also passes 63 tests with 95.14% coverage, strict mypy for 19 files, and Ruff. Local verification used the requested existing conda environment at `E:/IAI/Lab/Research/venv`, including a real tiny-Qwen NF4 training/resume test on the RTX 4060 Laptop GPU.
