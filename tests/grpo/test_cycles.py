"""Sample provenance, alternating objectives, fixed-reference gradients and resume."""

import copy
import json
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import lightning as L
import pytest
import torch

from scripts.main.cycle_phase import run_phase
from scripts.main.cycles import execute, load_settings, phase_plan, sample_pairs, sha256
from scripts.verify.assets import create_assets
from src.data.translation_datamodule import TranslationDataModule
from src.models.components.optim import TransferAdamW
from src.models.cyclic_module import CyclicTranslationModule
from src.models.sft_module import TranslationSFTModule


def test_sampling_is_deterministic_balanced_and_does_not_edit_full_splits(
    tmp_path: Path,
) -> None:
    source = tmp_path / "full.csv"
    source.write_text(
        "en,vi\n" + "\n".join(f"source {i},target {i}" for i in range(30)),
        encoding="utf-8",
    )
    before = source.read_bytes()
    a = sample_pairs(source, tmp_path / "a.csv", 7, 42)
    b = sample_pairs(source, tmp_path / "b.csv", 7, 42)
    c = sample_pairs(source, tmp_path / "c.csv", 7, 43)
    assert a["indices"] == b["indices"] != c["indices"]
    assert len(set(a["indices"])) == 7 and a["source_pairs"] == 30
    assert source.read_bytes() == before
    chunks = [
        sample_pairs(source, tmp_path / f"chunk_{i}.csv", 10, 42, offset=10 * i)
        for i in range(3)
    ]
    assert set().union(*(set(chunk["indices"]) for chunk in chunks)) == set(range(30))
    data = TranslationDataModule(
        train_file=str(tmp_path / "a.csv"),
        val_file=str(tmp_path / "a.csv"),
        bidirectional=True,
    )
    data.setup("fit")
    directions = [row["direction"] for row in data.splits["train"]]
    assert directions.count("en_vi") == directions.count("vi_en") == 7
    with pytest.raises(ValueError, match="overwrite"):
        sample_pairs(source, tmp_path / "a.csv", 7, 42)
    with pytest.raises(ValueError, match="Requested"):
        sample_pairs(source, tmp_path / "large.csv", 31, 42)


def test_phase_counts_and_validation_plan() -> None:
    settings, cfg = load_settings(Path("configs/runs/cycles.yaml"), [])
    cfg["run_group"] = "test-cycles"
    a = phase_plan(settings, cfg, 0, "sft", "initial", "initial", 24393, 24393)
    b = phase_plan(settings, cfg, 0, "rl", "after-sft", None, 0, 25393)
    assert (a["batches"], a["updates"], a["train_pairs"]) == (1000, 1000, 12000)
    assert (b["batches"], b["updates"], b["train_pairs"]) == (100, 200, 800)
    assert a["validation_file"] == b["validation_file"]
    assert a["evaluate"] and b["evaluate"]
    with pytest.raises(ValueError, match="divisible"):
        load_settings(Path("configs/runs/cycles.yaml"), ["rl_updates=201"])
    with pytest.raises(ValueError, match="credentials"):
        load_settings(Path("configs/runs/cycles.yaml"), ["training.token=bad"])


def test_two_cycles_preserve_anchor_weights_optimizer_moments_and_update_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("peft")
    torch.set_num_threads(1)
    L.seed_everything(42)
    asset = create_assets(tmp_path / "model", "seq2seq")
    data_file = Path("data/example_train.jsonl").resolve()
    initial = TranslationSFTModule(
        model_name_or_path=str(asset),
        base_revision="main",
        reward=None,
        max_source_length=32,
        max_target_length=4,
        max_new_tokens=4,
        lora={"enabled": True, "r": 2, "alpha": 4, "target_modules": ["q", "v"]},
        optimizer=partial(
            TransferAdamW, lr=1e-4, l2sp_weight=1e-3, max_relative_drift=0.1
        ),
    )
    data = TranslationDataModule(
        train_file=str(data_file), val_file=str(data_file), batch_size=2
    )
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=1,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path / "initial",
    )
    trainer.fit(initial, data)
    anchor = tmp_path / "initial.ckpt"
    trainer.save_checkpoint(anchor)
    settings, cfg = load_settings(
        Path("configs/runs/cycles.yaml"),
        [
            f"output_dir={tmp_path / 'cycles'}",
            f"initial_checkpoint={anchor}",
            "sft_batches=2",
            "rl_updates=2",
            "validation_pairs=2",
            "cycles=2",
            "training.wandb.enabled=false",
            "training.data_loader.num_workers=0",
            "training.data_loader.persistent_workers=false",
            "training.length_batching=false",
            "training.reward.comet_weight=0.0",
            "training.reward.bleu_weight=1.0",
            "training.batch_sizes.sft=2",
            "training.batch_sizes.rl=2",
            "training.trainer.accelerator=cpu",
            "training.trainer.precision=32-true",
            "training.source_cap=32",
            "training.target_cap=4",
            "training.group_size=2",
            "training.load_dtype=float32",
            "training.lora.r=2",
            "training.lora.alpha=4",
            "training.lora.target_modules=[q,v]",
            "training.warmup_steps.sft=0",
            "training.warmup_steps.rl=0",
            "training.lora.warmup_steps=0",
        ],
    )
    cfg.update(model_name=str(asset), policy_revision="main", run_group="tiny-cycles")
    sample_pairs(data_file, Path(settings["output_dir"]) / "validation.csv", 2, 42)
    policy, total = str(anchor), 1
    optimizers = {"sft": str(anchor), "rl": None}
    offsets = {"sft": 1, "rl": 0}
    frozen_before = {
        name: p.detach().clone()
        for name, p in initial.policy.named_parameters()
        if not p.requires_grad
    }
    original_step = CyclicTranslationModule._optimizer_step
    first_stop = Path(settings["output_dir"]) / "STOP"
    for cycle in range(2):
        for stage in ("sft", "rl"):
            plan = phase_plan(
                settings,
                cfg,
                cycle,
                stage,
                policy,
                optimizers[stage],
                offsets[stage],
                total,
            )
            sample_pairs(
                data_file, Path(plan["train_file"]), plan["train_pairs"], plan["seed"]
            )
            if cycle == 0 and stage == "sft":

                def stop_after_first_update(
                    module: CyclicTranslationModule, optimizer: object
                ) -> None:
                    original_step(module, optimizer)
                    first_stop.touch()

                with monkeypatch.context() as patch:
                    patch.setattr(
                        CyclicTranslationModule,
                        "_optimizer_step",
                        stop_after_first_update,
                    )
                    assert run_phase(plan) == 75
                Path(plan["stop_file"]).unlink()
            assert run_phase(plan) == 0
            record = json.loads((Path(plan["output"]) / "result.json").read_text())
            assert record["updates"] == 2
            assert record["evaluation"]["counts"] == {"en_vi": 2, "vi_en": 2}
            assert record["evaluation"]["global_step"] == total + 2
            checkpoint = torch.load(record["checkpoint"], weights_only=False)
            assert checkpoint["cycle_state"]["stage_updates"] == offsets[stage] + 2
            step = next(iter(checkpoint["optimizer_states"][0]["state"].values()))[
                "step"
            ]
            assert int(step) == offsets[stage] + 2
            policy, optimizers[stage] = record["checkpoint"], record["checkpoint"]
            offsets[stage], total = record["stage_updates"], record["total_updates"]
    assert run_phase(plan) == 0
    module = CyclicTranslationModule.load_from_checkpoint(
        policy, weights_only=False, reward=None
    )
    assert all(
        torch.equal(value, dict(module.policy.named_parameters())[name])
        for name, value in frozen_before.items()
    )
    sample = module._sample_prompts(["Hello"], 1)
    reference_before = sample["reference_logps"].clone()
    with torch.no_grad():
        for name, parameter in module.policy.named_parameters():
            if parameter.requires_grad:
                parameter.add_(0.01)
    assert torch.equal(
        reference_before,
        module._reference_logps(sample["inputs"], sample["sequences"], sample["mask"]),
    )
    assert all(
        not p.requires_grad
        for name, p in module.policy.named_parameters()
        if ".cycle_reference." in name
    )
    # Exercise the actual parent orchestration, including durable phase skip on resume.
    parent_settings, parent_cfg = copy.deepcopy(settings), copy.deepcopy(cfg)
    parent_root = tmp_path / "parent"
    parent_settings["output_dir"] = parent_cfg["output_dir"] = str(parent_root)
    files = {}
    for name in ("sft", "rl", "validation", "test"):
        path = tmp_path / f"parent_{name}.csv"
        path.write_text(
            "en,vi\n"
            + "\n".join(f"{name} source {i},{name} target {i}" for i in range(8)),
            encoding="utf-8",
        )
        files[name] = str(path)
    manifest = tmp_path / "parent_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "outputs": {
                    name: {"sha256": sha256(Path(path))} for name, path in files.items()
                }
            }
        )
    )
    parent_cfg.update(data_files=files, dataset_manifest=str(manifest))
    calls = []

    def worker(command: list[str], **kwargs: object) -> SimpleNamespace:
        plan_file = Path(command[-1])
        calls.append(plan_file)
        return SimpleNamespace(returncode=run_phase(json.loads(plan_file.read_text())))

    monkeypatch.setattr("scripts.main.cycles.subprocess.run", worker)
    assert execute(parent_settings, parent_cfg, resume=False) == 0
    assert len(calls) == 4
    assert execute(parent_settings, parent_cfg, resume=True) == 0
    assert len(calls) == 4
    parent_state = json.loads((parent_root / "cycle_state.json").read_text())
    assert parent_state["status"] == "completed"
    assert [record["stage"] for record in parent_state["completed"]] == [
        "sft",
        "rl",
        "sft",
        "rl",
    ]
    for stage in ("sft", "rl"):
        a = json.loads(
            (parent_root / "cycle_001" / stage / "sample.sampling.json").read_text()
        )
        b = json.loads(
            (parent_root / "cycle_002" / stage / "sample.sampling.json").read_text()
        )
        assert not set(a["indices"]) & set(b["indices"])
