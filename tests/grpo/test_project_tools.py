"""Source packaging and cleanup keep user data, credentials and results intact."""

import json
import tarfile
from pathlib import Path

import pytest

from scripts.main.common import (
    DEFAULT_CONFIG,
    check_feedback,
    check_logging,
    ensure_comet,
    load_config,
)
from scripts.main.cycles import main as cycles
from scripts.main.pipeline import cache_assets, file_record, write_state
from scripts.tools.clean import clean
from scripts.tools.package import package
from scripts.verify.pipeline import synthetic_config


def test_cleanup_removes_only_validated_disposable_paths(tmp_path: Path) -> None:
    disposable = tmp_path / "src/module/__pycache__"
    disposable.mkdir(parents=True)
    (disposable / "module.pyc").write_bytes(b"cache")
    (tmp_path / ".pytest_cache").mkdir()
    for directory in ("data", "logs", "archive"):
        protected = tmp_path / directory / "__pycache__"
        protected.mkdir(parents=True)
        (protected / "preserved").write_bytes(b"user artifact")
    assert disposable in clean(tmp_path)
    assert disposable.exists()
    clean(tmp_path, apply=True)
    assert not disposable.exists() and not (tmp_path / ".pytest_cache").exists()
    for directory in ("data", "logs", "archive"):
        assert (
            tmp_path / directory / "__pycache__/preserved"
        ).read_bytes() == b"user artifact"


def test_source_bundle_contains_execution_files_without_user_artifacts(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "source.tar.gz"
    package(destination)
    with tarfile.open(destination) as archive:
        names = set(archive.getnames())
    assert {
        "configs/runs/a100.yaml",
        "scripts/rental/start.sh",
        "docs/execution.md",
        ".env.example",
    } <= names
    assert all(
        not name.startswith(("logs/", "archive/", "data/full", ".git/"))
        for name in names
    )
    assert not any(Path(name).name in {".env", "id_ed25519"} for name in names)
    assert not any(name.endswith((".ckpt", ".safetensors", ".pyc")) for name in names)
    assert "data/example_train.jsonl" in names


def test_synthetic_probe_bounds_both_stages_and_preserves_production_profile(
    tmp_path: Path,
) -> None:
    before = DEFAULT_CONFIG.read_bytes()
    config = synthetic_config(DEFAULT_CONFIG, tmp_path / "probe", 5)
    cfg = load_config(config, [])
    assert "max_optimizer_steps" not in cfg and cfg["trainer"]["max_steps"] == 5
    assert cfg["sac_reward"]["fusion"]["total_steps"] == 5
    assert cfg["paired_batches"] and cfg["data_loader"]["num_workers"] == 12
    manifest = json.loads(Path(cfg["dataset_manifest"]).read_text())
    assert manifest["counts"]["sft"] == 160 and manifest["counts"]["rl"] == 40
    assert Path(cfg["output_dir"]).is_relative_to(tmp_path)
    assert DEFAULT_CONFIG.read_bytes() == before
    with pytest.raises(FileExistsError):
        synthetic_config(DEFAULT_CONFIG, tmp_path / "probe", 5)


@pytest.mark.parametrize("local_model", [False, True])
def test_asset_preflight_rejects_changed_data_before_downloading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, local_model: bool
) -> None:
    """A copied dataset must match its manifest before expensive model loading."""
    import huggingface_hub

    from src.models.components import comet_reward

    run = tmp_path / "run"
    run.mkdir()
    files = {
        name: tmp_path / f"{name}.csv" for name in ("sft", "rl", "validation", "test")
    }
    for name, path in files.items():
        path.write_text(f"en,vi\n{name} source,{name} target\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {"outputs": {name: file_record(path) for name, path in files.items()}}
        )
    )
    hf = tmp_path / "hf"
    snapshot = hf / "hub/models--unit--model/snapshots/pinned-revision"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    downloads: list[object] = []

    def download(*args: object, **kwargs: object) -> str:
        downloads.append(kwargs)
        return str(snapshot)

    class Comet:
        def __init__(self, *args: object, **kwargs: object) -> None:
            assert kwargs["device"] == "cpu"

        def __call__(self, *args: object) -> list[float]:
            return [0.8]

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    monkeypatch.setattr(comet_reward, "CometScorer", Comet)
    monkeypatch.setenv("HF_HOME", str(hf))
    monkeypatch.setenv("COMET_CHECKPOINT", str(tmp_path / "comet.ckpt"))
    (run / "rental_config.json").write_text(
        json.dumps(
            {
                "dataset_manifest": str(manifest),
                "data_files": {name: str(path) for name, path in files.items()},
                "model_name": str(snapshot) if local_model else "unit/model",
                "policy_revision": "pinned-revision",
                "reward": {"comet_weight": 1},
            }
        )
    )
    assert cache_assets(run) == "pinned-revision"
    cached = json.loads((run / "cache_manifest.json").read_text())
    assert cached["cached_before_training"] and cached["comet_probe"] == [0.8]
    assert cached["hf_snapshots"][0]["files"]["config.json"] == file_record(
        snapshot / "config.json"
    )
    expected_downloads = 0 if local_model else 1
    assert len(downloads) == expected_downloads
    files["rl"].write_text("changed during transfer", encoding="utf-8")
    with pytest.raises(ValueError, match="Dataset hash mismatch: rl"):
        cache_assets(run)
    assert len(downloads) == expected_downloads
    write_state(run, "verified", stage="preflight")
    assert json.loads((run / "state.json").read_text())["stage"] == "preflight"


def test_gated_download_uses_environment_token_and_reuses_cached_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import huggingface_hub

    from src.utils import credentials

    target = tmp_path / "comet/checkpoints/model.ckpt"
    cfg = load_config(DEFAULT_CONFIG, [f"comet_checkpoint={target.as_posix()}"])
    monkeypatch.setattr(credentials, "environment_secret", lambda name: None)
    with pytest.raises(ValueError, match="HUGGING_FACE_API_KEY"):
        ensure_comet(cfg)
    token = "synthetic-test-token"
    monkeypatch.setattr(credentials, "environment_secret", lambda name: token)
    calls: list[object] = []

    def download(model: str, **kwargs: object) -> None:
        assert model == "Unbabel/wmt22-cometkiwi-da"
        assert kwargs["token"] == token
        assert kwargs["revision"] == cfg["comet_revision"]
        assert kwargs["allow_patterns"] == ["checkpoints/model.ckpt", "hparams.yaml"]
        calls.append(model)
        target.parent.mkdir(parents=True)
        target.write_bytes(b"synthetic weights")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    ensure_comet(cfg)
    ensure_comet(cfg)
    assert len(calls) == 1 and token not in json.dumps(cfg)
    target.unlink()
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda *args, **kwargs: None
    )
    with pytest.raises(ValueError, match="COMET checkpoint path"):
        ensure_comet(cfg)
    cfg["reward"]["comet_weight"] = 0
    ensure_comet(cfg)


def test_profiles_reject_recursive_or_nonmapping_inheritance(tmp_path: Path) -> None:
    recursive = tmp_path / "recursive.yaml"
    recursive.write_text("base_config: recursive.yaml\n")
    with pytest.raises(ValueError, match="Recursive"):
        load_config(recursive, [])
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("[]\n")
    with pytest.raises(TypeError, match="mapping"):
        load_config(invalid, [])


def test_service_preflight_fails_before_training_and_keeps_keys_out_of_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.utils import credentials

    cfg = load_config(DEFAULT_CONFIG, [])
    monkeypatch.setattr(credentials, "environment_secret", lambda name: None)
    with pytest.raises(ValueError, match="WANDB_API_KEY"):
        check_logging(cfg)
    with pytest.raises(ValueError, match="ANTHROPIC_AUTH_TOKEN"):
        check_feedback(cfg)
    token = "synthetic-test-token"
    monkeypatch.setattr(credentials, "environment_secret", lambda name: token)
    monkeypatch.setenv("WANDB_API_KEY", "previous")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "previous")
    check_logging(cfg)
    check_feedback(cfg)
    assert token not in json.dumps(cfg)


def test_cyclic_cli_preview_needs_no_checkpoint_access(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "preview-only"
    args = ["--set", f"output_dir={output.as_posix()}"]
    assert cycles([*args, "--dry-run"]) == 0
    assert "first_cycle" in json.loads(capsys.readouterr().out)
    assert not output.exists()
    assert cycles(args) == 2
    assert "trust-checkpoint" in capsys.readouterr().err


@pytest.mark.parametrize(
    "failure",
    ["data", "manifest", "resume", "rl_base", "existing_result", "existing_export"],
)
def test_launcher_rejects_incomplete_or_existing_runs_before_model_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    from omegaconf import OmegaConf

    from scripts.main import common
    from scripts.main.train import main as train

    cfg = load_config(DEFAULT_CONFIG, [])
    cfg["output_dir"] = str(tmp_path / "run")
    cfg["sft_export_dir"] = str(tmp_path / "sft_export")
    cfg["data_files"] = {
        name: str(tmp_path / f"{name}.csv")
        for name in ("sft", "rl", "validation", "test")
    }
    for path in cfg["data_files"].values():
        Path(path).write_text("en,vi\nsource,target\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    cfg["dataset_manifest"] = str(manifest)
    args = ["--stage", "rl" if failure == "rl_base" else "sft"]
    if failure == "data":
        Path(cfg["data_files"]["rl"]).unlink()
    elif failure == "manifest":
        manifest.unlink()
    elif failure == "resume":
        args += ["--resume", str(tmp_path / "missing.ckpt"), "--trust-checkpoint"]
    elif failure == "existing_result":
        previous = tmp_path / "run/sft/result.json"
        previous.parent.mkdir(parents=True)
        previous.write_text("{}")
    elif failure == "existing_export":
        Path(cfg["sft_export_dir"]).mkdir()
    config = tmp_path / "run.yaml"
    OmegaConf.save(OmegaConf.create(cfg), config)

    def unexpected(*args: object, **kwargs: object) -> None:
        pytest.fail("Invalid launch reached model preparation")

    monkeypatch.setattr(common, "compose_stage", unexpected)
    monkeypatch.setattr(common, "ensure_comet", unexpected)
    assert train(["--config", str(config), *args]) == 2
    assert not (tmp_path / "run/.launcher.lock").exists()
