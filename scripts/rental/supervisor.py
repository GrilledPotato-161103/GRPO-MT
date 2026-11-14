"""Supervise one authorized rental using official Vast CLI, SSH, and local backups."""

import argparse
import hashlib
import json
import os
import subprocess
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


def command(args: list[str], timeout: int = 90) -> str:
    child_environment = os.environ.copy()
    for secret in ("WANDB_API_KEY", "HUGGING_FACE_API_KEY", "CLAUDE_AUTH_KEY"):
        child_environment.pop(secret, None)
    result = subprocess.run(
        args,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env=child_environment,
    )
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): {args[0]}")
    return result.stdout


class Supervisor:
    def __init__(self, path: Path):
        self.path = path.resolve()
        self.root = self.path.parent
        self.state = json.loads(self.path.read_text(encoding="utf-8-sig"))
        self.instance = str(self.state["instance_id"])
        self.ssh = [
            "ssh",
            "-i",
            self.state["key"],
            "-p",
            str(self.state["port"]),
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=20",
            "-o",
            f"UserKnownHostsFile={self.state['known_hosts']}",
            self.state["host"],
        ]
        self.scp = [
            "scp",
            "-i",
            self.state["key"],
            "-P",
            str(self.state["port"]),
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=20",
            "-o",
            f"UserKnownHostsFile={self.state['known_hosts']}",
        ]

    def record(self, **fields: Any) -> None:
        with (self.root / "supervisor.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"unix_time": time.time(), **fields}) + "\n")

    def remote(self, script: str, timeout: int = 90) -> str:
        try:
            return command([*self.ssh, script], timeout)
        except Exception:
            # Read official instance diagnostics before any recovery action.
            try:
                diagnostics = command(["vastai", "logs", self.instance, "--raw"])
                (self.root / "connection_diagnostics.json").write_text(
                    diagnostics, encoding="utf-8"
                )
            except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                self.record(diagnostics_error=type(error).__name__)
            raise

    def backup(self) -> None:
        self.remote(
            "cd /workspace/grpo-mt && /venv/main/bin/python -m scripts.rental.snapshot --root logs/production --output /workspace/run-backup.tar.gz",
            180,
        )
        temporary = self.root / "run-backup.download.tar.gz"
        command(
            [
                *self.scp,
                f"{self.state['host']}:/workspace/run-backup.tar.gz",
                str(temporary),
            ],
            600,
        )
        with tarfile.open(temporary) as archive:
            if not archive.getmembers():
                raise RuntimeError("Empty run backup")
            archive.extractall(self.root / "checkpoints_and_logs", filter="data")
        temporary.replace(self.root / "run-backup.tar.gz")
        self.record(backup_verified=True)

    def copy_verified(
        self, remote: str, destination: Path, files: dict[str, Any]
    ) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        for name, expected in files.items():
            path = destination / name

            def matches(path: Path = path, expected: dict[str, Any] = expected) -> bool:
                if not path.is_file() or path.stat().st_size != expected["size"]:
                    return False
                with path.open("rb") as stream:
                    return bool(
                        hashlib.file_digest(stream, "sha256").hexdigest()
                        == expected["sha256"]
                    )

            if not matches():
                command(
                    [*self.scp, f"{self.state['host']}:{remote}/{name}", str(path)],
                    7200,
                )
            if not matches():
                raise RuntimeError(f"Model cache checksum failed: {name}")

    def cache_sft(self) -> None:
        if not self.state.get("cache_sft_base", True):
            return
        if (self.root / "SFT_BASE_CACHED").exists():
            return
        present = self.remote(
            "test ! -f /workspace/grpo-mt/logs/production/SFT_EXPORT_READY || printf READY"
        )
        if "READY" not in present:
            return
        destination = self.root / "cached_sft_base"
        destination.mkdir(exist_ok=True)
        manifest = json.loads(
            (self.root / "checkpoints_and_logs/sft_export_manifest.json").read_text()
        )
        self.copy_verified(
            "/workspace/grpo-mt/exports/phomt_sft", destination, manifest
        )
        (self.root / "SFT_BASE_CACHED").touch()
        self.record(sft_base_cached=True)

    def cache_models(self) -> None:
        """Copy resolved snapshot files without duplicating HF symlink blobs."""
        if (self.root / "MODEL_CACHE_COMPLETE").exists():
            return
        manifest_path = self.root / "checkpoints_and_logs/cache_manifest.json"
        if not manifest_path.exists():
            return
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for snapshot in manifest["hf_snapshots"]:
            cache = self.root / "cached_hf/hub" / snapshot["cache_name"]
            destination = cache / "snapshots" / snapshot["revision"]
            destination.mkdir(parents=True, exist_ok=True)
            self.copy_verified(snapshot["remote_path"], destination, snapshot["files"])
            (cache / "refs").mkdir(exist_ok=True)
            (cache / "refs/main").write_text(snapshot["revision"], encoding="utf-8")
        (self.root / "MODEL_CACHE_COMPLETE").touch()
        self.record(foundation_and_encoder_cached=True)

    def cache_all(self) -> None:
        self.cache_sft()
        if self.state.get("cache_public_models", False):
            self.cache_models()

    def destroy(self, reason: str) -> None:
        if (self.root / "DESTROYED").exists():
            return
        command(["vastai", "destroy", "instance", self.instance, "-y", "--raw"])
        listing = json.loads(
            command(["vastai", "show", "instances-v1", "--limit", "25", "--raw"])
        )
        if any(str(row["id"]) == self.instance for row in listing["instances"]):
            raise RuntimeError("Instance destruction not yet verified")
        (self.root / "DESTROYED").write_text(reason, encoding="utf-8")
        self.record(destroyed_verified=True, reason=reason)
        if self.state.get("ssh_key_managed") is False:
            return
        command(["vastai", "delete", "ssh-key", str(self.state["ssh_key_id"]), "--raw"])
        for suffix in ("", ".pub"):
            key = Path(self.state["key"] + suffix).resolve()
            if key.parent != self.root:
                raise ValueError("Temporary key must remain in the task directory")
            key.unlink(missing_ok=True)

    def run(self, hard_only: bool = False, await_approval: bool = False) -> None:
        executor = ThreadPoolExecutor(max_workers=1)
        cache_job = None
        relay = None
        if not hard_only and not await_approval and self.state.get("wandb_relay", True):
            from scripts.rental.wandb_relay import WandbRelay

            relay = WandbRelay(
                self.root, self.state["run_group"], self.state["wandb_entity"]
            )
        last_backup = 0.0
        last_credit = 0.0
        while not (self.root / "DESTROYED").exists():
            now = time.time()
            if await_approval:
                if (self.root / "DATA_UPLOAD_AUTHORIZED").exists():
                    return
                if now >= self.state["created_unix"] + 1800:
                    self.destroy(
                        "data upload approval pending; avoiding idle rental charges"
                    )
                    return
                time.sleep(10)
                continue
            if now >= self.state["hard_deadline_unix"]:
                self.destroy("hard budget deadline; earlier checkpoints retained")
                return
            if hard_only:
                time.sleep(20)
                continue
            if (self.root / "TUNING_HOLD").exists():
                time.sleep(10)
                continue
            try:
                stop = now >= self.state["graceful_deadline_unix"]
                if now - last_credit > 300:
                    account = json.loads(command(["vastai", "show", "user", "--raw"]))
                    credit = float(account.get("credit", 0))
                    stop |= (
                        credit < 0.6
                        or self.state.get("previous_cost_reserve_usd", 0)
                        + self.state["starting_credit"]
                        - credit
                        >= self.state.get("budget_usd", 10) - 1.0
                    )
                    self.record(credit=credit)
                    last_credit = now
                if stop:
                    self.remote(
                        "mkdir -p /workspace/grpo-mt/logs/production && touch /workspace/grpo-mt/logs/production/STOP"
                    )
                    self.record(graceful_stop_requested=True)
                raw = self.remote(
                    "test ! -f /workspace/grpo-mt/logs/production/state.json || cat /workspace/grpo-mt/logs/production/state.json"
                )
                payload = json.loads(raw) if raw.strip() else {}
                (self.root / "remote_state.json").write_text(
                    json.dumps(payload, indent=2), encoding="utf-8"
                )
                terminal = payload.get("status") in {
                    "completed",
                    "failed",
                    "quota_stopped",
                }
                if now - last_backup > 120 or terminal:
                    self.backup()
                    if relay:
                        relay.sync()
                    if cache_job is not None and cache_job.done():
                        previous, cache_job = cache_job, None
                        previous.result()
                    if cache_job is None:
                        cache_job = executor.submit(self.cache_all)
                    last_backup = time.time()
                if terminal:
                    if cache_job is not None:
                        cache_job.result()
                    self.cache_all()
                    if relay:
                        relay.finish(str(payload["status"]))
                    if self.state.get("destroy_on_terminal", False):
                        self.destroy(str(payload["status"]))
                        return
                    self.record(
                        training_status=payload["status"],
                        instance_preserved=True,
                        budget_guard_only=True,
                    )
                    # Completion is not permission to delete a user's instance.
                    # Continue enforcing the separately authorized budget limit.
                    hard_only = True
            except (
                OSError,
                ValueError,
                RuntimeError,
                tarfile.TarError,
                subprocess.SubprocessError,
            ) as error:
                self.record(error_type=type(error).__name__)
            time.sleep(60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--hard-only", action="store_true")
    parser.add_argument("--await-approval", action="store_true")
    args = parser.parse_args()
    Supervisor(args.state).run(args.hard_only, args.await_approval)
