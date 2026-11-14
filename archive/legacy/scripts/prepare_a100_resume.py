"""Rebatch a clean GRPO interruption without changing weights or optimizer state."""

import json
from pathlib import Path

from scripts.rebatch_checkpoint import migrate


def main() -> None:
    root = Path("logs/production").resolve()
    cfg = json.loads((root / "rental_config.json").read_text())
    if cfg["start_stage"] != "rl":
        raise ValueError("This continuation requires completed SFT and interrupted GRPO")
    if not (root / "RESTORED_BASE_VERIFIED").exists():
        raise RuntimeError("Restore and verify the exact SFT reference base first")
    destination = root / "rl/checkpoints/a100_resume.ckpt"
    metadata = root / "rl_resume.json"
    if metadata.exists():
        raise RuntimeError("Resume preparation already exists; refusing to overwrite")
    source = root / "rl/checkpoints/interrupted.ckpt"
    prior = cfg.get("previous_rl_examples", 0)
    consumed = migrate(source, destination, original_batch_size=cfg["previous_rl_batch_size"], previous_examples=prior, total_examples=1900)
    metadata.write_text(json.dumps({"checkpoint": str(destination), "consumed_examples": consumed}, indent=2))
    print(f"GRPO Adam state preserved; {consumed}/1900 directional examples already consumed.")


if __name__ == "__main__":
    main()
