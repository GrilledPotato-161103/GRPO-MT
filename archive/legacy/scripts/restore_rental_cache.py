"""Restore pinned public caches and reproduce the exact saved SFT reference base."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-only", action="store_true", help="Prepare public caches for a fresh run without restoring old weights.")
    args = parser.parse_args()
    root = Path("logs/production")
    manifest = json.loads((root / "cache_manifest.json").read_text())
    for snapshot in manifest["hf_snapshots"]:
        model_id = snapshot["cache_name"].removeprefix("models--").replace("--", "/")
        snapshot_download(
            model_id,
            revision=snapshot["revision"],
            allow_patterns=list(snapshot["files"]),
        )
        # COMET requests its encoder/tokenizer at "main". Commit-only downloads
        # do not recreate that alias, which offline Hub resolution requires.
        reference = Path(os.environ["HF_HOME"]) / "hub" / snapshot["cache_name"] / "refs/main"
        if not reference.exists():
            reference.parent.mkdir(parents=True, exist_ok=True)
            reference.write_text(snapshot["revision"])
    if args.cache_only:
        print("Pinned public model/tokenizer caches ready for fresh training.", flush=True)
        return
    # Offline mode is selected before importing Transformers/COMET in the child.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch

    from src.models.sft_module import export_sft

    torch.set_num_threads(8)

    destination = Path("exports/phomt_sft")
    if not destination.exists():
        checkpoint = Path(
            json.loads((root / "sft/result.json").read_text())["checkpoint"]
        )
        export_sft(checkpoint, destination)
    expected = json.loads((root / "sft_export_manifest.json").read_text())
    for name, record in expected.items():
        path = destination / name
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if path.stat().st_size != record["size"] or actual != record["sha256"]:
            raise ValueError(f"Restored SFT base differs from original: {name}")
    (root / "RESTORED_BASE_VERIFIED").write_text(
        "All original SFT export hashes match.\n"
    )
    print("Pinned public caches restored; exact SFT base verified.", flush=True)


if __name__ == "__main__":
    main()
