"""Prepare exactly 10,000 whole NLLB pairs from the grouped original corpus."""

import csv
import hashlib
import json
from pathlib import Path


def main() -> None:
    from transformers import AutoTokenizer

    from scripts.data.filter_token_pairs import prepare
    from scripts.data.prepare_full import file_hash

    root = Path(__file__).resolve().parents[2]
    base = root / "data/nllb_10k_fp16_20261116"
    revision = json.loads(
        (root / "runs/nllb-10k-fp16-20261116/policy_revision.json").read_text()
    )
    tokenizer = AutoTokenizer.from_pretrained(base / "tokenizer", local_files_only=True)
    bounded = base / "bounded"
    manifest = prepare(
        base / "grouped",
        root / "data/reference_audit/test.csv",
        bounded,
        tokenizer,
        revision["model"],
        revision["revision"],
        content_limit=256,
        prompt_template="{source}",
        language_codes={"English": "eng_Latn", "Vietnamese": "vie_Latn"},
    )
    pool = bounded / "sft.csv"
    with pool.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(
        key=lambda r: hashlib.sha256(
            ("20261116:sft:" + r["en"] + "\0" + r["vi"]).encode()
        ).digest()
    )
    if len(rows) < 10000:
        raise RuntimeError("Too few complete candidate pairs for the requested sample")
    pool.rename(bounded / "sft_pool.csv")
    with pool.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["en", "vi"])
        writer.writeheader()
        writer.writerows({k: r[k] for k in ("en", "vi")} for r in rows[:10000])
    manifest["sft_sampling"] = {
        "seed": 20261116,
        "population_pairs": len(rows),
        "selected_pairs": 10000,
        "method": "SHA256 seeded permutation of complete pairs in the independent SFT group pool",
        "one_bidirectional_epoch_directional_records": 20000,
        "excluded_sft_pool_pairs_never_enter_rl": True,
    }
    manifest["outputs"]["sft"].update(path=str(pool.resolve()), sha256=file_hash(pool))
    manifest["counts"].update(
        sft=10000, sft_pool=len(rows), training_pool=10000 + manifest["counts"]["rl"]
    )
    manifest["actual_sft_fraction"] = 10000 / manifest["counts"]["training_pool"]
    # A deterministic complete development panel, not an ordered prefix.
    with (bounded / "validation.csv").open(encoding="utf-8", newline="") as handle:
        validation = list(csv.DictReader(handle))
    validation.sort(
        key=lambda r: hashlib.sha256(
            ("20261116:validation:" + r["en"] + "\0" + r["vi"]).encode()
        ).digest()
    )
    panel = bounded / "validation_eval.csv"
    with panel.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["en", "vi"])
        writer.writeheader()
        writer.writerows({k: r[k] for k in ("en", "vi")} for r in validation[:512])
    manifest["outputs"]["validation_eval"] = {
        "path": str(panel.resolve()),
        "sha256": file_hash(panel),
        "pairs": min(512, len(validation)),
    }
    # Explicitly verify the selected view's normalized language exclusion.
    from scripts.data.prepare_stages import _normalize

    for language in ("en", "vi"):
        selected = {_normalize(r[language]) for r in rows[:10000]}
        for stage in ("rl", "validation", "test"):
            path = Path(manifest["outputs"][stage]["path"])
            with path.open(encoding="utf-8", newline="") as handle:
                if any(
                    _normalize(r[language]) in selected for r in csv.DictReader(handle)
                ):
                    raise RuntimeError(f"Selected SFT overlap with {stage}/{language}")
    manifest["selected_sft_normalized_overlap_verified"] = 0
    (bounded / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (root / "runs/nllb-10k-fp16-20261116/dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "sft_pairs": 10000,
                "validation_panel_pairs": min(512, len(validation)),
                "sft_sha256": file_hash(pool),
                "output": str(bounded),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
