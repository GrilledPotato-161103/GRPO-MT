"""Select complete, bidirectionally token-bounded pairs from existing disjoint splits."""

import argparse
import csv
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, cast

from scripts.data.prepare_full import file_hash
from src.data.components.multilingual_tokens import multilingual_ids
from src.data.components.translation_prompt import (
    PRECISE_TRANSLATION_PROMPT,
    prompt_overhead,
    translation_prompts,
)


def reuse_manifest(
    source: Path, output: Path, tokenizer: Any, model_name: str, revision: str
) -> dict[str, Any]:
    """Reuse bounded CSVs only after exact native-tokenizer and file-hash checks."""
    manifest: dict[str, Any] = json.loads(source.read_text(encoding="utf-8"))
    budget = manifest["token_budget"]
    codes = {"English": "eng_Latn", "Vietnamese": "vie_Latn"}
    if (
        budget.get("language_codes") != codes
        or budget.get("prompt_template") != "{source}"
        or budget.get("use_chat_template")
    ):
        raise ValueError("Only native NLLB manifests can be reused")
    multilingual_ids(
        tokenizer, ["", ""], ["English", "Vietnamese"], ["Vietnamese", "English"], codes
    )
    fingerprint = hashlib.sha256(
        tokenizer.backend_tokenizer.to_str().encode()
    ).hexdigest()
    if fingerprint != budget["tokenizer_state_sha256"]:
        raise ValueError("Tokenizers differ; re-filter complete pairs")
    for name, record in manifest["outputs"].items():
        if file_hash(Path(record["path"])) != record["sha256"]:
            raise ValueError(f"Reused {name} CSV hash differs from its manifest")
    manifest["compatibility_reuse"] = {
        "source_manifest": str(source.resolve()),
        "sha256": file_hash(source),
        "verified_tokenizer_state_sha256": fingerprint,
    }
    budget.update(model_name=model_name, tokenizer_revision=revision)
    output.mkdir(parents=True, exist_ok=False)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def prepare(
    input_dir: Path,
    test_file: Path,
    output: Path,
    tokenizer: Any,
    model_name: str,
    revision: str,
    content_limit: int = 256,
    batch_size: int = 512,
    prompt_template: str = PRECISE_TRANSLATION_PROMPT,
    language_codes: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Retain whole pairs and stage membership; never truncate either language.

    Reuse the original grouped SFT/RL/validation split to preserve leakage controls.
    The official test stays intact; a separately named bounded subset is produced
    and its coverage is recorded rather than presented as full-test performance.
    """
    if content_limit < 1 or batch_size < 1:
        raise ValueError("Content limit and batch size must be positive")
    manifest_path = input_dir / "manifest.json"
    original = json.loads(manifest_path.read_text(encoding="utf-8"))
    if original.get("normalized_en_vi_overlap") != 0:
        raise ValueError("Use an existing grouped, leakage-checked stage manifest")
    paths = {name: input_dir / f"{name}.csv" for name in ("sft", "rl", "validation")}
    paths["test"] = test_file
    for name in ("sft", "rl", "validation"):
        expected = original["outputs"][name]["sha256"]
        if file_hash(paths[name]) != expected:
            raise ValueError(f"Original {name} split does not match its manifest")
    if file_hash(test_file) != original["inputs"]["test"]["sha256"]:
        raise ValueError("Official test does not match the original split manifest")
    options = None if language_codes else {"enable_thinking": False}
    if language_codes:
        if prompt_template != "{source}":
            raise ValueError("NLLB filtering uses raw {source} and language controls")
        overhead = max(
            map(
                len,
                multilingual_ids(
                    tokenizer,
                    ["", ""],
                    ["English", "Vietnamese"],
                    ["Vietnamese", "English"],
                    language_codes,
                ),
            )
        )
    else:
        overhead = prompt_overhead(tokenizer, prompt_template, True, options)
    budget: dict[str, Any] = {
        "content_limit": content_limit,
        "prompt_overhead": overhead,
        "prompt_limit": content_limit + overhead,
        "target_limit": content_limit + 1,
        "prompt_template": prompt_template,
        "use_chat_template": not bool(language_codes),
        "chat_template_kwargs": options,
        "model_name": model_name,
        "tokenizer_revision": revision,
        "tokenizer_state_sha256": hashlib.sha256(
            tokenizer.backend_tokenizer.to_str().encode()
        ).hexdigest(),
        "language_codes": language_codes,
        "sft_target_limit": content_limit + 1 + int(bool(language_codes)),
    }
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    stats: dict[str, Any] = {}
    outputs: dict[str, Any] = {}
    for stage, path in paths.items():
        target = output / ("test_bounded.csv" if stage == "test" else f"{stage}.csv")
        count: Counter[str] = Counter(
            {
                key: 0
                for key in (
                    "input_pairs",
                    "retained_pairs",
                    "rejected_pairs",
                    "overlength_en",
                    "overlength_vi",
                    "overlength_complete_prompt",
                )
            }
        )
        sums: Counter[str] = Counter()
        maxima: Counter[str] = Counter()
        histogram: Counter[str] = Counter()
        with path.open(encoding="utf-8-sig", newline="") as source, target.open(
            "x", encoding="utf-8", newline=""
        ) as dest:
            reader = csv.DictReader(source)
            if not reader.fieldnames or not {"en", "vi"}.issubset(reader.fieldnames):
                raise ValueError(f"Expected en/vi CSV columns: {path}")
            writer = csv.DictWriter(dest, fieldnames=["en", "vi"])
            writer.writeheader()

            def consume(
                rows: list[dict[str, str]],
                count: Counter[str] = count,
                writer: Any = writer,
                sums: Counter[str] = sums,
                maxima: Counter[str] = maxima,
                histogram: Counter[str] = histogram,
            ) -> None:
                size = len(rows)
                english = [row["en"] for row in rows]
                vietnamese = [row["vi"] for row in rows]
                forward = translation_prompts(
                    tokenizer,
                    english,
                    ["English"] * size,
                    ["Vietnamese"] * size,
                    prompt_template,
                    not bool(language_codes),
                    options,
                )
                reverse = translation_prompts(
                    tokenizer,
                    vietnamese,
                    ["Vietnamese"] * size,
                    ["English"] * size,
                    prompt_template,
                    not bool(language_codes),
                    options,
                )
                if language_codes:
                    raw = tokenizer(
                        english + vietnamese,
                        add_special_tokens=False,
                        truncation=False,
                        return_attention_mask=False,
                        return_token_type_ids=False,
                    )["input_ids"]
                    complete = multilingual_ids(
                        tokenizer,
                        forward + reverse,
                        ["English"] * size + ["Vietnamese"] * size,
                        ["Vietnamese"] * size + ["English"] * size,
                        language_codes,
                    )
                    ids = raw + complete
                else:
                    ids = tokenizer(
                        english + vietnamese + forward + reverse,
                        add_special_tokens=False,
                        truncation=False,
                        return_attention_mask=False,
                        return_token_type_ids=False,
                    )["input_ids"]
                lengths = [len(value) for value in ids]
                for index, row in enumerate(rows):
                    en, vi, fwd, rev = (
                        lengths[index + offset * size] for offset in range(4)
                    )
                    count["input_pairs"] += 1
                    too_en, too_vi = en > content_limit, vi > content_limit
                    too_prompt = max(fwd, rev) > budget["prompt_limit"]
                    count["overlength_en"] += too_en
                    count["overlength_vi"] += too_vi
                    count["overlength_complete_prompt"] += too_prompt
                    if too_en or too_vi or too_prompt:
                        count["rejected_pairs"] += 1
                        continue
                    writer.writerow({"en": row["en"], "vi": row["vi"]})
                    count["retained_pairs"] += 1
                    for key, value in (
                        ("en_tokens", en),
                        ("vi_tokens", vi),
                        ("prompt_tokens", max(fwd, rev)),
                    ):
                        sums[key] += value
                        maxima[key] = max(maxima[key], value)
                    histogram[str(((max(en, vi) - 1) // 32 + 1) * 32)] += 1

            buffer = []
            for row in reader:
                if any(
                    not isinstance(row.get(key), str) or not row[key].strip()
                    for key in ("en", "vi")
                ):
                    raise ValueError(
                        f"Invalid pair in {stage}; rerun original preparation"
                    )
                buffer.append(row)
                if len(buffer) >= batch_size:
                    consume(buffer)
                    buffer.clear()
            if buffer:
                consume(buffer)
        if not count["retained_pairs"]:
            raise ValueError(f"No complete {stage} pairs fit the token budget")
        stats[stage] = {
            **dict(count),
            "coverage": count["retained_pairs"] / count["input_pairs"],
            "maximum_retained": dict(maxima),
            "mean_retained": {
                key: value / count["retained_pairs"] for key, value in sums.items()
            },
            "retained_max_content_tokens_histogram": dict(histogram),
        }
        outputs[stage] = {"path": str(target.resolve()), "sha256": file_hash(target)}
        print(json.dumps({"stage": stage, **dict(count)}, sort_keys=True), flush=True)
    counts = {name: value["retained_pairs"] for name, value in stats.items()}
    manifest = {
        "version": 3,
        "token_budget": budget,
        "counts": {**counts, "training_pool": counts["sft"] + counts["rl"]},
        "actual_sft_fraction": counts["sft"] / (counts["sft"] + counts["rl"]),
        "filter_statistics": stats,
        "outputs": outputs,
        "original_manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": file_hash(manifest_path),
        },
        "normalized_en_vi_overlap": 0,
        "stage_membership_preserved": True,
        "original_test": {
            "path": str(test_file.resolve()),
            "sha256": file_hash(test_file),
        },
        "official_test_unchanged": True,
        "test_is_length_bounded_subset": True,
        "elapsed_seconds": time.monotonic() - started,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/full_prepared"))
    parser.add_argument(
        "--test-file", type=Path, default=Path("data/full/PhoMT_test.csv")
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--model-name")
    parser.add_argument("--format", choices=("chat", "nllb"), default="chat")
    parser.add_argument("--revision")
    parser.add_argument("--content-limit", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--prompt-template")
    parser.add_argument(
        "--reuse-manifest",
        type=Path,
        help="Reuse native NLLB CSVs after exact tokenizer and SHA-256 verification",
    )
    args = parser.parse_args()
    nllb = args.format == "nllb"
    args.model_name = args.model_name or (
        "facebook/nllb-200-distilled-600M" if nllb else "Qwen/Qwen3-1.7B"
    )
    args.revision = args.revision or (
        "f8d333a098d19b4fd9a8b18f94170487ad3f821d"
        if nllb
        else "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    )
    args.output = args.output or Path(
        "data/token256_nllb" if nllb else "data/token256_qwen3"
    )
    args.prompt_template = args.prompt_template or (
        "{source}" if nllb else PRECISE_TRANSLATION_PROMPT
    )
    from transformers import AutoTokenizer

    tokenizer = cast(Any, AutoTokenizer).from_pretrained(
        args.tokenizer, revision=args.revision
    )
    if args.reuse_manifest:
        report = reuse_manifest(
            args.reuse_manifest, args.output, tokenizer, args.model_name, args.revision
        )
        print(
            json.dumps(
                {
                    "reused": str(args.reuse_manifest),
                    "output": str(args.output),
                    "counts": report["counts"],
                }
            )
        )
        return
    report = prepare(
        args.input,
        args.test_file,
        args.output,
        tokenizer,
        args.model_name,
        args.revision,
        args.content_limit,
        args.batch_size,
        args.prompt_template,
        {"English": "eng_Latn", "Vietnamese": "vie_Latn"} if nllb else None,
    )
    print(
        json.dumps(
            {
                "counts": report["counts"],
                "token_budget": report["token_budget"],
                "elapsed_seconds": report["elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
