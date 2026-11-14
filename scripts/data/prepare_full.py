"""Prepare full PhoMT with grouped splits, deduplication, and test-leakage removal."""

import argparse
import csv
import hashlib
import json
import sqlite3
import time
from array import array
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from scripts.data.prepare_stages import _normalize


def pairs(path: Path) -> Iterator[tuple[str, str] | None]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not {"en", "vi"}.issubset(reader.fieldnames):
            raise ValueError(f"Expected en/vi columns: {path}")
        for row in reader:
            if None in row or any(
                not isinstance(row.get(k), str) or not row[k].strip()
                for k in ("en", "vi")
            ):
                yield None
            else:
                yield row["en"].strip(), row["vi"].strip()


def digest(text: str) -> bytes:
    return hashlib.sha256(_normalize(text).encode("utf-8")).digest()


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _take_groups(groups: list[tuple[int, int]], target: int, reserve: int) -> set[int]:
    chosen: set[int] = set()
    count = 0
    for identifier, size in groups:
        if len(groups) - len(chosen) <= reserve:
            break
        if abs(count + size - target) < abs(count - target):
            chosen.add(identifier)
            count += size
    if not chosen:
        chosen.add(min(groups, key=lambda item: item[1])[0])
    return chosen


def prepare_full(
    train_file: Path,
    test_file: Path,
    output: Path,
    validation_pairs: int = 2000,
    sft_fraction: float = 0.8,
    seed: int = 42,
) -> dict[str, Any]:
    """Keep normalized EN/VI connected components together across all partitions.

    A training component touching either normalized language of the official test
    set is excluded entirely. The official test file is not rewritten or sampled.
    SQLite holds text; compact union-find arrays hold only row relationships.
    """
    if not 0 < sft_fraction < 1 or validation_pairs < 1:
        raise ValueError("Need 0 < sft_fraction < 1 and validation_pairs >= 1")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    database = output / "preparation.sqlite"
    counts: Counter[str] = Counter()
    lengths: dict[str, Counter[str]] = {"en": Counter(), "vi": Counter()}
    connection = sqlite3.connect(database)
    try:
        connection.executescript("""
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            PRAGMA temp_store=FILE;
            PRAGMA cache_size=-65536;
            CREATE TABLE pairs(id INTEGER PRIMARY KEY,en TEXT,vi TEXT,
                               eh BLOB,vh BLOB,ph BLOB UNIQUE);
        """)
        buffer = []
        for pair in pairs(train_file):
            counts["training_input_rows"] += 1
            if pair is None:
                counts["invalid_training_rows"] += 1
                continue
            en, vi = pair
            eh, vh = digest(en), digest(vi)
            buffer.append((en, vi, eh, vh, hashlib.sha256(eh + vh).digest()))
            for language, text in (("en", en), ("vi", vi)):
                boundary = next(
                    (v for v in (64, 128, 256, 512, 1024, 2048) if len(text) <= v), None
                )
                lengths[language][
                    f"chars_le_{boundary}" if boundary else "chars_gt_2048"
                ] += 1
            if len(buffer) == 5000:
                connection.executemany(
                    "INSERT OR IGNORE INTO pairs(en,vi,eh,vh,ph) VALUES(?,?,?,?,?)",
                    buffer,
                )
                buffer.clear()
        connection.executemany(
            "INSERT OR IGNORE INTO pairs(en,vi,eh,vh,ph) VALUES(?,?,?,?,?)", buffer
        )
        connection.commit()
        size = connection.execute("SELECT coalesce(max(id),0) FROM pairs").fetchone()[0]
        counts["unique_training_pairs"] = connection.execute(
            "SELECT count(*) FROM pairs"
        ).fetchone()[0]
        counts["duplicate_training_pairs"] = (
            counts["training_input_rows"]
            - counts["invalid_training_rows"]
            - counts["unique_training_pairs"]
        )
        if size < 3:
            raise ValueError("Need at least three unique training pairs")
        print(
            json.dumps(
                {"phase": "grouping", "unique_pairs": counts["unique_training_pairs"]}
            ),
            flush=True,
        )
        parent = array("I", range(size + 1))
        weight = array("I", [1]) * (size + 1)

        def find(node: int) -> int:
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        def union(left: int, right: int) -> None:
            left, right = find(left), find(right)
            if left != right:
                if weight[left] < weight[right]:
                    left, right = right, left
                parent[right] = left
                weight[left] += weight[right]

        for column in ("eh", "vh"):
            connection.execute(f"CREATE INDEX index_{column} ON pairs({column})")
            previous = None
            first = 0
            for identifier, value in connection.execute(
                f"SELECT id,{column} FROM pairs ORDER BY {column}"
            ):
                if value == previous:
                    union(first, identifier)
                else:
                    first, previous = identifier, value
        forbidden = set()
        for pair in pairs(test_file):
            counts["test_pairs"] += 1
            if pair is None:
                raise ValueError(
                    "Official test contains an invalid pair; it was not modified"
                )
            for column, text in zip(("eh", "vh"), pair, strict=True):
                for (identifier,) in connection.execute(
                    f"SELECT id FROM pairs WHERE {column}=?", (digest(text),)
                ):
                    forbidden.add(find(identifier))
        if not counts["test_pairs"]:
            raise ValueError("Official test is empty")
        groups: dict[int, tuple[int, bytes]] = {}
        for identifier, ph in connection.execute("SELECT id,ph FROM pairs"):
            component = find(identifier)
            if component in forbidden:
                counts["test_overlapping_training_pairs_removed"] += 1
                continue
            n, key = groups.get(component, (0, ph))
            groups[component] = (n + 1, min(key, ph))
        total = sum(n for n, _ in groups.values())
        if len(groups) < 3 or validation_pairs >= total - 1:
            raise ValueError(
                "Not enough independent groups for validation and two training stages"
            )
        ordered = sorted(
            groups,
            key=lambda g: hashlib.sha256(str(seed).encode() + groups[g][1]).digest(),
        )
        validation = _take_groups(
            [(g, groups[g][0]) for g in ordered], validation_pairs, 2
        )
        remaining = [g for g in ordered if g not in validation]
        remaining_count = sum(groups[g][0] for g in remaining)
        sft = _take_groups(
            [(g, groups[g][0]) for g in remaining],
            round(remaining_count * sft_fraction),
            1,
        )
        paths = {name: output / f"{name}.csv" for name in ("sft", "rl", "validation")}
        streams = {
            name: path.open("x", encoding="utf-8", newline="")
            for name, path in paths.items()
        }
        try:
            writers = {name: csv.writer(stream) for name, stream in streams.items()}
            for writer in writers.values():
                writer.writerow(["en", "vi"])
            for identifier, en, vi in connection.execute(
                "SELECT id,en,vi FROM pairs ORDER BY id"
            ):
                component = find(identifier)
                if component in forbidden:
                    continue
                stage = (
                    "validation"
                    if component in validation
                    else "sft"
                    if component in sft
                    else "rl"
                )
                writers[stage].writerow([en, vi])
                counts[stage] += 1
        finally:
            for stream in streams.values():
                stream.close()
        counts["training_pool"] = counts["sft"] + counts["rl"]
        counts["test_directional_examples"] = 2 * counts["test_pairs"]
        manifest = {
            "version": 2,
            "seed": seed,
            "sft_fraction": sft_fraction,
            "actual_sft_fraction": counts["sft"] / counts["training_pool"],
            "validation_pairs_requested": validation_pairs,
            "counts": dict(counts),
            "groups": {
                "total_clean": len(groups),
                "largest_clean": max(n for n, _ in groups.values()),
                "validation": len(validation),
                "sft": len(sft),
                "rl": len(groups) - len(validation) - len(sft),
            },
            "inputs": {
                name: {"path": str(path.resolve()), "sha256": file_hash(path)}
                for name, path in (("train", train_file), ("test", test_file))
            },
            "outputs": {
                name: {"path": str(path), "sha256": file_hash(path)}
                for name, path in paths.items()
            },
            "test_file": str(test_file.resolve()),
            "official_test_unchanged": True,
            "normalized_en_vi_overlap": 0,
            "split_before_bidirectional_expansion_and_augmentation": True,
            "grouping": "Transitive components sharing normalized EN or VI; NFKC/casefold/whitespace; test-touching components excluded",
            "character_length_histogram_input": {
                k: dict(v) for k, v in lengths.items()
            },
            "elapsed_seconds": time.time() - started,
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        return manifest
    finally:
        connection.close()
        # Only this invocation's disposable database is removed; source CSVs stay intact.
        database.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-file", type=Path, default=Path("data/full/PhoMT_training.csv")
    )
    parser.add_argument(
        "--test-file", type=Path, default=Path("data/full/PhoMT_test.csv")
    )
    parser.add_argument("--output", type=Path, default=Path("data/full_prepared"))
    parser.add_argument("--validation-pairs", type=int, default=2000)
    parser.add_argument("--sft-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_full(
                args.train_file,
                args.test_file,
                args.output,
                args.validation_pairs,
                args.sft_fraction,
                args.seed,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
