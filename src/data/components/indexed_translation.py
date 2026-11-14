"""Memory-mapped CSV/JSONL translation rows, safe for spawned data workers."""

import csv
import io
import json
import mmap
import os
from array import array
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, BinaryIO, overload

import numpy as np


class IndexedTranslationRows(Sequence[dict[str, str]]):
    """Index logical CSV records including quoted newlines, without copying text.

    Sidecars are rebuilt when input size/mtime or column mapping changes. Every
    record is validated at index creation. Open mmap handles are worker-local.
    """

    def __init__(
        self,
        path: str | Path,
        source_column: str = "en",
        target_column: str = "vi",
        translation_column: str | None = None,
    ):
        self.path = Path(path).resolve()
        if self.path.suffix not in {".csv", ".jsonl", ".ndjson"}:
            raise ValueError("Indexed data must be CSV or JSONL")
        self.source_column, self.target_column, self.translation_column = (
            source_column,
            target_column,
            translation_column,
        )
        self._file: BinaryIO | None = None
        self._text: mmap.mmap | None = None
        self._offsets: Any = None
        self._pid: int | None = None
        stat = self.path.stat()
        signature = {
            "version": 1,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "columns": [source_column, target_column, translation_column],
        }
        self.index_path = self.path.with_suffix(self.path.suffix + ".offsets.npy")
        metadata = self.path.with_suffix(self.path.suffix + ".index.json")
        cached = (
            json.loads(metadata.read_text(encoding="utf-8"))
            if metadata.exists()
            else {}
        )
        if cached.get("signature") != signature or not self.index_path.exists():
            offsets = array("Q")
            fields: list[str] | None = None
            with self.path.open("rb") as stream:
                if self.path.suffix == ".csv":

                    def lines() -> Iterator[str]:
                        while line := stream.readline():
                            yield line.decode(
                                "utf-8-sig" if stream.tell() == len(line) else "utf-8"
                            )

                    reader = csv.DictReader(lines())
                    fields = list(reader.fieldnames) if reader.fieldnames else None
                    if not fields or not {source_column, target_column}.issubset(
                        fields
                    ):
                        raise ValueError("CSV is missing translation columns")
                    offsets.append(stream.tell())
                    for row in reader:
                        self._normalize(row)
                        offsets.append(stream.tell())
                else:
                    offsets.append(0)
                    for line in stream:
                        if not line.strip():
                            continue
                        self._normalize(json.loads(line.decode("utf-8-sig")))
                        offsets.append(stream.tell())
            if len(offsets) < 2:
                raise ValueError("Translation split is empty")
            temporary = self.index_path.with_name(
                self.index_path.name + f".{os.getpid()}.tmp"
            )
            with temporary.open("wb") as target:
                np.save(
                    target, np.asarray(offsets, dtype=np.uint64), allow_pickle=False
                )
            temporary.replace(self.index_path)
            cached = {
                "signature": signature,
                "rows": len(offsets) - 1,
                "fields": fields,
            }
            temporary_meta = metadata.with_name(metadata.name + f".{os.getpid()}.tmp")
            temporary_meta.write_text(json.dumps(cached), encoding="utf-8")
            temporary_meta.replace(metadata)
        self.fields = cached["fields"]
        self._count = int(cached["rows"])

    def _normalize(self, row: dict[str, Any]) -> dict[str, str]:
        pair = row[self.translation_column] if self.translation_column else row
        source, reference = pair[self.source_column], pair[self.target_column]
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (source, reference)
        ):
            raise ValueError("Empty or non-text translation pair")
        return {"source": source, "reference": reference}

    def __len__(self) -> int:
        return self._count

    @overload
    def __getitem__(self, index: int) -> dict[str, str]: ...

    @overload
    def __getitem__(self, index: slice) -> list[dict[str, str]]: ...

    def __getitem__(self, index: int | slice) -> dict[str, str] | list[dict[str, str]]:
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self._pid != os.getpid():
            self.close()
            self._file = self.path.open("rb")
            self._text = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
            self._offsets = np.load(self.index_path, mmap_mode="r", allow_pickle=False)
            self._pid = os.getpid()
        assert self._text is not None
        begin, end = int(self._offsets[index]), int(self._offsets[index + 1])
        text = self._text[begin:end].decode("utf-8-sig" if begin == 0 else "utf-8")
        if self.fields is not None:
            reader = csv.DictReader(
                io.StringIO(text, newline=""), fieldnames=self.fields
            )
            row = next(reader)
        else:
            row = json.loads(text)
        return self._normalize(row)

    def __iter__(self) -> Iterator[dict[str, str]]:
        return (self[i] for i in range(len(self)))

    def close(self) -> None:
        if hasattr(self, "_text") and self._text is not None:
            self._text.close()
        if hasattr(self, "_file") and self._file is not None:
            self._file.close()
        self._file = self._text = self._offsets = None
        self._pid = None

    def __getstate__(self) -> dict[str, Any]:
        return {
            **self.__dict__,
            "_file": None,
            "_text": None,
            "_offsets": None,
            "_pid": None,
        }

    def __del__(self) -> None:
        self.close()
