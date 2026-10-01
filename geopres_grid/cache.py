"""Persistent, prompt-aware embedding cache for WP-C."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


class GeoPresCache:
    """Store fp32 embeddings and a positional parquet index.

    ``prompted_texts`` is explicit because the raw item text is not necessarily
    the text sent to the backbone. The MTEB cache protocol remains supported for
    bare-text callers; the project wrapper should use the prompted-text methods.
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        dimension: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.directory = Path(directory)
        self.embeddings_file = self.directory / "embeddings.npy"
        self.ids_file = self.directory / "ids.parquet"
        self.meta_file = self.directory / "meta.json"
        self.dimension = dimension
        self.metadata = metadata or {}
        self._index: dict[str, int] = {}
        self._ids = pd.DataFrame(
            columns=("row_idx", "item_id", "text_hash", "prompted_text")
        )
        self._vectors: np.ndarray | None = None

    @staticmethod
    def text_hash(prompted_text: str) -> str:
        """Return the stable key for the exact text sent to the model."""
        return hashlib.sha256(prompted_text.encode("utf-8")).hexdigest()

    @staticmethod
    def _item_id(item: dict[str, Any], prompted_text: str) -> str:
        value = item.get("id", item.get("doc_id", item.get("text", prompted_text)))
        return str(value)

    def _ensure_loaded(self) -> None:
        if self._vectors is not None:
            return
        if not self.embeddings_file.exists() or not self.ids_file.exists():
            return
        self._ids = pd.read_parquet(self.ids_file)
        self._ids["row_idx"] = self._ids["row_idx"].astype(int)
        self._index = {
            str(text_hash): int(row_idx)
            for text_hash, row_idx in zip(self._ids["text_hash"], self._ids["row_idx"])
        }
        if self.dimension is None:
            shape = np.load(self.embeddings_file, mmap_mode="r").shape
            self.dimension = int(shape[1])
        self._vectors = np.load(self.embeddings_file, mmap_mode="r+")

    def load(self) -> None:
        """Load the index and memory-map the embedding array."""
        if self.meta_file.exists():
            self.metadata = json.loads(self.meta_file.read_text(encoding="utf-8"))
        self._ensure_loaded()

    @staticmethod
    def _hashes(prompted_texts: Iterable[str]) -> list[str]:
        return [GeoPresCache.text_hash(text) for text in prompted_texts]

    def get_vectors(
        self,
        items: list[dict[str, Any]],
        *,
        prompted_texts: Iterable[str] | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray]:
        """Return vectors in request order and a boolean mask for cache misses."""
        self._ensure_loaded()
        texts = list(prompted_texts) if prompted_texts is not None else [
            str(item.get("text", "")) for item in items
        ]
        if len(texts) != len(items):
            raise ValueError("prompted_texts must have one entry per item")
        rows = np.array([self._index.get(key, -1) for key in self._hashes(texts)], dtype=np.int64)
        missing = rows < 0
        if self._vectors is None:
            return None, missing
        result = np.empty((len(items), self.dimension), dtype=np.float32)
        if (~missing).any():
            result[~missing] = self._vectors[rows[~missing]]
        return result, missing

    def get_vector(self, item: dict[str, Any]) -> np.ndarray | None:
        """Protocol-compatible bare-text lookup."""
        vectors, missing = self.get_vectors([item])
        return None if missing[0] else vectors[0]

    def __contains__(self, item: dict[str, Any]) -> bool:
        _, missing = self.get_vectors([item])
        return not bool(missing[0])

    def add(
        self,
        items: list[dict[str, Any]],
        vectors: np.ndarray,
        *,
        prompted_texts: Iterable[str] | None = None,
    ) -> None:
        """Add vectors, rejecting conflicting duplicate keys."""
        self._ensure_loaded()
        values = np.asarray(vectors, dtype=np.float32)
        if values.ndim == 1:
            values = values[None, :]
        if len(items) != len(values):
            raise ValueError("items and vectors must have the same length")
        if self.dimension is None:
            self.dimension = int(values.shape[1])
        if values.shape[1] != self.dimension:
            raise ValueError("vector dimension does not match the cache")
        texts = list(prompted_texts) if prompted_texts is not None else [
            str(item.get("text", "")) for item in items
        ]
        if len(texts) != len(items):
            raise ValueError("prompted_texts must have one entry per item")

        new_rows: list[np.ndarray] = []
        new_records: list[dict[str, Any]] = []
        pending: dict[str, int] = {}
        for item, vector, text in zip(items, values, texts):
            text_hash = self.text_hash(text)
            existing = self._index.get(text_hash)
            if existing is None:
                existing = pending.get(text_hash)
            if existing is not None:
                previous = (
                    self._vectors[existing]
                    if self._vectors is not None and existing < len(self._index)
                    else new_rows[existing - len(self._index)]
                )
                if not np.array_equal(previous, vector):
                    raise ValueError(f"conflicting vector for text hash {text_hash}")
                continue
            row_idx = len(self._index) + len(new_rows)
            pending[text_hash] = row_idx
            new_rows.append(vector)
            new_records.append({
                "row_idx": row_idx,
                "item_id": self._item_id(item, text),
                "text_hash": text_hash,
                "prompted_text": text,
            })

        if not new_rows:
            return
        existing = self._vectors[: len(self._index)] if self._vectors is not None else None
        combined = np.vstack([part for part in (existing, np.asarray(new_rows, dtype=np.float32)) if part is not None])
        self._write_embeddings(combined)
        self._ids = pd.concat([self._ids, pd.DataFrame(new_records)], ignore_index=True)
        self._index.update({record["text_hash"]: record["row_idx"] for record in new_records})
        self._vectors = np.load(self.embeddings_file, mmap_mode="r+")

    def _write_embeddings(self, values: np.ndarray) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(dir=self.directory, suffix=".npy")
        os.close(fd)
        temporary = Path(name)
        try:
            np.save(temporary, values.astype(np.float32, copy=False))
            os.replace(temporary, self.embeddings_file)
        finally:
            temporary.unlink(missing_ok=True)

    def save(self) -> None:
        """Persist embeddings, positional identifiers, and metadata."""
        self._ensure_loaded()
        if self._vectors is None:
            return
        self._vectors.flush()
        self.directory.mkdir(parents=True, exist_ok=True)
        self._ids.to_parquet(self.ids_file, index=False)
        self.meta_file.write_text(
            json.dumps(self.metadata, indent=2, sort_keys=True), encoding="utf-8"
        )

    def close(self) -> None:
        """Flush and release the memory map."""
        if self._vectors is not None:
            self._vectors.flush()
            self._vectors = None

    def __del__(self) -> None:
        self.close()