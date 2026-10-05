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

from geopres_grid.backbones import Backbone
from geopres_grid.identity import EncodeConfig


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


def side_of(prompt_type: Any) -> str:
    """Cache side for an MTEB `prompt_type`; tasks without one (STS) are queries."""
    value = getattr(prompt_type, "value", prompt_type)
    if value in (None, "query"):
        return "query"
    if value in ("passage", "document"):
        return "document"
    raise ValueError(f"Unsupported MTEB prompt_type: {prompt_type!r}")


class CachedBackbone:
    """MTEB encoder wrapper backed by prompt-aware ``GeoPresCache`` blocks.

    ``inputs`` arrive already prepared by MTEB (documents as ``title + " " +
    text``, stripped). Misses are encoded from exactly that text with the
    registry's literal prefix passed as ``prompt``, so the string the model sees
    is byte-identical to the one the cache key hashes. ``prompt=""`` on a bare
    side also suppresses a model's ``default_prompt_name``, which
    ``prompt_name=None`` would not (sentence-transformers ``_resolve_prompt``).
    """

    def __init__(
        self,
        model: Any,
        backbone: Backbone,
        encode_config: EncodeConfig,
        cache_root: str | Path,
    ) -> None:
        self._model = model
        self.backbone = backbone
        self.encode_config = encode_config
        self.cache_root = Path(cache_root)
        self._caches: dict[Path, GeoPresCache] = {}
        self.newly_encoded = 0

    @property
    def mteb_model_meta(self) -> Any:
        """Name and revision under which MTEB stores this wrapper's results.

        MTEB's result cache is keyed by `ModelMeta` name and revision only, and
        skips a task whose result already exists there. An empty meta gives every
        run the same slot, so a second backbone or a post-processed run would
        silently return the first run's scores. The revision is the encode hash.
        """
        return run_model_meta(self.backbone, self.encode_config.hash)

    def load_model(self) -> "CachedBackbone":
        """Satisfy MTEB's model lifecycle without loading a second model."""
        return self

    def similarity(self, embeddings1: Any, embeddings2: Any) -> Any:
        return self._model.similarity(embeddings1, embeddings2)

    def similarity_pairwise(self, embeddings1: Any, embeddings2: Any) -> Any:
        return self._model.similarity_pairwise(embeddings1, embeddings2)

    def _side(self, prompt_type: Any) -> str:
        return side_of(prompt_type)

    def task_directory(self, task_name: str) -> Path:
        """Root of every cached block for one task: `{split}/{subset}/{side}` below."""
        return (
            self.cache_root
            / f"{self.backbone.slug}@{self.backbone.revision}"
            / self.encode_config.hash
            / task_name
        )

    def _cache_path(
        self, task_name: str, hf_split: str, hf_subset: str, side: str
    ) -> Path:
        return self.task_directory(task_name) / hf_split / hf_subset / side

    def _get_cache(self, path: Path) -> GeoPresCache:
        if path not in self._caches:
            cache = GeoPresCache(path, metadata=self.encode_config.to_meta())
            cache.load()
            self._caches[path] = cache
        return self._caches[path]

    @staticmethod
    def _items(inputs: Any) -> list[dict[str, Any]]:
        return [dict(item) for item in inputs.dataset]

    def encode(
        self,
        inputs: Any,
        *,
        task_metadata: Any,
        hf_split: str,
        hf_subset: str,
        prompt_type: Any = None,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> np.ndarray:
        """Return cached vectors, encoding only prompted-text misses."""
        side = self._side(prompt_type)
        items = self._items(inputs)
        raw_texts = [str(item.get("text", "")) for item in items]
        prefix = self.backbone.prompt_for(side)
        prompted_texts = [prefix + text for text in raw_texts]
        cache = self._get_cache(
            self._cache_path(task_metadata.name, hf_split, hf_subset, side)
        )
        cached, missing = cache.get_vectors(items, prompted_texts=prompted_texts)
        missing_indices = np.flatnonzero(missing)
        newly_encoded = np.empty((len(missing_indices), cache.dimension or self.backbone.native_dim), dtype=np.float32)
        if len(missing_indices):
            missing_items = [items[index] for index in missing_indices]
            encoded = self._model.encode(
                [raw_texts[index] for index in missing_indices],
                prompt=prefix,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=bool(kwargs.get("show_progress_bar", False)),
            )
            if hasattr(encoded, "detach"):
                encoded = encoded.detach().cpu().numpy()
            newly_encoded = np.asarray(encoded, dtype=np.float32)
            cache.add(
                missing_items,
                newly_encoded,
                prompted_texts=[prompted_texts[index] for index in missing_indices],
            )
            cache.save()
            self.newly_encoded += len(missing_indices)

        if cached is None:
            result = np.empty((len(items), newly_encoded.shape[1]), dtype=np.float32)
        else:
            result = cached.copy()
        if len(missing_indices):
            result[missing_indices] = newly_encoded
        return result

    def close(self) -> None:
        for cache in self._caches.values():
            cache.close()

    def __del__(self) -> None:
        self.close()


def run_model_meta(backbone: Backbone, revision: str) -> Any:
    """`ModelMeta` that gives one grid run its own slot in MTEB's result cache."""
    from mteb.models.model_meta import ModelMeta

    return ModelMeta.create_empty(
        overwrites={"name": f"geopres-grid/{backbone.key}", "revision": revision}
    )