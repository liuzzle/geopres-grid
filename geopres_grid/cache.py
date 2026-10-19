"""Persistent, prompt-aware embedding cache for WP-C."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from geopres_grid.backbones import Backbone
from geopres_grid.identity import EncodeConfig


CORPUS_COPIES: dict[str, str] = {"ClimateFEVER": "FEVER"}
"""Retrieval tasks whose document blocks start as a copy of another task's.

BEIR's ClimateFEVER reuses FEVER's Wikipedia dump: 5,416,568 unique texts in
FEVER, 5,416,593 in ClimateFEVER (mteb 2.15.1 descriptive stats), about 18 GB
per model at 768-d fp32. Each task keeps its own block -- disk is not the
constraint (Andrianos, 13.10) -- but the second need not be encoded from scratch:
`precompute_task_cache` copies the source's finished block first
(`copy_document_blocks`) and then encodes only the misses.

No key changes in the copy. Keys are `sha256(prompted text)`, and the document
prompt does not depend on the task, so FEVER's keys are ClimateFEVER's keys. The
copied block keeps whatever FEVER-only passages exist; they are never looked up,
but they sit in ClimateFEVER's calibration sample like any cached row.
"""


def copy_document_blocks(
    encode_root: str | Path,
    source: str,
    target: str,
    *,
    splits: Iterable[str],
    subsets: Iterable[str],
) -> list[Path]:
    """Copy `source`'s document blocks to `target` under one encode hash, verified.

    Only a block that does not exist yet for `target` is copied, and only from a
    finished `source`: precompute writes a task's documents before its queries,
    so the source's query block must exist too. Each copy is written to a
    temporary directory, compared byte for byte with the source, loaded (the
    row-count check), and only then renamed into place, so a crash leaves no
    half-copied block. Returns the blocks copied.
    """
    import filecmp
    import shutil

    encode_root = Path(encode_root)
    copied: list[Path] = []
    for split in splits:
        for subset in subsets:
            destination = encode_root / target / split / subset / "document"
            origin = encode_root / source / split / subset / "document"
            finished = (encode_root / source / split / subset / "query" / "ids.parquet").exists()
            if destination.exists() or not (origin / "ids.parquet").exists() or not finished:
                continue
            GeoPresCache(origin).load()
            destination.parent.mkdir(parents=True, exist_ok=True)
            staging = Path(tempfile.mkdtemp(dir=destination.parent, prefix=".document-copy-"))
            try:
                names = ("embeddings.npy", "ids.parquet", "meta.json")
                for name in names:
                    if (origin / name).exists():
                        shutil.copyfile(origin / name, staging / name)
                for name in names:
                    if (origin / name).exists() and not filecmp.cmp(origin / name, staging / name, shallow=False):
                        raise RuntimeError(f"copy of {origin / name} differs from its source")
                GeoPresCache(staging).load()
                os.rename(staging, destination)
            finally:
                shutil.rmtree(staging, ignore_errors=True)
            copied.append(destination)
    return copied


def committed_rows(directory: str | Path) -> int:
    """Rows of a block that its index commits to: `ids.parquet`'s row count, 0 without one.

    Read from the parquet footer, so it costs nothing for a multi-million-row block.
    """
    import pyarrow.parquet as pq

    ids_file = Path(directory) / "ids.parquet"
    return int(pq.read_metadata(ids_file).num_rows) if ids_file.exists() else 0


def committed_embeddings(directory: str | Path) -> np.ndarray:
    """A block's committed vectors, memory-mapped: `embeddings.npy` cut to `committed_rows`.

    Rows past the index are an append that stopped before its index was written
    (`GeoPresCache.add`); they belong to no key and the next write replaces them.
    """
    directory = Path(directory)
    rows = committed_rows(directory)
    vectors = np.load(directory / "embeddings.npy", mmap_mode="r")
    if vectors.shape[0] < rows:
        raise ValueError(
            f"{directory}: embeddings.npy has {vectors.shape[0]} rows but ids.parquet "
            f"{rows} -- two writers or a damaged block; delete the block and "
            "precompute it again"
        )
    return vectors[:rows]


class GeoPresCache:
    """Store fp32 embeddings and a positional parquet index.

    ``prompted_texts`` is explicit because the raw item text is not necessarily
    the text sent to the backbone. The MTEB cache protocol remains supported for
    bare-text callers; the project wrapper should use the prompted-text methods.

    Writes are appends, so a corpus can be cached chunk by chunk with bounded
    memory and a killed job keeps every chunk it saved. `add` appends the new
    rows to `embeddings.npy` in place and then grows the array's header;
    `save` then replaces `ids.parquet` atomically, and that is the commit: a
    block is exactly the rows its index names. A job killed between the two
    leaves rows past the index, which every reader ignores and the next `add`
    overwrites. Fewer array rows than index rows cannot come from one writer
    and is refused.
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
        self._committed = 0

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
        # Raises on fewer array rows than index rows; cuts off an uncommitted tail.
        self._vectors = committed_embeddings(self.directory)
        if len(self._vectors) != len(self._ids):
            raise ValueError(f"{self.directory}: ids.parquet changed while loading -- two writers")
        self._committed = len(self._ids)
        self._index = {
            str(text_hash): int(row_idx)
            for text_hash, row_idx in zip(self._ids["text_hash"], self._ids["row_idx"])
        }
        if self.dimension is None:
            self.dimension = int(self._vectors.shape[1])

    def load(self) -> None:
        """Load the index and memory-map the embedding array."""
        if self.meta_file.exists():
            self.metadata = json.loads(self.meta_file.read_text(encoding="utf-8"))
        self._ensure_loaded()

    @staticmethod
    def _hashes(prompted_texts: Iterable[str]) -> list[str]:
        return [GeoPresCache.text_hash(text) for text in prompted_texts]

    def _rows(self, prompted_texts: Iterable[str]) -> np.ndarray:
        return np.array([self._index.get(key, -1) for key in self._hashes(prompted_texts)], dtype=np.int64)

    def missing(self, prompted_texts: Iterable[str]) -> np.ndarray:
        """Boolean mask of the texts not cached. Reads the index only, no vectors."""
        self._ensure_loaded()
        return self._rows(prompted_texts) < 0

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
        rows = self._rows(texts)
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
        self._append_embeddings(np.asarray(new_rows, dtype=np.float32), start=len(self._index))
        records = pd.DataFrame(new_records)
        # Not concatenated onto the empty frame: its object columns would turn every
        # string column into Python objects, about twice the memory of pandas' own
        # string dtype -- per row of a multi-million-passage index.
        self._ids = records if self._ids.empty else pd.concat([self._ids, records], ignore_index=True)
        self._index.update({record["text_hash"]: record["row_idx"] for record in new_records})
        self._vectors = np.load(self.embeddings_file, mmap_mode="r")[: len(self._index)]

    def _check_sole_writer(self) -> None:
        on_disk = committed_rows(self.directory)
        if on_disk != self._committed:
            raise ValueError(
                f"{self.directory}: ids.parquet has {on_disk} rows, this writer last "
                f"committed {self._committed} -- another process is writing this block"
            )

    def _append_embeddings(self, values: np.ndarray, *, start: int) -> None:
        """Write `values` as rows `start`, `start + 1`, ... of `embeddings.npy`, in place.

        Only the new rows and the header are written, never the existing rows, so
        caching a corpus in chunks costs linear, not quadratic, I/O. Anything past
        `start` (an uncommitted tail) is overwritten and cut off. The rows go in
        before the header names them, so a kill in between leaves the old header,
        which is still a valid file. numpy leaves room in every header for axis 0
        to grow to 21 digits; a header that would change length is refused rather
        than shifted.
        """
        self.directory.mkdir(parents=True, exist_ok=True)
        self._check_sole_writer()
        if not self.embeddings_file.exists():
            if start:
                raise ValueError(f"{self.directory}: {start} rows indexed but no embeddings.npy")
            self._write_embeddings(values)
            return
        from numpy.lib import format as npy

        writers = {(1, 0): npy.write_array_header_1_0, (2, 0): npy.write_array_header_2_0}
        readers = {(1, 0): npy.read_array_header_1_0, (2, 0): npy.read_array_header_2_0}
        with open(self.embeddings_file, "r+b") as handle:
            version = npy.read_magic(handle)
            if version not in readers:
                raise ValueError(f"{self.embeddings_file}: unsupported .npy version {version}")
            shape, fortran_order, dtype = readers[version](handle)
            offset = handle.tell()
            if (
                dtype != np.float32
                or fortran_order
                or len(shape) != 2
                or shape[1] != values.shape[1]
                or shape[0] < start
            ):
                raise ValueError(
                    f"{self.embeddings_file}: cannot append {values.shape} at row {start} "
                    f"to a {dtype} array of shape {shape}"
                )
            header = io.BytesIO()
            writers[version](
                header,
                {
                    "descr": npy.dtype_to_descr(np.dtype(np.float32)),
                    "fortran_order": False,
                    "shape": (start + len(values), int(shape[1])),
                },
            )
            if len(header.getvalue()) != offset:
                raise ValueError(f"{self.embeddings_file}: the grown header no longer fits")
            handle.seek(offset + start * shape[1] * np.dtype(np.float32).itemsize)
            handle.write(np.ascontiguousarray(values, dtype=np.float32).tobytes())
            handle.truncate()
            handle.flush()
            handle.seek(0)
            handle.write(header.getvalue())

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
        """Commit: write the index (atomically) and the metadata."""
        self._ensure_loaded()
        if self._vectors is None:
            return
        self._check_sole_writer()
        self.directory.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(dir=self.directory, suffix=".parquet")
        os.close(fd)
        temporary = Path(name)
        try:
            self._ids.to_parquet(temporary, index=False)
            os.replace(temporary, self.ids_file)
        finally:
            temporary.unlink(missing_ok=True)
        self._committed = len(self._ids)
        self.meta_file.write_text(
            json.dumps(self.metadata, indent=2, sort_keys=True), encoding="utf-8"
        )

    def close(self) -> None:
        """Release the memory map."""
        self._vectors = None

    def __del__(self) -> None:
        self.close()


def side_of(prompt_type: Any) -> str:
    """Cache side for an MTEB `prompt_type`; tasks without one (STS,
    classification, clustering) are queries."""
    value = getattr(prompt_type, "value", prompt_type)
    if value in (None, "query"):
        return "query"
    if value in ("passage", "document"):
        return "document"
    raise ValueError(f"Unsupported MTEB prompt_type: {prompt_type!r}")


def prompt_side(prompt_type: Any) -> str:
    """Which registry prompt an MTEB `prompt_type` gets: `side_of`, except that a
    task without one is "symmetric" (`Backbone.task_prompt_names`). The cache
    path and the quantizer still use `side_of`."""
    if getattr(prompt_type, "value", prompt_type) is None:
        return "symmetric"
    return side_of(prompt_type)


def _unit_rows(embeddings: Any) -> Any:
    import torch

    tensor = torch.as_tensor(np.asarray(embeddings, dtype=np.float32))
    return torch.nn.functional.normalize(tensor, dim=-1)


def cosine_similarity(embeddings1: Any, embeddings2: Any) -> Any:
    """All-pairs cosine, as `SentenceTransformer.similarity` computes it for the five
    backbones -- every one declares `similarity_fn_name: cosine`."""
    return _unit_rows(embeddings1) @ _unit_rows(embeddings2).T


def pairwise_cosine_similarity(embeddings1: Any, embeddings2: Any) -> Any:
    """Row-wise cosine, `SentenceTransformer.similarity_pairwise` for cosine models."""
    return (_unit_rows(embeddings1) * _unit_rows(embeddings2)).sum(dim=-1)


class CachedBackbone:
    """MTEB encoder wrapper backed by prompt-aware ``GeoPresCache`` blocks.

    ``inputs`` arrive already prepared by MTEB (documents as ``title + " " +
    text``, stripped). Misses are encoded from exactly that text with the
    registry's literal prefix passed as ``prompt``, so the string the model sees
    is byte-identical to the one the cache key hashes. ``prompt=""`` on a bare
    side also suppresses a model's ``default_prompt_name``, which
    ``prompt_name=None`` would not (sentence-transformers ``_resolve_prompt``).

    ``model=None`` serves a warm cache with no model loaded -- the evaluation
    path, which needs neither a GPU nor the backbone's environment. A miss is then
    an error naming the block, never a silent encode on the CPU box. Similarity is
    cosine either way, so a score does not depend on whether a model was loaded.
    """

    def __init__(
        self,
        model: Any | None,
        backbone: Backbone,
        encode_config: EncodeConfig,
        cache_root: str | Path,
    ) -> None:
        expected = tuple(sorted(backbone.task_prompt_names.items()))
        if encode_config.task_prompt_names != expected:
            raise ValueError(
                f"EncodeConfig names task prompts {dict(encode_config.task_prompt_names)!r} "
                f"but {backbone.key} uses {backbone.task_prompt_names!r}; pass "
                "task_prompt_names=backbone.task_prompt_names to from_model"
            )
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
        return cosine_similarity(embeddings1, embeddings2)

    def similarity_pairwise(self, embeddings1: Any, embeddings2: Any) -> Any:
        return pairwise_cosine_similarity(embeddings1, embeddings2)

    def _side(self, prompt_type: Any) -> str:
        return side_of(prompt_type)

    @property
    def encode_directory(self) -> Path:
        return backbone_cache_root(self.cache_root, self.backbone) / self.encode_config.hash

    def task_directory(self, task_name: str) -> Path:
        """Root of every cached block for one task: `{split}/{subset}/{side}` below."""
        return self.encode_directory / task_name

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

    def _prepare(
        self, inputs: Any, task_metadata: Any, hf_split: str, hf_subset: str, prompt_type: Any
    ) -> tuple[GeoPresCache, list[dict[str, Any]], list[str], str, list[str]]:
        items = self._items(inputs)
        raw_texts = [str(item.get("text", "")) for item in items]
        prefix = self.backbone.prompt_for(
            prompt_side(prompt_type), getattr(task_metadata, "type", None)
        )
        prompted_texts = [prefix + text for text in raw_texts]
        cache = self._get_cache(
            self._cache_path(task_metadata.name, hf_split, hf_subset, self._side(prompt_type))
        )
        return cache, items, raw_texts, prefix, prompted_texts

    def _encode_misses(
        self,
        cache: GeoPresCache,
        items: list[dict[str, Any]],
        raw_texts: list[str],
        prefix: str,
        prompted_texts: list[str],
        missing_indices: np.ndarray,
        *,
        batch_size: int,
        show_progress_bar: bool,
    ) -> np.ndarray:
        """Encode, store and commit the missing inputs; return their vectors in order."""
        if self._model is None:
            raise RuntimeError(
                f"{len(missing_indices)} of {len(items)} inputs are not cached in "
                f"{cache.directory} and no model is loaded. Precompute this task "
                "(scripts/precompute.py) or evaluate with --load-model."
            )
        # One encode per distinct prompted text. Corpora repeat documents
        # (NanoArguAna does), and two copies in differently padded batches come
        # back differing in the last bits -- which `GeoPresCache.add` rightly
        # refuses to store under one key.
        first: dict[str, int] = {}
        for index in missing_indices:
            first.setdefault(prompted_texts[index], int(index))
        unique = list(first.values())
        encoded = self._model.encode(
            [raw_texts[index] for index in unique],
            prompt=prefix,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=show_progress_bar,
        )
        if hasattr(encoded, "detach"):
            encoded = encoded.detach().cpu().numpy()
        encoded = np.asarray(encoded, dtype=np.float32)
        cache.add(
            [items[index] for index in unique],
            encoded,
            prompted_texts=[prompted_texts[index] for index in unique],
        )
        cache.save()
        self.newly_encoded += len(unique)
        row_of = {prompted_texts[index]: row for row, index in enumerate(unique)}
        return encoded[[row_of[prompted_texts[index]] for index in missing_indices]]

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
        cache, items, raw_texts, prefix, prompted_texts = self._prepare(
            inputs, task_metadata, hf_split, hf_subset, prompt_type
        )
        cached, missing = cache.get_vectors(items, prompted_texts=prompted_texts)
        missing_indices = np.flatnonzero(missing)
        newly_encoded = np.empty((len(missing_indices), cache.dimension or self.backbone.native_dim), dtype=np.float32)
        if len(missing_indices):
            newly_encoded = self._encode_misses(
                cache,
                items,
                raw_texts,
                prefix,
                prompted_texts,
                missing_indices,
                batch_size=batch_size,
                show_progress_bar=bool(kwargs.get("show_progress_bar", False)),
            )

        if cached is None:
            result = np.empty((len(items), newly_encoded.shape[1]), dtype=np.float32)
        else:
            result = cached.copy()
        if len(missing_indices):
            result[missing_indices] = newly_encoded
        return result

    def fill(
        self,
        inputs: Any,
        *,
        task_metadata: Any,
        hf_split: str,
        hf_subset: str,
        prompt_type: Any,
        batch_size: int = 32,
    ) -> int:
        """Cache `inputs` like `encode`, without reading or returning any vector.

        Precompute's path: a resumed job finds most chunks fully cached, and
        `encode` would read every one of their vectors back from disk for nothing.
        Returns the number of inputs that were misses.
        """
        cache, items, raw_texts, prefix, prompted_texts = self._prepare(
            inputs, task_metadata, hf_split, hf_subset, prompt_type
        )
        missing_indices = np.flatnonzero(cache.missing(prompted_texts))
        if len(missing_indices):
            self._encode_misses(
                cache,
                items,
                raw_texts,
                prefix,
                prompted_texts,
                missing_indices,
                batch_size=batch_size,
                show_progress_bar=False,
            )
        return len(missing_indices)

    def close(self) -> None:
        for cache in getattr(self, "_caches", {}).values():
            cache.close()

    def __del__(self) -> None:
        self.close()


def run_model_meta(backbone: Backbone, revision: str) -> Any:
    """`ModelMeta` that gives one grid run its own slot in MTEB's result cache."""
    from mteb.models.model_meta import ModelMeta

    return ModelMeta.create_empty(
        overwrites={"name": f"geopres-grid/{backbone.key}", "revision": revision}
    )


def backbone_cache_root(cache_root: str | Path, backbone: Backbone) -> Path:
    """Directory holding every encode hash of one backbone at its pinned revision."""
    return Path(cache_root) / f"{backbone.slug}@{backbone.revision}"


def encode_configs_on_disk(cache_root: str | Path, backbone: Backbone) -> dict[str, EncodeConfig]:
    """Encode configs with cached blocks for `backbone`, by hash, read from `meta.json`."""
    configs: dict[str, EncodeConfig] = {}
    root = backbone_cache_root(cache_root, backbone)
    if not root.exists():
        return configs
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        meta_file = next(directory.glob("*/*/*/*/meta.json"), None)
        if meta_file is not None:
            configs[directory.name] = EncodeConfig.from_meta(
                json.loads(meta_file.read_text(encoding="utf-8"))
            )
    return configs


def resolve_encode_config(
    cache_root: str | Path,
    backbone: Backbone,
    *,
    encode_hash: str | None = None,
    max_seq_length: int | None = None,
) -> EncodeConfig:
    """The one cached encode config for `backbone`, read off disk -- no model needed.

    `encode_hash` picks one explicitly; otherwise the configs are filtered by
    `max_seq_length` and the backbone's task prompts, and exactly one may remain.
    Two can legitimately exist -- a harrier precomputed in both transformers
    environments -- and then the caller has to choose.
    """
    configs = encode_configs_on_disk(cache_root, backbone)
    root = backbone_cache_root(cache_root, backbone)
    if encode_hash is not None:
        if encode_hash not in configs:
            raise LookupError(f"no cached encode hash {encode_hash} under {root}; found {sorted(configs) or 'none'}")
        return configs[encode_hash]
    expected = tuple(sorted(backbone.task_prompt_names.items()))
    candidates = {
        key: config
        for key, config in configs.items()
        if config.task_prompt_names == expected
        and (max_seq_length is None or config.max_seq_length == max_seq_length)
    }
    if len(candidates) != 1:
        found = ", ".join(
            f"{key} (max_seq_length={c.max_seq_length}, transformers {c.transformers_version}, "
            f"sentence-transformers {c.sentence_transformers_version})"
            for key, c in configs.items()
        )
        raise LookupError(
            f"{len(candidates)} cached encode configs for {backbone.key} match "
            f"max_seq_length={max_seq_length} under {root} -- found: {found or 'none'}. "
            "Precompute first, or pick one with --encode-hash."
        )
    return next(iter(candidates.values()))
