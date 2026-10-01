"""WP-C.4 task precompute entry points.

The cache backend is already implemented in ``geopres_grid.cache``. This module
adds the runtime path that actually populates the cache for a real MTEB task by
walking the corpus and query dicts in task order and routing each side through the
``CachedBackbone`` wrapper.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from datasets import Dataset

from geopres_grid.backbones import Backbone
from geopres_grid.cache import CachedBackbone
from geopres_grid.identity import EncodeConfig


class _CompatTaskMetadata:
    """Minimise the task metadata surface expected by MTEB's dataloader.

    Real MTEB tasks expose `get_modalities()` and the task name; lightweight test
    doubles often do not. The wrapper only needs the task name and the modality
    metadata when it hands off to `create_dataloader`, so we provide the minimal
    behaviour instead of forcing every task object to be a full MTEB task.
    """

    def __init__(self, task: Any) -> None:
        self._task = task

    def get_modalities(self, prompt_type: Any = None) -> list[str]:
        return ["text"]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._task, name)


class TaskInputs:
    """Adapter that satisfies the ``CachedBackbone.encode`` contract."""

    def __init__(self, rows: Iterable[dict[str, Any]]) -> None:
        self.dataset = Dataset.from_list([dict(row) for row in rows])


def _coerce_row(item_id: Any, raw: Any) -> dict[str, Any]:
    if isinstance(raw, str):
        text = raw
    elif isinstance(raw, dict):
        text = raw.get("text") or raw.get("title") or ""
        if isinstance(text, list):
            text = " ".join(str(piece) for piece in text)
    else:
        text = str(raw)
    return {"id": str(item_id), "text": str(text)}


def _rows_from_mapping(mapping: dict[str, Any] | Any) -> list[dict[str, Any]]:
    if mapping is None:
        return []
    if hasattr(mapping, "items"):
        return [_coerce_row(item_id, payload) for item_id, payload in mapping.items()]
    return [_coerce_row(index, payload) for index, payload in enumerate(mapping)]


def precompute_task_cache(
    *,
    model: Any,
    backbone: Backbone,
    encode_config: EncodeConfig,
    task: Any,
    cache_root: str | Path,
    hf_split: str = "test",
    hf_subset: str = "default",
    batch_size: int = 32,
    **kwargs: Any,
) -> dict[str, Any]:
    """Populate the cache for the corpus and queries of one MTEB task.

    Returns a compact summary that is useful for CLI output and tests.
    """
    task_name = getattr(task, "name", None) or getattr(task, "metadata", None).name
    corpus = getattr(task, "corpus", {}).get(hf_split, {})
    queries = getattr(task, "queries", {}).get(hf_split, {})

    document_rows = _rows_from_mapping(corpus)
    query_rows = _rows_from_mapping(queries)

    wrapper = CachedBackbone(model, backbone, encode_config, cache_root)
    task_metadata = _CompatTaskMetadata(task)
    try:
        wrapper.encode(
            TaskInputs(document_rows),
            task_metadata=task_metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            prompt_type="document",
            batch_size=batch_size,
            **kwargs,
        )
        wrapper.encode(
            TaskInputs(query_rows),
            task_metadata=task_metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            prompt_type="query",
            batch_size=batch_size,
            **kwargs,
        )
        return {
            "task": task_name,
            "split": hf_split,
            "subset": hf_subset,
            "documents": len(document_rows),
            "queries": len(query_rows),
            "cache_root": str(Path(cache_root)),
            "newly_encoded": wrapper.newly_encoded,
        }
    finally:
        wrapper.close()
