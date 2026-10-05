"""WP-C.4 task precompute entry points.

The cache backend is already implemented in ``geopres_grid.cache``. This module
adds the runtime path that populates the cache for a real MTEB retrieval task by
handing its corpus and queries to the ``CachedBackbone`` wrapper exactly as MTEB's
own retrieval evaluator does.

"Exactly" is the point. The cache key is the prompted text, and MTEB prepares the
text before encoding -- documents become ``(title + " " + text).strip()``, queries
get any instruction appended. Precompute therefore goes through MTEB's
``create_dataloader`` rather than reading raw fields: a precomputed key that
differs from the evaluation key by a title or a stripped space is a cache miss,
and a miss at evaluation time is an encode on the CPU box. ``mteb==2.15.1`` is
pinned; this relies on its ``_create_dataloaders`` behaviour.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mteb._create_dataloaders import create_dataloader
from mteb.abstasks.aggregated_task import AbsTaskAggregate
from mteb.types import PromptType

from geopres_grid.backbones import Backbone
from geopres_grid.cache import CachedBackbone
from geopres_grid.identity import EncodeConfig


def leaf_tasks(task: Any) -> list[Any]:
    """The tasks that actually call `encode`.

    An aggregate such as BEIR's `CQADupstackRetrieval` evaluates twelve subtasks
    and averages them; every subtask encodes under its own name, so caching and
    calibration happen per subtask.
    """
    if isinstance(task, AbsTaskAggregate):
        return [leaf for subtask in task.metadata.tasks for leaf in leaf_tasks(subtask)]
    return [task]


def precompute_task_cache(
    *,
    model: Any,
    backbone: Backbone,
    encode_config: EncodeConfig,
    task: Any,
    cache_root: str | Path,
    hf_splits: list[str] | None = None,
    hf_subsets: list[str] | None = None,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Populate the cache for the corpus and queries of one loaded retrieval task.

    Defaults to every evaluation split and subset the task declares; NanoBEIR's
    only split is `train`, MSMARCO's is `dev`, so no single default name is right.
    Returns a compact summary that is useful for CLI output and tests.
    """
    # No-op for tasks already in the v2 `task.dataset[subset][split]` layout; the
    # NanoBEIR tasks still load in v1 (`task.corpus`, `task.queries`).
    task.convert_v1_dataset_format_to_v2(num_proc=None)
    splits = list(hf_splits or task.eval_splits)
    subsets = list(hf_subsets or task.hf_subsets)

    wrapper = CachedBackbone(model, backbone, encode_config, cache_root)
    documents = queries = 0
    try:
        for subset in subsets:
            for split in splits:
                data = task.dataset[subset][split]
                for prompt_type, rows in (
                    (PromptType.document, data["corpus"]),
                    (PromptType.query, data["queries"]),
                ):
                    loader = create_dataloader(
                        rows,
                        task_metadata=task.metadata,
                        prompt_type=prompt_type,
                        batch_size=batch_size,
                    )
                    wrapper.encode(
                        loader,
                        task_metadata=task.metadata,
                        hf_split=split,
                        hf_subset=subset,
                        prompt_type=prompt_type,
                        batch_size=batch_size,
                    )
                    if prompt_type == PromptType.document:
                        documents += len(rows)
                    else:
                        queries += len(rows)
    finally:
        wrapper.close()

    if documents == 0 or queries == 0:
        raise ValueError(
            f"{task.metadata.name}: found {documents} documents and {queries} queries "
            f"in splits {splits}, subsets {subsets}; nothing was cached"
        )
    return {
        "task": task.metadata.name,
        "splits": splits,
        "subsets": subsets,
        "documents": documents,
        "queries": queries,
        "cache_root": str(Path(cache_root)),
        "newly_encoded": wrapper.newly_encoded,
    }
