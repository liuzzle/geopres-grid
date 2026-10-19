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
from typing import Any, Callable

from mteb._create_dataloaders import create_dataloader
from mteb.abstasks.aggregated_task import AbsTaskAggregate
from mteb.types import PromptType

from geopres_grid.backbones import Backbone
from geopres_grid.cache import CORPUS_COPIES, CachedBackbone, copy_document_blocks
from geopres_grid.identity import EncodeConfig


DEFAULT_CHUNK_SIZE = 200_000
"""Passages encoded and committed per step. Bounds precompute's memory and what
a killed job loses; every NanoBEIR corpus fits in one chunk, so tier 1 encodes
exactly as before (one encode call per block). At 2048-d fp32 a chunk is 1.6 GB."""


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
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Populate the cache for the corpus and queries of one loaded retrieval task.

    Defaults to every evaluation split and subset the task declares; NanoBEIR's
    only split is `train`, MSMARCO's is `dev`, so no single default name is right.
    A task in `cache.CORPUS_COPIES` (ClimateFEVER) first copies its source's
    finished document blocks, so only the passages the source lacks are encoded;
    without a finished source it encodes everything, which is slower, not wrong.

    Each side is cached `chunk_size` rows at a time, and each chunk is committed
    before the next is read, so memory does not grow with the corpus (beyond the
    block's index) and a job killed at its time limit resumes, when run again,
    at the first chunk it had not committed. `progress` gets one line per chunk.
    Returns a compact summary that is useful for CLI output and tests.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    # No-op for tasks already in the v2 `task.dataset[subset][split]` layout; the
    # NanoBEIR tasks still load in v1 (`task.corpus`, `task.queries`).
    task.convert_v1_dataset_format_to_v2(num_proc=None)
    splits = list(hf_splits or task.eval_splits)
    subsets = list(hf_subsets or task.hf_subsets)

    wrapper = CachedBackbone(model, backbone, encode_config, cache_root)
    copied: list[Path] = []
    if task.metadata.name in CORPUS_COPIES:
        copied = copy_document_blocks(
            wrapper.encode_directory,
            CORPUS_COPIES[task.metadata.name],
            task.metadata.name,
            splits=splits,
            subsets=subsets,
        )
    documents = queries = 0
    try:
        for subset in subsets:
            for split in splits:
                data = task.dataset[subset][split]
                for prompt_type, rows in (
                    (PromptType.document, data["corpus"]),
                    (PromptType.query, data["queries"]),
                ):
                    for begin in range(0, len(rows), chunk_size):
                        end = min(begin + chunk_size, len(rows))
                        # Sliced before MTEB prepares the text, as its own
                        # `_full_corpus_search` does, so only one chunk is ever
                        # materialised.
                        loader = create_dataloader(
                            rows.select(range(begin, end)),
                            task_metadata=task.metadata,
                            prompt_type=prompt_type,
                            batch_size=batch_size,
                        )
                        misses = wrapper.fill(
                            loader,
                            task_metadata=task.metadata,
                            hf_split=split,
                            hf_subset=subset,
                            prompt_type=prompt_type,
                            batch_size=batch_size,
                        )
                        if progress is not None:
                            progress(
                                f"{task.metadata.name} {split}/{subset} {prompt_type.value}: "
                                f"{end:,}/{len(rows):,} cached, {misses:,} of this chunk encoded"
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
        "copied_blocks": [str(path) for path in copied],
    }
