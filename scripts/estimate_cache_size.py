#!/usr/bin/env python
"""Estimate fp32 cache size from loaded MTEB task row counts.

Retrieval tasks count corpus and queries. Every other task (STS, classification,
clustering) is cached on one side, `query`, and counts the texts of its
evaluation split -- an upper bound, since repeated texts share one row, and one
that leaves out classification's few-shot training sample.

Output is `{"rows": [...], "totals": {...}}`. A row counts its task's own corpus;
the per-backbone totals count each corpus once, so FEVER and ClimateFEVER
(`cache.SHARED_CORPORA`) contribute one block, sized as the larger of the two.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from geopres_grid.backbones import all_backbones, get
from geopres_grid.cache import corpus_directory_name


def _split_rows(collection: Any, split: str) -> int:
    if collection is None:
        return 0
    if hasattr(collection, "get"):
        values = collection.get(split, {})
    else:
        values = collection
    return len(values)


def _dataset_rows(task: Any, split: str, side: str) -> int | None:
    """Rows of `side` in MTEB's v2 layout, `task.dataset[subset][split][side]`.

    Standard retrieval tasks (all of full BEIR) load only into this layout and
    have no `task.corpus`; reading `corpus` alone would report 0 bytes for them.
    Returns None when the task is not in the v2 layout.
    """
    dataset = getattr(task, "dataset", None)
    if not isinstance(dataset, dict):
        return None
    counts = [
        len(subset[split][side])
        for subset in dataset.values()
        if isinstance(subset, dict) and split in subset and side in subset[split]
    ]
    return sum(counts) if counts else None


def _symmetric_texts(task: Any, split: str) -> int | None:
    """Texts a non-retrieval task encodes from `split`, across subsets.

    An STS row holds two sentences and a legacy clustering row a list of them;
    anything else is one text per row. None when the task has no such rows.
    A monolingual task loads as `task.dataset[split]`, a multilingual one as
    `task.dataset[subset][split]`; reading only the second counted STSBenchmark
    as empty.
    """
    dataset = getattr(task, "dataset", None)
    if not isinstance(dataset, dict):
        return None
    monolingual = split in dataset and not isinstance(dataset[split], dict)
    total, seen = 0, False
    for subset in [dataset] if monolingual else dataset.values():
        rows = subset.get(split) if hasattr(subset, "get") else None
        if rows is None or isinstance(rows, dict):
            continue
        seen = True
        columns = set(getattr(rows, "column_names", None) or (rows[0].keys() if len(rows) else ()))
        if "sentences" in columns:
            total += sum(len(row["sentences"]) for row in rows)
        elif {"sentence1", "sentence2"} <= columns:
            total += 2 * len(rows)
        else:
            total += len(rows)
    return total if seen else None


def estimate_task(task: Any, backbone: Any, split: str | None = None, itemsize: int = 4) -> dict[str, Any]:
    """Return document/query row counts and fp32 byte estimates for one task."""
    split = split or task.metadata.eval_splits[0]
    if getattr(task.metadata, "type", "Retrieval") not in ("Retrieval", None):
        texts = _symmetric_texts(task, split) or 0
        return _row(task, split, backbone, itemsize, documents=0, queries=texts)
    documents = _dataset_rows(task, split, "corpus")
    if documents is None:
        documents = _split_rows(getattr(task, "corpus", None), split)
    queries = _dataset_rows(task, split, "queries")
    if queries is None:
        queries = _split_rows(getattr(task, "queries", None), split)
    return _row(task, split, backbone, itemsize, documents=documents, queries=queries)


def _row(task: Any, split: str, backbone: Any, itemsize: int, *, documents: int, queries: int) -> dict[str, Any]:
    document_bytes = documents * backbone.native_dim * itemsize
    query_bytes = queries * backbone.native_dim * itemsize
    name = getattr(task, "name", None) or task.metadata.name
    return {
        "task": name,
        "corpus": corpus_directory_name(name),
        "split": split,
        "backbone": backbone.key,
        "dimension": backbone.native_dim,
        "itemsize": itemsize,
        "documents": documents,
        "queries": queries,
        "document_bytes": document_bytes,
        "query_bytes": query_bytes,
        "total_bytes": document_bytes + query_bytes,
    }


def totals(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Bytes on disk per backbone and over all of them (`"all"`), each corpus once.

    The rows of one shared corpus are sized as their largest member: the cache
    holds the union, which the descriptive stats put 25 rows above FEVER alone.
    """
    shared: dict[tuple[str, str, str], int] = {}
    result: dict[str, dict[str, int]] = {}
    for row in rows:
        total = result.setdefault(row["backbone"], {"document_bytes": 0, "query_bytes": 0, "total_bytes": 0})
        total["query_bytes"] += row["query_bytes"]
        key = (row["backbone"], row["corpus"], row["split"])
        shared[key] = max(shared.get(key, 0), row["document_bytes"])
    for (backbone, _, _), document_bytes in shared.items():
        result[backbone]["document_bytes"] += document_bytes
    for total in result.values():
        total["total_bytes"] = total["document_bytes"] + total["query_bytes"]
    result["all"] = {
        field: sum(total[field] for total in result.values())
        for field in ("document_bytes", "query_bytes", "total_bytes")
    }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", action="append", help="MTEB task name; repeat for multiple tasks")
    parser.add_argument("--tier", action="append", help="Estimate a whole tier or task set instead; repeatable")
    parser.add_argument("--backbone", action="append", help="Backbone key; defaults to all registry entries")
    parser.add_argument("--split", default=None, help="Split override; defaults to each task's first eval split")
    parser.add_argument("--itemsize", type=int, default=4, help="Bytes per scalar; 4 is fp32")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.itemsize <= 0:
        raise SystemExit("--itemsize must be positive")

    if bool(args.task) == bool(args.tier):
        raise SystemExit("pass either --task (repeatable) or --tier")

    from geopres_grid.evaluation import load_tasks, task_names
    from geopres_grid.precompute import leaf_tasks

    names = args.task or [name for tier in args.tier for name in task_names(tier)]
    tasks = [leaf for task in load_tasks(names) for leaf in leaf_tasks(task)]
    for task in tasks:
        task.load_data()
    backbones = [get(key) for key in args.backbone] if args.backbone else all_backbones()
    rows = [estimate_task(task, backbone, args.split, args.itemsize) for task in tasks for backbone in backbones]
    print(json.dumps({"rows": rows, "totals": totals(rows)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())