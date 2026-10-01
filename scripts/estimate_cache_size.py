#!/usr/bin/env python
"""Estimate fp32 cache size from loaded MTEB task row counts."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from geopres_grid.backbones import all_backbones, get


def _split_rows(collection: Any, split: str) -> int:
    if collection is None:
        return 0
    if hasattr(collection, "get"):
        values = collection.get(split, {})
    else:
        values = collection
    return len(values)


def estimate_task(task: Any, backbone: Any, split: str = "test", itemsize: int = 4) -> dict[str, Any]:
    """Return document/query row counts and fp32 byte estimates for one task."""
    documents = _split_rows(getattr(task, "corpus", None), split)
    queries = _split_rows(getattr(task, "queries", None), split)
    document_bytes = documents * backbone.native_dim * itemsize
    query_bytes = queries * backbone.native_dim * itemsize
    return {
        "task": task.name,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", action="append", required=True, help="MTEB task name; repeat for multiple tasks")
    parser.add_argument("--backbone", action="append", help="Backbone key; defaults to all registry entries")
    parser.add_argument("--split", default="test")
    parser.add_argument("--itemsize", type=int, default=4, help="Bytes per scalar; 4 is fp32")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.itemsize <= 0:
        raise SystemExit("--itemsize must be positive")

    import mteb

    tasks = mteb.get_tasks(tasks=args.task)
    for task in tasks:
        task.load_data()
    backbones = [get(key) for key in args.backbone] if args.backbone else all_backbones()
    rows = [estimate_task(task, backbone, args.split, args.itemsize) for task in tasks for backbone in backbones]
    print(json.dumps(rows, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())