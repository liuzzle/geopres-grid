#!/usr/bin/env python
"""Populate the project cache for one or more MTEB tasks.

This is the WP-C.4 precompute entry point: it encodes the document and query
corpora for a task through the project-owned ``CachedBackbone`` so the warm-start
cache is created in the exact filesystem layout the downstream DR and scoring
stages expect.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer

from geopres_grid.backbones import PRIMARY_MAX_SEQ_LENGTH, get, require_runnable
from geopres_grid.config import EMBEDDING_CACHE_PATH, resolve_device
from geopres_grid.identity import EncodeConfig
from geopres_grid.precompute import precompute_task_cache


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", default="mdenseon", help="Backbone lookup key")
    parser.add_argument("--task", default="NanoSciFactRetrieval", help="MTEB task name")
    parser.add_argument("--split", default="test", help="Task split to precompute")
    parser.add_argument("--subset", default="default", help="Task subset to precompute")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-seq-length", type=int, default=PRIMARY_MAX_SEQ_LENGTH)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--device", default=None)
    parser.add_argument("--cache-root", default=str(EMBEDDING_CACHE_PATH))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    backbone = get(args.backbone)
    require_runnable(backbone)

    device = resolve_device(args.device)
    model = SentenceTransformer(
        backbone.model_id,
        revision=backbone.revision,
        device=device,
        trust_remote_code=backbone.trust_remote_code,
        model_kwargs={"dtype": args.dtype},
    )
    model.max_seq_length = args.max_seq_length

    encode_config = EncodeConfig.from_model(
        model,
        backbone.model_id,
        backbone.revision,
        dtype=args.dtype,
        device=device,
        batch_size=args.batch_size,
    )

    import mteb

    task = mteb.get_tasks(tasks=[args.task])[0]
    task.load_data()

    summary = precompute_task_cache(
        model=model,
        backbone=backbone,
        encode_config=encode_config,
        task=task,
        cache_root=args.cache_root,
        hf_split=args.split,
        hf_subset=args.subset,
        batch_size=args.batch_size,
    )

    print(
        f"Precomputed {summary['documents']} documents and {summary['queries']} queries "
        f"for {summary['task']} ({summary['split']}/{summary['subset']}) in "
        f"{summary['cache_root']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
