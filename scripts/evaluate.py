#!/usr/bin/env python
"""Run a task through the cached model wrapper on CPU / evaluation hardware."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mteb
from sentence_transformers import SentenceTransformer

from geopres_grid.backbones import get, require_runnable
from geopres_grid.cache import CachedBackbone
from geopres_grid.config import EMBEDDING_CACHE_PATH, resolve_device
from geopres_grid.identity import EncodeConfig
from geopres_grid.evaluation import task_names


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", default="mdenseon")
    parser.add_argument("--task", action="append", help="Explicit task; repeat to evaluate multiple tasks")
    parser.add_argument("--tier", choices=["tier0", "tier1", "tier2"], default="tier0")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cache-root", default=str(EMBEDDING_CACHE_PATH))
    args = parser.parse_args(argv)

    backbone = get(args.backbone)
    require_runnable(backbone)
    device = resolve_device()

    model = SentenceTransformer(
        backbone.model_id,
        revision=backbone.revision,
        device=device,
        trust_remote_code=backbone.trust_remote_code,
        model_kwargs={"dtype": "float32"},
    )
    model.max_seq_length = 512

    encode_config = EncodeConfig.from_model(
        model,
        backbone.model_id,
        backbone.revision,
        dtype="float32",
        device=device,
        batch_size=args.batch_size,
    )
    wrapped = CachedBackbone(model, backbone, encode_config, Path(args.cache_root))

    selected_tasks = tuple(args.task) if args.task else task_names(args.tier)
    tasks = mteb.get_tasks(tasks=list(selected_tasks))
    for task in tasks:
        task.load_data()
    result = mteb.evaluate(wrapped, tasks=tasks, eval_splits=[args.split])
    print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
