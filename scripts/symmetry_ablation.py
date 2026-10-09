#!/usr/bin/env python
"""Calibration symmetry ablation (WP-E.5) on one task's cached embeddings.

For one backbone, one task and one DR setting, compares shared calibration,
per-side calibration and per-side raw codes against the fp32 scores, for each
quantization method. CPU only; needs a warm cache, no model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from geopres_grid.backbones import get
from geopres_grid.cache import backbone_cache_root, resolve_encode_config, task_block_roots
from geopres_grid.evaluation import CALIBRATION_MAX_ROWS, CALIBRATION_SEED, cached_blocks, sample_rows
from geopres_grid.identity import DR_METHODS, PostProcConfig
from geopres_grid.quantizers import calibration_symmetry_ablation
from geopres_grid.reducers import PostProcessingPipeline

DEFAULT_METHODS = ("int8", "int4", "int2", "equal_count_8", "equal_count_4", "equal_count_2")


def resolve_block_roots(cache_root: Path, backbone_key: str, encode_hash: str | None, task: str) -> tuple[Path, ...]:
    """`CachedBackbone.block_roots` without loading the model.

    The encode hash comes from the model, so it is either given or read off the
    cache (`cache.resolve_encode_config`): exactly one config must match.
    """
    backbone = get(backbone_key)
    try:
        config = resolve_encode_config(cache_root, backbone, encode_hash=encode_hash)
    except LookupError as error:
        raise SystemExit(str(error)) from error
    return task_block_roots(backbone_cache_root(cache_root, backbone) / config.hash, task)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--encode-hash")
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--dr-method", choices=DR_METHODS, default="none")
    parser.add_argument("--target-dim", type=int)
    parser.add_argument("--dr-seed", type=int)
    parser.add_argument("--quant-method", action="append", help=f"repeatable; default {DEFAULT_METHODS}")
    parser.add_argument("--max-queries", type=int, default=1_000)
    parser.add_argument("--max-documents", type=int, default=10_000)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args(argv)

    if args.cache_root is None:
        from geopres_grid.config import EMBEDDING_CACHE_PATH

        args.cache_root = Path(EMBEDDING_CACHE_PATH)
    blocks = cached_blocks(resolve_block_roots(args.cache_root, args.backbone, args.encode_hash, args.task))
    if not blocks["query"] or not blocks["document"]:
        raise SystemExit(f"{args.task} needs cached query and document blocks")

    dr = PostProcConfig(dr_method=args.dr_method, target_dim=args.target_dim, dr_seed=args.dr_seed)
    calibration = sample_rows(blocks["query"] + blocks["document"], CALIBRATION_MAX_ROWS, CALIBRATION_SEED)
    pipeline = PostProcessingPipeline(dr, calibration.shape[1])
    pipeline.fit(calibration)
    queries = pipeline.transform(sample_rows(blocks["query"], args.max_queries, CALIBRATION_SEED))
    documents = pipeline.transform(sample_rows(blocks["document"], args.max_documents, CALIBRATION_SEED))

    print(f"{args.backbone} {args.task} dr={args.dr_method} d'={queries.shape[1]} "
          f"queries={len(queries)} documents={len(documents)}")
    print(f"{'method':<15}{'variant':<23}{'spearman':>10}{f'recall@{args.k}':>12}")
    for method in args.quant_method or DEFAULT_METHODS:
        config = PostProcConfig(
            dr_method=args.dr_method, target_dim=args.target_dim, dr_seed=args.dr_seed, quant_method=method
        )
        scores = calibration_symmetry_ablation(config, queries, documents, k=args.k)
        for variant, values in scores.items():
            print(f"{method:<15}{variant:<23}{values['spearman']:>10.4f}{values[f'recall@{args.k}']:>12.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
