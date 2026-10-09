#!/usr/bin/env python
"""Evaluate one backbone's grid cells (`geopres_grid.grid`) on one tier, from a warm cache.

No model is loaded and no GPU is needed: the encode config is read back from the
cache, as in `scripts/evaluate.py`. Tasks are the outer loop, so each task's data
is loaded once and every cell is scored against it.

Run `--baselines-only` once per backbone before the cells. Every cell reads the
backbone's fp32 baseline result; with the baselines in place that is a cache hit,
so parallel shards never write the same result file. On a Slurm array the shard
comes from `SLURM_ARRAY_TASK_ID` / `SLURM_ARRAY_TASK_COUNT`.

Rerunning is cheap: MTEB skips any (cell, task) whose result already exists, so
a shard that hit its time limit is resumed by submitting it again.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

import mteb

from geopres_grid.backbones import PRIMARY_MAX_SEQ_LENGTH, get
from geopres_grid.cache import CachedBackbone, resolve_encode_config
from geopres_grid.config import EMBEDDING_CACHE_PATH, EVALUATION_RESULTS_PATH
from geopres_grid.evaluation import TIER_TASKS, PostProcessedBackbone, baseline_pass, evaluate_cell, load_tasks, task_names
from geopres_grid.grid import GRID_DR_METHODS, GRID_QUANT_METHODS, GRID_TARGET_DIMS, grid_cells, shard
from geopres_grid.identity import BITS_PER_DIM, DR_METHODS
from geopres_grid.precompute import leaf_tasks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--tier", choices=sorted(TIER_TASKS), default="tier1")
    parser.add_argument("--task", action="append", help="Restrict to these tasks; repeatable")
    parser.add_argument("--baselines-only", action="store_true", help="Score the fp32 baselines and stop")
    parser.add_argument("--list", action="store_true", help="Print this shard's cells and stop")
    parser.add_argument("--dr-method", action="append", choices=DR_METHODS, help=f"Restrict; default {GRID_DR_METHODS}")
    parser.add_argument("--target-dim", action="append", type=int, help=f"Restrict; default {GRID_TARGET_DIMS}")
    parser.add_argument("--quant-method", action="append", choices=sorted(BITS_PER_DIM), help=f"Restrict; default {GRID_QUANT_METHODS}")
    parser.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    parser.add_argument("--num-shards", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)))
    parser.add_argument("--max-seq-length", type=int, default=PRIMARY_MAX_SEQ_LENGTH)
    parser.add_argument("--encode-hash", help="Cached encode config to use when several match")
    parser.add_argument("--cache-root", default=str(EMBEDDING_CACHE_PATH))
    parser.add_argument("--results-root", default=str(EVALUATION_RESULTS_PATH))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    backbone = get(args.backbone)
    cells = shard(
        grid_cells(
            backbone,
            dr_methods=args.dr_method or GRID_DR_METHODS,
            target_dims=args.target_dim or GRID_TARGET_DIMS,
            quant_methods=args.quant_method or GRID_QUANT_METHODS,
        ),
        args.shard,
        args.num_shards,
    )
    if args.list:
        for config in cells:
            print(config.dr_method, config.target_dim, config.quant_method)
        print(f"{len(cells)} cells in shard {args.shard}/{args.num_shards} for {backbone.key}")
        return 0

    encode_config = resolve_encode_config(
        args.cache_root, backbone, encode_hash=args.encode_hash, max_seq_length=args.max_seq_length
    )
    cached = CachedBackbone(None, backbone, encode_config, Path(args.cache_root))
    results_cache = mteb.ResultCache(args.results_root)
    names = tuple(args.task) if args.task else task_names(args.tier)
    print(
        f"{backbone.key} (encode {encode_config.hash}), {len(names)} tasks, "
        + ("baselines only" if args.baselines_only else f"{len(cells)} cells, shard {args.shard}/{args.num_shards}"),
        flush=True,
    )

    failures: list[str] = []
    for task in load_tasks(names):
        name = task.metadata.name
        leaves = leaf_tasks(task)
        for leaf in leaves:
            if not leaf.data_loaded:
                leaf.load_data()
        try:
            if args.baselines_only:
                result = baseline_pass(cached, task, results_cache=results_cache)
                print(f"{name}: fp32 {result.task_results[0].get_score():.4f}", flush=True)
                continue
            for config in cells:
                label = f"{name} {config.dr_method}-{config.target_dim} {config.quant_method}"
                started = time.monotonic()
                try:
                    baseline, scored = evaluate_cell(
                        PostProcessedBackbone(cached, config), task, results_cache=results_cache
                    )
                except Exception:  # one bad cell must not cost the rest of the shard
                    failures.append(label)
                    print(f"FAILED {label}\n{traceback.format_exc()}", flush=True)
                    continue
                print(
                    f"{label}: {baseline.task_results[0].get_score():.4f} -> "
                    f"{scored.task_results[0].get_score():.4f} ({time.monotonic() - started:.1f} s)",
                    flush=True,
                )
        finally:
            for leaf in leaves:
                leaf.unload_data()

    if failures:
        print(f"{len(failures)} failed:\n  " + "\n  ".join(failures))
        return 1
    print(f"Done; results in {args.results_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
