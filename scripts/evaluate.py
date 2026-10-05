#!/usr/bin/env python
"""Evaluate one grid cell -- backbone x DR x quantization -- on an evaluation tier.

Per task: a warm-up pass through the embedding cache (the backbone's fp32
baseline), calibration on that task's cached inputs, then the post-processed
cell. Both results land in `$EVALUATION_RESULTS_PATH`, the baseline under the
encode hash and the cell under its run id, so no two runs share a result slot.
Runs on CPU / evaluation hardware once the cache is warm.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mteb
import numpy as np
from sentence_transformers import SentenceTransformer

from geopres_grid.backbones import PRIMARY_MAX_SEQ_LENGTH, get, require_runnable
from geopres_grid.cache import CachedBackbone
from geopres_grid.config import EMBEDDING_CACHE_PATH, EVALUATION_RESULTS_PATH, resolve_device
from geopres_grid.evaluation import PostProcessedBackbone, evaluate_cell, task_names
from geopres_grid.identity import BITS_PER_DIM, DR_METHODS, EncodeConfig, PostProcConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", default="mdenseon")
    parser.add_argument("--task", action="append", help="Explicit task; repeat to evaluate multiple tasks")
    parser.add_argument("--tier", choices=["tier0", "tier1", "tier2"], default="tier0")
    parser.add_argument("--dr-method", choices=DR_METHODS, default="none")
    parser.add_argument("--target-dim", type=int)
    parser.add_argument("--dr-seed", type=int)
    parser.add_argument("--quant-method", choices=sorted(BITS_PER_DIM), default="none")
    parser.add_argument(
        "--per-side-calibration",
        action="store_true",
        help="Fit one quantization table per side (symmetry ablation) instead of one shared",
    )
    parser.add_argument("--geopres-weights", type=Path, help="Trained GeoPres projection (.npy)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-seq-length", type=int, default=PRIMARY_MAX_SEQ_LENGTH)
    parser.add_argument("--device", default=None)
    parser.add_argument("--cache-root", default=str(EMBEDDING_CACHE_PATH))
    parser.add_argument("--results-root", default=str(EVALUATION_RESULTS_PATH))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    postproc = PostProcConfig(
        dr_method=args.dr_method,
        target_dim=args.target_dim,
        dr_seed=args.dr_seed,
        quant_method=args.quant_method,
        quant_symmetric=not args.per_side_calibration,
    )
    backbone = get(args.backbone)
    require_runnable(backbone)
    device = resolve_device(args.device)

    model = SentenceTransformer(
        backbone.model_id,
        revision=backbone.revision,
        device=device,
        trust_remote_code=backbone.trust_remote_code,
        model_kwargs={"dtype": "float32"},
    )
    model.max_seq_length = args.max_seq_length

    encode_config = EncodeConfig.from_model(
        model,
        backbone.model_id,
        backbone.revision,
        dtype="float32",
        device=device,
        batch_size=args.batch_size,
    )
    cached = CachedBackbone(model, backbone, encode_config, Path(args.cache_root))
    weights = np.load(args.geopres_weights) if args.geopres_weights else None
    cell = PostProcessedBackbone(cached, postproc, geopres_weights=weights)
    results_cache = mteb.ResultCache(args.results_root)
    print(f"run id {cell.run_id}  ({postproc.bits_per_dim} bits/dim)")

    selected_tasks = tuple(args.task) if args.task else task_names(args.tier)
    for task in mteb.get_tasks(tasks=list(selected_tasks)):
        baseline, scored = evaluate_cell(
            cell,
            task,
            results_cache=results_cache,
            encode_kwargs={"batch_size": args.batch_size},
        )
        before = baseline.task_results[0].get_score()
        after = scored.task_results[0].get_score()
        print(f"{task.metadata.name}: fp32 {before:.4f} -> cell {after:.4f}")
    print(f"{cached.newly_encoded} items newly encoded; results in {args.results_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
