#!/usr/bin/env python
"""Populate the project cache for one or more MTEB tasks. The only GPU step.

This is the WP-C.4 precompute entry point. Retrieval tasks have their corpus and
queries encoded through the project-owned ``CachedBackbone`` directly. Every
other task (STS, classification, clustering) is run once through MTEB with the
same wrapper -- the fp32 baseline pass -- because what those tasks encode
depends on MTEB's own sampling; the baseline scores land in the results root as
a by-product. Afterwards `scripts/evaluate.py` runs on CPU with no model loaded.
"""

from __future__ import annotations

import argparse
import sys

from sentence_transformers import SentenceTransformer

from geopres_grid.backbones import PRIMARY_MAX_SEQ_LENGTH, get, require_runnable
from geopres_grid.cache import CachedBackbone
from geopres_grid.config import EMBEDDING_CACHE_PATH, EVALUATION_RESULTS_PATH, resolve_device
from geopres_grid.evaluation import TIER_TASKS, baseline_pass, load_tasks, task_names
from geopres_grid.identity import EncodeConfig
from geopres_grid.precompute import leaf_tasks, precompute_task_cache


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", default="mdenseon", help="Backbone lookup key")
    parser.add_argument("--task", action="append", help="MTEB task name; repeat for several")
    parser.add_argument("--tier", choices=sorted(TIER_TASKS), help="Precompute a whole tier or task set")
    parser.add_argument("--split", action="append", help="Override the task's eval splits")
    parser.add_argument("--subset", action="append", help="Override the task's subsets")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-seq-length", type=int, default=PRIMARY_MAX_SEQ_LENGTH)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--device", default=None)
    parser.add_argument("--cache-root", default=str(EMBEDDING_CACHE_PATH))
    parser.add_argument(
        "--results-root",
        default=str(EVALUATION_RESULTS_PATH),
        help="Where the baseline pass of non-retrieval tasks writes its scores",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if bool(args.task) == bool(args.tier):
        raise SystemExit("pass either --task (repeatable) or --tier")
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
        task_prompt_names=backbone.task_prompt_names,
    )

    import mteb
    from mteb.abstasks import AbsTaskRetrieval

    results_cache = mteb.ResultCache(args.results_root)
    names = list(args.task or task_names(args.tier))
    for named in load_tasks(names):
        if not all(isinstance(leaf, AbsTaskRetrieval) for leaf in leaf_tasks(named)):
            if args.split or args.subset:
                raise SystemExit(f"--split/--subset apply to retrieval tasks only, not {named.metadata.name}")
            cached = CachedBackbone(model, backbone, encode_config, args.cache_root)
            baseline_pass(
                cached,
                named,
                results_cache=results_cache,
                encode_kwargs={"batch_size": args.batch_size},
                overwrite_strategy="always",
            )
            cached.close()
            print(
                f"Baseline pass for {named.metadata.name}: {cached.newly_encoded} newly "
                f"encoded, in {args.cache_root}; scores in {args.results_root}"
            )
            continue
        for task in leaf_tasks(named):
            task.load_data()
            summary = precompute_task_cache(
                model=model,
                backbone=backbone,
                encode_config=encode_config,
                task=task,
                cache_root=args.cache_root,
                hf_splits=args.split,
                hf_subsets=args.subset,
                batch_size=args.batch_size,
            )
            print(
                f"Precomputed {summary['documents']} documents and {summary['queries']} queries "
                f"for {summary['task']} ({summary['splits']}/{summary['subsets']}), "
                f"{summary['newly_encoded']} newly encoded, in {summary['cache_root']}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
