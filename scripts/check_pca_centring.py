#!/usr/bin/env python
"""How much of PCA's loss is the mean shift? Top-10 agreement with fp32 on one task.

`PCA.transform` subtracts the fitted mean before projecting. When embeddings sit in
a narrow cone, cosine on centred vectors is a different similarity, so PCA can lose
to truncation even where it keeps more variance (logging §23.2). This compares, per
dimension and on one task's cached vectors: centred PCA (as the grid runs it), the
same components on uncentred vectors, truncation and random selection.

Reads the cache only; no model, no qrels, CPU. The measure is the share of each
query's fp32 top-10 a reduction keeps -- geometry, not relevance.

    uv run --extra analysis python scripts/check_pca_centring.py --backbone harrier-270m --task NanoArguAnaRetrieval
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from geopres_grid.analysis import centring_check
from geopres_grid.backbones import PRIMARY_MAX_SEQ_LENGTH, get
from geopres_grid.cache import backbone_cache_root, resolve_encode_config
from geopres_grid.config import EMBEDDING_CACHE_PATH, STORAGE_PATH
from geopres_grid.evaluation import cached_blocks
from geopres_grid.grid import GRID_TARGET_DIMS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--split", action="append", help="Restrict to these splits (default: all cached)")
    parser.add_argument("--subset", action="append", help="Restrict to these subsets (default: all cached)")
    parser.add_argument("--max-seq-length", type=int, default=PRIMARY_MAX_SEQ_LENGTH)
    parser.add_argument("--encode-hash", help="Cached encode config to use when several match")
    parser.add_argument("--max-documents", type=int, default=20_000, help="Random document sample cap (full BEIR corpora are millions)")
    parser.add_argument("--cache-root", default=str(EMBEDDING_CACHE_PATH))
    parser.add_argument("--out", type=Path, default=Path(STORAGE_PATH) / "analysis" / "pca_centring")
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    backbone = get(args.backbone)
    if args.encode_hash is None:
        encode_hash = resolve_encode_config(args.cache_root, backbone, max_seq_length=args.max_seq_length).hash
    else:
        # Straight to the directory: resolving would parse every config on disk, and
        # one stale `meta.json` from an older hash scheme would refuse them all.
        encode_hash = args.encode_hash
    directory = backbone_cache_root(args.cache_root, backbone) / encode_hash / args.task
    blocks = cached_blocks(directory, splits=args.split, subsets=args.subset)
    if not blocks["query"] or not blocks["document"]:
        raise SystemExit(f"{directory} needs both a query and a document block (a retrieval task)")
    queries = np.vstack([np.asarray(block, dtype=np.float32) for block in blocks["query"]])
    documents = np.vstack([np.asarray(block, dtype=np.float32) for block in blocks["document"]])
    if len(documents) > args.max_documents:
        chosen = np.sort(np.random.default_rng(0).choice(len(documents), args.max_documents, replace=False))
        documents = documents[chosen]

    dims = [backbone.native_dim] + [d for d in sorted(GRID_TARGET_DIMS, reverse=True) if d < backbone.native_dim]
    table = centring_check(queries, documents, dims)
    name = f"{backbone.key}_{args.task}_{encode_hash}"
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / f"{name}.csv")

    print(f"{backbone.key} {args.task} (encode hash {encode_hash}): "
          f"{len(queries)} queries, {len(documents)} documents, ||mean|| = {table.attrs['mean_norm']:.3f}")
    print(table.round(3).to_string())
    if not args.no_plots:
        from geopres_grid import plots

        figure = plots.centring_check(table, title=f"PCA centring: {backbone.key}, {args.task.removeprefix('Nano').removesuffix('Retrieval')}")
        plots.save(figure, args.out, name)
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
