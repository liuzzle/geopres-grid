#!/usr/bin/env python
"""ESTIMATE of the per-query latency that query expansion adds -- not a measurement
of a serving system (logging.md §71).

Three parts, each timed separately and in isolation; their sum is an online
query's latency only under the assumption that nothing else (network to an index,
batching, queueing, a warm or cold model) matters.

1. LLM expansion: the wall-clock of every recorded API call (`--expansion HASH`),
   p50/p95. Recorded during `expand_queries.py`, from this machine, with the SDK's
   retries inside and concurrent workers beside it, at whatever load OpenAI had.
2. Query encoding (`--backbone`, loads the model -- run it in the backbone's
   environment, on the device you want numbers for): one query at a time
   (batch 1, the online case), original Impresso queries and, with `--expansion`,
   their expanded texts.
3. Search: one query scored against a corpus of the Impresso size (738 documents;
   `--corpus-size` adds others) at every grid dimension and quantizer, plus top-10.
   The evaluation itself fake-quantizes and scores in float32, so for every width
   of 2 bits or more this times a float32 matrix-vector product over dequantized
   vectors (asymmetric distance); `binary` times XOR + popcount over packed bits,
   what a binary index would run. No integer SIMD kernels, no ANN index:
   the multi-bit numbers are an upper bound on what a real int kernel costs, and at
   738 documents every one of them is call overhead, not arithmetic.

    uv run python scripts/estimate_latency.py --expansion <hash> --out latency.json
    uv run --project envs/transformers5 python scripts/estimate_latency.py --backbone mdenseon \\
        --expansion <hash> --device cuda --out latency_mdenseon.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from geopres_grid import config
from geopres_grid.backbones import BACKBONES, PRIMARY_MAX_SEQ_LENGTH
from geopres_grid.expansion import (
    COMBINE_MODES,
    api_latency,
    combine,
    load_expansions,
    percentiles,
    read_queries,
)
from geopres_grid.grid import GRID_QUANT_METHODS, GRID_TARGET_DIMS
from geopres_grid.identity import BITS_PER_DIM

IMPRESSO_CORPUS_SIZE = 738
"""The larger of the two Impresso corpora (DE 738, FR 725)."""


def time_call(function, repeats: int, warmup: int = 3) -> list[float]:
    for _ in range(warmup):
        function()
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        times.append(time.perf_counter() - started)
    return times


def search_kernel(quant_method: str, dim: int, corpus_size: int, rng: np.random.Generator):
    """A zero-argument callable scoring one query and taking its top 10."""
    top = min(10, corpus_size)
    if quant_method == "binary":
        # 64-bit words, as a binary index stores them; every grid and native dim is a multiple of 64.
        documents = np.packbits(rng.random((corpus_size, dim)) > 0.5, axis=1)
        query = np.packbits(rng.random(dim) > 0.5)
        if dim % 64 == 0:
            documents, query = documents.view(np.uint64), query.view(np.uint64)

        def run() -> np.ndarray:
            distances = np.bitwise_count(np.bitwise_xor(documents, query)).sum(axis=1, dtype=np.int32)
            return np.argpartition(distances, top - 1)[:top]

        return run
    documents = rng.standard_normal((corpus_size, dim), dtype=np.float32)
    query = rng.standard_normal(dim, dtype=np.float32)

    def run() -> np.ndarray:
        scores = documents @ query
        return np.argpartition(-scores, top - 1)[:top]

    return run


def search_latency(dims: list[int], quant_methods: list[str], corpus_sizes: list[int], repeats: int) -> list[dict]:
    rng = np.random.default_rng(0)
    rows = []
    for corpus_size in corpus_sizes:
        for dim in dims:
            for quant in quant_methods:
                times = time_call(search_kernel(quant, dim, corpus_size, rng), repeats)
                rows.append(
                    {
                        "corpus_size": corpus_size,
                        "dim": dim,
                        "quant_method": quant,
                        "bits_per_dim": BITS_PER_DIM[quant],
                        "index_bytes": corpus_size * dim * BITS_PER_DIM[quant] // 8,
                        "kernel": "xor+popcount" if quant == "binary" else "float32 matvec (dequantized)",
                        **{f"{key}_ms": value * 1e3 for key, value in percentiles(times).items() if key != "n"},
                    }
                )
    return rows


def encode_latency(backbone_key: str, texts: dict[str, list[str]], device: str | None, repeats_cap: int) -> dict:
    from sentence_transformers import SentenceTransformer

    from geopres_grid.backbones import require_runnable
    from geopres_grid.config import resolve_device

    backbone = BACKBONES[backbone_key]
    require_runnable(backbone)
    device = resolve_device(device)
    model = SentenceTransformer(
        backbone.model_id,
        revision=backbone.revision,
        device=device,
        trust_remote_code=backbone.trust_remote_code,
        model_kwargs={"dtype": "float32"},
    )
    model.max_seq_length = PRIMARY_MAX_SEQ_LENGTH
    prefix = backbone.prompt_for("query", "Retrieval")
    result = {"backbone": backbone_key, "device": device, "batch_size": 1}
    for label, items in texts.items():
        items = items[:repeats_cap]
        for text in items[:3]:  # warm-up: kernels, allocator, tokenizer caches
            model.encode([text], prompt=prefix, batch_size=1)
        times = []
        for text in items:
            if device.startswith("cuda"):
                import torch

                torch.cuda.synchronize()
            started = time.perf_counter()
            model.encode([text], prompt=prefix, batch_size=1, convert_to_numpy=True)
            times.append(time.perf_counter() - started)
        result[label] = {key: (value * 1e3 if key != "n" else value) for key, value in percentiles(times).items()}
        result[label]["unit"] = "ms"
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--expansion", action="append", default=[], metavar="HASH", help="Repeatable")
    parser.add_argument("--expansion-root", default=config.QUERY_EXPANSION_PATH)
    parser.add_argument("--backbone", action="append", choices=sorted(BACKBONES), help="Time query encoding; loads the model")
    parser.add_argument("--device", default=None)
    parser.add_argument("--encode-queries", type=int, default=200, help="Queries timed per text kind")
    parser.add_argument("--corpus-size", action="append", type=int, help=f"Default {IMPRESSO_CORPUS_SIZE}")
    parser.add_argument("--search-repeats", type=int, default=200)
    parser.add_argument("--skip-search", action="store_true")
    parser.add_argument("--impresso-root", default=config.IMPRESSO_PATH)
    parser.add_argument("--out", help="Write the estimate as JSON here too")
    args = parser.parse_args(argv)

    report: dict = {
        "label": "ESTIMATE -- components timed in isolation, not an end-to-end serving measurement",
        "api": {},
        "encode": [],
        "search": [],
    }
    for value in args.expansion:
        report["api"][value] = {"seconds": api_latency(args.expansion_root, value)}

    if args.backbone:
        queries = read_queries(args.impresso_root)
        rng = np.random.default_rng(0)
        sample = [queries[i] for i in sorted(rng.choice(len(queries), min(args.encode_queries, len(queries)), replace=False))]
        texts = {"original": [q.text for q in sample]}
        for value in args.expansion:
            records = load_expansions(args.expansion_root, value)
            for mode in COMBINE_MODES:
                texts[f"{value}-{mode}"] = [
                    combine(q.text, records[q.query_id]["expansion"], mode) for q in sample if q.query_id in records
                ]
        for key in args.backbone:
            report["encode"].append(encode_latency(key, texts, args.device, args.encode_queries))

    if not args.skip_search:
        dims = sorted({*GRID_TARGET_DIMS, *(b.native_dim for b in BACKBONES.values())})
        report["search"] = search_latency(
            dims, list(GRID_QUANT_METHODS), args.corpus_size or [IMPRESSO_CORPUS_SIZE], args.search_repeats
        )

    print(report["label"])
    for value, stats in report["api"].items():
        print(f"API {value}: {json.dumps(stats['seconds'])} s")
    for row in report["encode"]:
        print(f"encode {row['backbone']} on {row['device']}: " + json.dumps({k: v for k, v in row.items() if isinstance(v, dict)}))
    if report["search"]:
        print(f"{'N':>8} {'dim':>5} {'quant':>14} {'p50 us':>9} {'p95 us':>9}")
        for row in report["search"]:
            print(f"{row['corpus_size']:>8} {row['dim']:>5} {row['quant_method']:>14} {row['p50_ms'] * 1e3:>9.1f} {row['p95_ms'] * 1e3:>9.1f}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
