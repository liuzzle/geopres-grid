#!/usr/bin/env python
"""Query-expanded Impresso tasks against their originals: does LLM query expansion
help, and does it recover what compression loses?

    uv run python scripts/analyse_expansion.py storage/impresso_qe_results/results.csv \\
        --out storage/impresso_qe_results/analysis/ndcg_at_10 [--metric recall_at_5]

Reads a `collect_results.py` table holding the original Impresso tasks and their
expanded variants (`ImpressoMultilingualDE__qe-<variant>-<hash>-<combine>`). Run ids are
shared by design (logging §71.1), so every expanded (backbone, run, task) is paired with
the original's; cells scored on one side only are dropped.

Read fp32 first: the fp32 baseline is untouched by compression, so its difference is the
expansion alone. Under compression each task is calibrated on its own inputs, expanded
queries included (§71.1), so a compressed cell's difference mixes the queries with the
refit. `recovered` = expanded compressed score - original fp32 score: >= 0 means the
expanded queries make up for the whole compression loss.

Writes into `--out`: `fp32.csv` (per backbone and task), `cells.csv` (every paired
cell), `reference.csv` (the thesis plan's reference configurations, mean over tasks).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from geopres_grid.expansion import parse_expanded_task_name

REFERENCE_CONFIGS: tuple[str, ...] = (
    "Full/fp16",
    "Full/int4",
    "Full/EC-2",
    "Full/binary",
    "PCA-256/fp32",
    "PCA-128/EC-4",
    "PCA+ROR-256/EC-2",
    "Truncation-128/fp32",
)
"""Thesis plan §3.3: the configurations every "selected configurations" figure shows."""

DR_LABELS = {"none": "Full", "pca": "PCA", "pca_ror": "PCA+ROR", "truncate": "Truncation",
             "random_projection": "RandProj", "random_selection": "RandSel"}
EXPANSION = ["variant", "hash", "combine"]


def config_label(row: pd.Series) -> str:
    reduction = DR_LABELS.get(row.dr_method, row.dr_method)
    if row.dr_method != "none":
        reduction += f"-{int(row.target_dim)}"
    quant = row.quant_method
    quant = "fp32" if quant == "none" else quant.replace("equal_count_", "EC-")
    return f"{reduction}/{quant}"


def paired_cells(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    frame = frame[frame["task"].str.startswith("Impresso")]
    parsed = frame["task"].map(parse_expanded_task_name)
    expanded = frame[parsed.notna()].join(pd.DataFrame(list(parsed.dropna()), index=frame.index[parsed.notna()]))
    original = frame[parsed.isna()].rename(columns={"task": "base"})
    if expanded.empty:
        raise SystemExit("no expanded Impresso task in the table")

    keys = ["backbone", "run_id", "base"]
    config = ["dr_method", "target_dim", "quant_method", "bytes_per_vector", "compression_factor"]
    pairs = expanded[keys + EXPANSION + [metric]].rename(columns={metric: "expanded"}).merge(
        original[keys + config + [metric]].rename(columns={metric: "original"}),
        on=keys,
        how="left",
        validate="many_to_one",
    )
    unmatched = pairs["original"].isna().sum()
    if unmatched:
        print(f"warning: {unmatched} expanded rows have no original counterpart; dropped", file=sys.stderr)
        pairs = pairs.dropna(subset=["original"])
    pairs["config"] = pairs.apply(config_label, axis=1)

    fp32 = pairs[pairs["config"] == "Full/fp32"]
    pairs = pairs.merge(
        fp32[["backbone", "base", *EXPANSION, "original", "expanded"]].rename(
            columns={"original": "original_fp32", "expanded": "expanded_fp32"}
        ),
        on=["backbone", "base", *EXPANSION],
        how="left",
    )
    pairs["delta"] = pairs["expanded"] - pairs["original"]
    pairs["original_retention"] = pairs["original"] / pairs["original_fp32"]
    pairs["expanded_retention"] = pairs["expanded"] / pairs["expanded_fp32"]
    pairs["loss"] = pairs["original_fp32"] - pairs["original"]
    pairs["recovered"] = pairs["expanded"] - pairs["original_fp32"]
    return pairs.sort_values([*EXPANSION, "backbone", "base", "bytes_per_vector"], ascending=[True] * 5 + [False])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path, help="collect_results.py results.csv")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--metric", default="main_score", help="Score column; main_score is nDCG@10 for Impresso")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    cells = paired_cells(pd.read_csv(args.results), args.metric)
    cells.to_csv(args.out / "cells.csv", index=False)

    fp32 = cells[cells["config"] == "Full/fp32"][["backbone", "base", *EXPANSION, "original", "expanded", "delta"]]
    fp32.to_csv(args.out / "fp32.csv", index=False)

    reference = (
        cells[cells["config"].isin(REFERENCE_CONFIGS)]
        .groupby([*EXPANSION, "backbone", "config"], sort=False)
        .agg(
            original=("original", "mean"),
            expanded=("expanded", "mean"),
            delta=("delta", "mean"),
            delta_positive=("delta", lambda d: f"{(d > 0).sum()}/{len(d)}"),
            original_retention=("original_retention", "mean"),
            expanded_retention=("expanded_retention", "mean"),
            loss=("loss", "mean"),
            recovered=("recovered", "mean"),
        )
        .reset_index()
    )
    reference["config"] = pd.Categorical(reference["config"], categories=REFERENCE_CONFIGS, ordered=True)
    reference = reference.sort_values([*EXPANSION, "backbone", "config"])
    reference.to_csv(args.out / "reference.csv", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print(f"{len(cells)} paired cells, metric {args.metric}\n")
    print("fp32, expanded - original:")
    print(fp32.pivot_table(index="backbone", columns="base", values="delta").round(4).assign(
        mean=lambda t: t.mean(axis=1)).round(4), "\n")
    print("Reference configurations, expanded - original (mean over tasks):")
    print(reference.pivot_table(index="config", columns="backbone", values="delta", sort=False).round(4), "\n")
    print("Reference configurations, expanded compressed - original fp32 (>= 0: loss fully recovered):")
    print(reference.pivot_table(index="config", columns="backbone", values="recovered", sort=False).round(4), "\n")
    print("Retention (cell / own fp32), original -> expanded, mean over backbones and tasks:")
    print(reference.groupby("config", sort=False)[["original_retention", "expanded_retention"]].mean().round(4))
    print(f"\nTables in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
