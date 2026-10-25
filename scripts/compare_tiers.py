#!/usr/bin/env python
"""Tier 1 against tier 2: do configuration rankings transfer, what scale costs, and
whether the small tier would choose the same configuration for a storage budget.

    uv run --extra analysis python scripts/compare_tiers.py \\
        tier1=storage/tier1_results/results.csv tier2=storage/tier2_results/results_beir.csv \\
        --out storage/tier2_results/analysis/tiers

Inputs are `LABEL=results.csv[,more.csv]`, small corpora first (tables read as one by
`analysis.read_results`). Both tiers are cut to the BEIR datasets they share
(`analysis.shared_datasets`: 13 -- tier 2's TRECCOVID and CQADupstack have no NanoBEIR
twin) and scored against their own fp32 baselines. PCA cells are compared only in one
`pca_mean` mode: `--pca-mean` picks it for both tiers; without it, PCA cells are dropped
when the tiers differ (tier 1 `"subtract"`, tier 2 `"restore"` until §35's rerun).

Writes into `--out`: `tier_cells.csv` (every cell in both tiers), `rank_transfer.csv`,
`scale_cost.csv`, `budget_choice.csv`, and with `--extra analysis` the figures
`tier_transfer`, `scale_cost`, `budget_choice` (png + pdf).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from geopres_grid import analysis


def labelled(value: str) -> tuple[str, list[str]]:
    label, sep, paths = value.partition("=")
    if not sep or not label or not paths:
        raise SystemExit(f"expected LABEL=results.csv[,more.csv], got {value!r}")
    return label, paths.split(",")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("small", help="LABEL=results.csv[,more.csv], the small-corpus tier")
    parser.add_argument("large", help="LABEL=results.csv[,more.csv], the large-corpus tier")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pca-mean", choices=("restore", "subtract"), help="PCA mode for both tiers")
    parser.add_argument("--task-type", default="Retrieval")
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    (small, small_paths), (large, large_paths) = labelled(args.small), labelled(args.large)
    args.out.mkdir(parents=True, exist_ok=True)

    rows = {
        label: analysis.scored_rows(analysis.read_results(paths), task_type=args.task_type, pca_mean=args.pca_mean)
        for label, paths in ((small, small_paths), (large, large_paths))
    }
    rows_small, rows_large = analysis.shared_datasets(rows[small], rows[large])
    modes = {
        label: sorted(frame.loc[frame["dr_method"].isin(analysis.PCA_METHODS), "pca_mean"].dropna().unique())
        for label, frame in ((small, rows_small), (large, rows_large))
    }
    if modes[small] and modes[large] and modes[small] != modes[large]:
        print(f"PCA cells dropped: pca_mean {modes[small]} in {small}, {modes[large]} in {large}")
        rows_small = rows_small[~rows_small["dr_method"].isin(analysis.PCA_METHODS)]
        rows_large = rows_large[~rows_large["dr_method"].isin(analysis.PCA_METHODS)]

    paired = analysis.tier_cells(analysis.cell_summary(rows_small), analysis.cell_summary(rows_large))
    tables = {
        "tier_cells": paired,
        "rank_transfer": analysis.rank_transfer(paired),
        "scale_cost": analysis.scale_cost(paired),
        "budget_choice": analysis.budget_choice(paired),
    }
    for name, table in tables.items():
        table.to_csv(args.out / f"{name}.csv", index=not isinstance(table.index, pd.RangeIndex))

    choice = tables["budget_choice"]
    summary = {
        "small": small,
        "large": large,
        "datasets": sorted(rows_large["task"].unique()),
        "cells_per_backbone": paired.groupby("backbone").size().to_dict(),
        "pca_cells": bool(paired["dr_method"].isin(analysis.PCA_METHODS).any()),
        "pca_mean": modes,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.precision", 3):
        print("\nRank transfer (retention, baseline excluded):\n", tables["rank_transfer"])
        print(f"\nRetention {large} - {small} (pts), mean over backbones:\n", tables["scale_cost"].round(1))
        print(f"\nSame pick per budget, regret on {large} (nDCG@10 pts):\n",
              choice.groupby("backbone").agg(same=("same", "mean"), mean_regret=("regret", "mean"),
                                             max_regret=("regret", "max"), budgets=("budget", "size")))

    if not args.no_plots:
        from geopres_grid import plots

        figures = {
            "tier_transfer": plots.tier_transfer(paired, tables["rank_transfer"], small, large),
            "scale_cost": plots.scale_cost(tables["scale_cost"], small, large),
            "budget_choice": plots.budget_choice(choice, small, large),
        }
        for name, figure in figures.items():
            plots.save(figure, args.out, name)
        print(f"\n{len(figures)} figures (png + pdf) -> {args.out}")
    print(f"{len(tables)} tables -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
