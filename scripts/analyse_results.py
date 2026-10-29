#!/usr/bin/env python
"""Analyse a tidy `results.csv` (from `collect_results.py`): tables and figures.

Writes, into `--out` (default: `analysis/` next to the input):

- `cells.csv` -- one row per (backbone, DR, dimension, quantizer), means over tasks
- `baselines.csv`, `quantization_only.csv`, `reduction_only.csv`, `ror_gain.csv`,
  `equal_count_gain.csv`, `prefix_gap.csv`, `budget_frontier.csv`, `task_deltas.csv`
- the absolute twins `quantization_only_score.csv`, `reduction_only_score.csv`,
  `task_scores.csv` (mean nDCG@10; the gain tables carry `better_score`/`worse_score`)
- `task_losses.csv` -- per backbone, task and representative cell: baseline, score,
  delta, retention; `corpus_sizes.csv` (MTEB descriptive statistics) and
  `corpus_size_correlation.csv` -- each cell's per-task retention against log corpus size
- one PNG + PDF per figure (with `--extra analysis`; `--no-plots` skips them),
  `*_by_backbone` ones facetting what the others average over backbones

and prints a summary. Runs on a laptop in seconds; on the cluster, run it as a job
or with `OPENBLAS_NUM_THREADS=1` -- never bare on the login node (logging §21.4).
See logging §23 for what the tier-1 numbers say.

    uv run --extra analysis python scripts/analyse_results.py storage/tier1_results/results.csv

Several tables are read as one (a rerun next to the run it extends, e.g. tier 1's
restore PCA cells: `results.csv results_restore.csv --pca-mean restore`); a row
present in two of them (same run, task, split, subset) is kept once.

`--tier` keeps one tier's tasks, which a table collected over every tier needs
(`--tier impresso`), and `--metric recall_at_5` analyses that column in place of
the main score (collect it with `collect_results.py --metric recall_at_5`). The
corpus-size tables and the figures are skipped for the Impresso tier (four custom
tasks, no MTEB statistics) and for any metric but nDCG@10 (the figures are labelled
nDCG@10).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from geopres_grid import analysis
from geopres_grid.evaluation import TIER_TASKS
from geopres_grid.results import add_task_metadata

TASK_DELTA_CELLS = analysis.REPRESENTATIVE_CELLS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path, nargs="+", help="Tidy results.csv from collect_results.py")
    parser.add_argument("--out", type=Path, help="Output directory (default: <first results dir>/analysis)")
    parser.add_argument("--task-type", default="Retrieval")
    parser.add_argument("--tier", choices=sorted(TIER_TASKS), help="Keep only this tier's tasks")
    parser.add_argument(
        "--metric",
        help="Analyse this column of the table (e.g. recall_at_5) in place of the main score (nDCG@10)",
    )
    parser.add_argument("--no-plots", action="store_true", help="Tables only; no matplotlib needed")
    parser.add_argument(
        "--pca-mean",
        choices=("restore", "subtract"),
        help="Which PCA cells to analyse when the table holds both (logging §29)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = args.out or args.results[0].parent / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    frame = analysis.read_results(args.results)
    if args.tier:
        frame = frame[frame["task"].isin(TIER_TASKS[args.tier])]
        if frame["task_type"].isna().any():  # collected before the Impresso tasks were known
            frame = add_task_metadata(frame.drop(columns=["task_type", "main_score_name"]))
    if args.metric:
        if args.metric not in frame.columns:
            raise SystemExit(f"no column {args.metric!r}; collect the table with --metric {args.metric}")
        frame = frame.assign(main_score=frame[args.metric]).dropna(subset=["main_score"])
    with_sizes = args.tier != "impresso"
    with_plots = not args.no_plots and not args.metric
    rows = analysis.scored_rows(frame, task_type=args.task_type, pca_mean=args.pca_mean)
    other = frame[frame["task_type"] != args.task_type]
    cells = analysis.cell_summary(rows)
    losses = analysis.task_losses(rows, TASK_DELTA_CELLS)
    sizes = analysis.corpus_sizes(analysis.task_splits(rows)) if with_sizes else None

    tables = {
        "cells": cells,
        "baselines": analysis.baselines(rows),
        "quantization_only": analysis.quantization_only(cells),
        "quantization_only_score": analysis.quantization_only(cells, "score"),
        "reduction_only": analysis.reduction_only(cells),
        "reduction_only_score": analysis.reduction_only(cells, "score"),
        "all_cells": analysis.all_cells(cells),
        "all_cells_score": analysis.all_cells(cells, "score"),
        "ror_gain": analysis.ror_gain(cells),
        "equal_count_gain": analysis.equal_count_gain(cells),
        "prefix_gap": analysis.prefix_gap(cells),
        "budget_frontier": analysis.budget_frontier(cells),
        "task_deltas": analysis.task_deltas(rows, TASK_DELTA_CELLS),
        "task_scores": analysis.task_scores(rows, TASK_DELTA_CELLS),
        "task_losses": losses,
    }
    if with_sizes:
        tables["corpus_sizes"] = sizes
        tables["corpus_size_correlation"] = analysis.size_correlation(losses, sizes)
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=not isinstance(table.index, pd.RangeIndex))

    noise = analysis.noise_floor(rows)
    noise_4bit = analysis.noise_floor(rows, ("int4", "equal_count_4"))
    incomplete = cells[cells["n_tasks"] < rows["task"].nunique()]
    summary = {
        "rows": len(rows),
        "runs": int(rows["run_id"].nunique()),
        "tasks": int(rows["task"].nunique()),
        "backbones": analysis.backbone_order(rows),
        "skipped_other_task_types": int(len(other)),
        "incomplete_cells": int(len(incomplete)),
        "noise_fp16_int8": noise,
        "noise_int4_ec4": noise_4bit,
        "cells_above_baseline_by_0.5pts": int((cells["delta"] > 0.005).sum()),
        "pca_mean": sorted(rows.loc[rows["dr_method"].isin(analysis.PCA_METHODS), "pca_mean"].dropna().unique()),
    }
    if with_sizes:
        summary["corpus_documents"] = [float(sizes["n_documents"].min()), float(sizes["n_documents"].max())]
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.precision", 3):
        print("\nBaselines (mean):", tables["baselines"].loc["mean"].round(4).to_dict())
        print("\nQuantization only (retention):\n", tables["quantization_only"])
        print("\nPCA+ROR - PCA (pts), mean over backbones:\n",
              tables["ror_gain"].pivot_table(index="quant_method", columns="target_dim", values="gain").round(1))
        print("\nEqual-count - int (pts), mean over backbones and dims:\n",
              tables["equal_count_gain"].pivot_table(index="dr_method", columns="bits", values="gain").round(1))
        print("\nTruncate - random selection, fp32 (pts):\n",
              tables["prefix_gap"].pivot_table(index="backbone", columns="target_dim", values="gain").round(1))
        if with_sizes:
            correlation = tables["corpus_size_correlation"]
            print("\nRetention vs log10 corpus size, Spearman (mean over backbones, then per backbone):\n",
                  correlation.pivot_table(index="config", columns="backbone", values="spearman").round(2))

    if with_plots:
        from geopres_grid import plots

        figures = {
            "quantization_only": plots.quantization_only(tables["quantization_only"], tables["quantization_only_score"]),
            "reduction_only": plots.reduction_only(tables["reduction_only"], tables["reduction_only_score"]),
            "all_cells": plots.all_cells(tables["all_cells"], tables["all_cells_score"]),
            "all_cells_by_backbone": plots.all_cells_by_backbone(tables["all_cells"], tables["all_cells_score"]),
            "budget_frontier": plots.budget_frontier(cells, tables["budget_frontier"]),
            "budget_frontier_by_backbone": plots.budget_frontier_by_backbone(cells, tables["budget_frontier"]),
            "ror_gain": plots.ror_gain(tables["ror_gain"]),
            "ror_gain_by_backbone": plots.ror_gain_by_backbone(tables["ror_gain"]),
            "equal_count_gain": plots.equal_count_gain(tables["equal_count_gain"]),
            "equal_count_gain_by_backbone": plots.equal_count_gain_by_backbone(tables["equal_count_gain"]),
            "prefix_gap": plots.prefix_gap(tables["prefix_gap"]),
            "task_deltas": plots.task_deltas(tables["task_deltas"], tables["task_scores"]["fp32"]),
            "task_deltas_by_backbone": plots.task_deltas_by_backbone(losses, list(TASK_DELTA_CELLS)),
        }
        if with_sizes:
            figures["corpus_size_loss"] = plots.corpus_size_loss(losses, sizes, tables["corpus_size_correlation"])
        for name, figure in figures.items():
            plots.save(figure, out, name)
        split = plots.all_cells_per_backbone(tables["all_cells"], tables["all_cells_score"])
        for backbone, figure in split.items():
            plots.save(figure, out / "all_cells_per_backbone", backbone)
        print(f"\n{len(figures)} figures (png + pdf) -> {out}, {len(split)} more in all_cells_per_backbone/")
    print(f"{len(tables)} tables -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
