#!/usr/bin/env python
"""Analyse a tidy `results.csv` (from `collect_results.py`): tables and figures.

Writes, into `--out` (default: `analysis/` next to the input):

- `cells.csv` -- one row per (backbone, DR, dimension, quantizer), means over tasks
- `baselines.csv`, `quantization_only.csv`, `reduction_only.csv`, `ror_gain.csv`,
  `equal_count_gain.csv`, `prefix_gap.csv`, `budget_frontier.csv`, `task_deltas.csv`
- one PNG + PDF per figure (with `--extra analysis`; `--no-plots` skips them)

and prints a summary. Runs on a laptop in seconds; on the cluster, run it as a job
or with `OPENBLAS_NUM_THREADS=1` -- never bare on the login node (logging §21.4).
See logging §23 for what the tier-1 numbers say.

    uv run --extra analysis python scripts/analyse_results.py storage/tier1_results/results.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from geopres_grid import analysis

TASK_DELTA_CELLS = {
    "full/binary": ("none", 0, "binary"),
    "full/int2": ("none", 0, "int2"),
    "full/EC-2": ("none", 0, "equal_count_2"),
    "pca-128": ("pca", 128, "none"),
    "pca-64": ("pca", 64, "none"),
    "truncate-128": ("truncate", 128, "none"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path, help="Tidy results.csv from collect_results.py")
    parser.add_argument("--out", type=Path, help="Output directory (default: <results dir>/analysis)")
    parser.add_argument("--task-type", default="Retrieval")
    parser.add_argument("--no-plots", action="store_true", help="Tables only; no matplotlib needed")
    parser.add_argument(
        "--pca-mean",
        choices=("restore", "subtract"),
        help="Which PCA cells to analyse when the table holds both (logging §29)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = args.out or args.results.parent / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.results)
    rows = analysis.scored_rows(frame, task_type=args.task_type, pca_mean=args.pca_mean)
    other = frame[frame["task_type"] != args.task_type]
    cells = analysis.cell_summary(rows)

    tables = {
        "cells": cells,
        "baselines": analysis.baselines(rows),
        "quantization_only": analysis.quantization_only(cells),
        "reduction_only": analysis.reduction_only(cells),
        "ror_gain": analysis.ror_gain(cells),
        "equal_count_gain": analysis.equal_count_gain(cells),
        "prefix_gap": analysis.prefix_gap(cells),
        "budget_frontier": analysis.budget_frontier(cells),
        "task_deltas": analysis.task_deltas(rows, TASK_DELTA_CELLS),
    }
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
    }
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

    if not args.no_plots:
        from geopres_grid import plots

        figures = {
            "quantization_only": plots.quantization_only(tables["quantization_only"]),
            "reduction_only": plots.reduction_only(tables["reduction_only"]),
            "budget_frontier": plots.budget_frontier(cells, tables["budget_frontier"]),
            "ror_gain": plots.ror_gain(tables["ror_gain"]),
            "equal_count_gain": plots.equal_count_gain(tables["equal_count_gain"]),
            "prefix_gap": plots.prefix_gap(tables["prefix_gap"]),
            "task_deltas": plots.task_deltas(tables["task_deltas"]),
        }
        for name, figure in figures.items():
            plots.save(figure, out, name)
        print(f"\n{len(figures)} figures (png + pdf) -> {out}")
    print(f"{len(tables)} tables -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
