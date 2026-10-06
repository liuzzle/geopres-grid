#!/usr/bin/env python
"""Collect MTEB results into `results.csv` (tidy) and `comparison_results.csv` (wide).

The wide table is upstream GeoPres's `comparison_results.csv` layout -- a row per
run, a column per task, per-task-type averages -- but keyed by configuration
columns instead of a model-name string. `--upstream` adds an upstream GeoPres
`evaluation_results` tree to both tables, marked `source=geopres-upstream`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from geopres_grid.evaluation import task_names
from geopres_grid.results import comparison_table, load_results, load_upstream_results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, help="default: $EVALUATION_RESULTS_PATH")
    parser.add_argument("--tier", choices=["tier0", "tier1", "tier2"], help="Restrict the wide table to one tier's tasks")
    parser.add_argument("--metric", action="append", default=[], help="Extra metric column for the tidy table, e.g. recall_at_100")
    parser.add_argument("--upstream", type=Path, help="Upstream GeoPres evaluation_results directory")
    args = parser.parse_args(argv)

    if args.results_root is None:
        from geopres_grid.config import EVALUATION_RESULTS_PATH

        args.results_root = Path(EVALUATION_RESULTS_PATH)
    frame = load_results(args.results_root, metrics=tuple(args.metric))
    if args.upstream is not None:
        frame = pd.concat([frame, load_upstream_results(args.upstream)], ignore_index=True)

    tidy_path = args.results_root / "results.csv"
    wide_path = args.results_root / "comparison_results.csv"
    frame.to_csv(tidy_path, index=False)
    wide = comparison_table(frame, tasks=list(task_names(args.tier)) if args.tier else None)
    wide.to_csv(wide_path)

    print(f"{len(frame)} rows from {frame['run_id'].nunique()} runs -> {tidy_path}")
    print(f"{len(wide)} runs x {wide.shape[1]} columns -> {wide_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
