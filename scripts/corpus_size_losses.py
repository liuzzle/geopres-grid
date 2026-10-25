#!/usr/bin/env python
"""Per-task losses of representative cells against corpus size, tiers side by side.

Each input is `LABEL=results.csv[,more.csv]` (tidy tables, read as one by
`analysis.read_results`, as in `analyse_results.py`); every
tier is scored against its own fp32 baselines, and corpus sizes come from MTEB's
packaged descriptive statistics (`analysis.corpus_sizes`, offline). NanoBEIR corpora
are all 2-6k passages, so the size axis needs tier 2's 3.6k-8.8M to say anything.

A PCA cell is only compared across tiers in one `pca_mean` mode: `--pca-mean
LABEL=MODE` picks it per tier, and a PCA cell whose tiers end up in different modes
is dropped from the combined tables and figure (logging §29; it would correlate two
different reductions as one). Writes into `--out`:

- `task_losses.csv` -- tier, backbone, task, cell: baseline, score, delta, retention
- `corpus_sizes.csv` -- tier, task, split, documents, queries
- `corpus_size_correlation.csv` -- Spearman / Pearson(log10) of retention vs corpus
  size per cell, per tier and pooled (`tier` empty), task means and per backbone
- with exactly two inputs (small corpora first): `paired_tiers.csv` and
  `paired_correlation.csv` -- the same BEIR dataset in both (NanoArguAna vs ArguAna),
  retention in the second minus the first against the second's corpus size. Unlike
  the pooled correlation, this compares a dataset with itself.
- `corpus_size_loss.png` / `.pdf` and `corpus_size_paired.png` / `.pdf` (with
  `--extra analysis`)

    uv run --extra analysis python scripts/corpus_size_losses.py \\
        tier1=storage/tier1_results/results.csv tier2=storage/tier2_results/results_beir.csv \\
        --out storage/tier2_results/analysis/corpus_size
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from geopres_grid import analysis


def pairs(values: list[str], what: str) -> dict[str, str]:
    parsed = {}
    for value in values:
        label, sep, rest = value.partition("=")
        if not sep or not label or not rest:
            raise SystemExit(f"{what} must be LABEL=VALUE, got {value!r}")
        parsed[label] = rest
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", help="LABEL=results.csv[,more.csv], one per tier")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pca-mean", action="append", default=[], help="LABEL=restore|subtract (repeatable)")
    parser.add_argument("--task-type", default="Retrieval")
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    inputs = pairs(args.inputs, "input")
    modes = pairs(args.pca_mean, "--pca-mean")
    args.out.mkdir(parents=True, exist_ok=True)

    losses, sizes = [], []
    for label, path in inputs.items():
        frame = analysis.read_results(path.split(","))
        rows = analysis.scored_rows(frame, task_type=args.task_type, pca_mean=modes.get(label))
        losses.append(analysis.task_losses(rows, analysis.REPRESENTATIVE_CELLS).assign(tier=label))
        sizes.append(analysis.corpus_sizes(analysis.task_splits(rows)).assign(tier=label))
    losses = pd.concat(losses, ignore_index=True)
    sizes = pd.concat(sizes).reset_index()

    pca_modes = losses.dropna(subset=["pca_mean"]).groupby("config")["pca_mean"].nunique()
    mixed = sorted(pca_modes[pca_modes > 1].index)
    if mixed:
        print(f"Dropped, PCA cells in different pca_mean modes across tiers: {mixed}")
        losses = losses[~losses["config"].isin(mixed)]

    correlation = pd.concat(
        [
            analysis.size_correlation(losses, sizes, groups=("tier", "config")),
            analysis.size_correlation(losses.drop(columns="tier"), sizes.drop(columns="tier").drop_duplicates("task"))
            .assign(tier=None),
        ],
        ignore_index=True,
    )
    losses.to_csv(args.out / "task_losses.csv", index=False)
    sizes.to_csv(args.out / "corpus_sizes.csv", index=False)
    correlation.to_csv(args.out / "corpus_size_correlation.csv", index=False)

    paired = paired_correlation = None
    if len(inputs) == 2:
        small, large = inputs
        paired = analysis.paired_tiers(losses, small, large)
        large_sizes = sizes[sizes["tier"] == large].drop(columns="tier").set_index("task")
        paired_correlation = analysis.size_correlation(paired, large_sizes, value="retention_change")
        paired.to_csv(args.out / "paired_tiers.csv", index=False)
        paired_correlation.to_csv(args.out / "paired_correlation.csv", index=False)

    means = correlation[correlation["backbone"] == "mean"].assign(tier=lambda f: f["tier"].fillna("pooled"))
    with pd.option_context("display.width", 200, "display.precision", 3):
        print("Retention vs log10 corpus size, Spearman on task means over backbones:\n",
              means.pivot_table(index="config", columns="tier", values="spearman").round(2))
        print("\np-values:\n", means.pivot_table(index="config", columns="tier", values="spearman_p").round(4))
        if paired is not None:
            print(f"\nPaired, same dataset: retention {large} - {small}, mean over backbones and datasets:\n",
                  paired.groupby("config")["retention_change"].mean().round(3))
            print("\nPaired change vs log10 corpus size, Spearman (mean, then per backbone):\n",
                  paired_correlation.pivot_table(index="config", columns="backbone", values="spearman").round(2))

    if not args.no_plots:
        from geopres_grid import plots

        plots.save(plots.corpus_size_loss(losses, sizes, correlation), args.out, "corpus_size_loss")
        if paired is not None:
            figure = plots.corpus_size_loss(
                paired, large_sizes, paired_correlation, value="retention_change",
                ylabel=f"Retention, {large} - {small}",
            )
            plots.save(figure, args.out, "corpus_size_paired")
        print(f"\nfigure (png + pdf) -> {args.out}")
    print(f"{3 if paired is None else 5} tables -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
