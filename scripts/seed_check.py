#!/usr/bin/env python
"""Seed sanity check: does the reducer seed move the grid's results? (logging §37)

Input is a tidy `results.csv` (`collect_results.py`) holding the seeded methods'
cells under several `dr_seed`s (`run_grid.py --dr-seed`); the default seed's cells
are the ones every earlier grid scored, stored with an empty `dr_seed`. One tier's
tasks are used (default tier 1). PCA cells are left out unless `--pca-mean` picks
one mode, as in `analyse_results.py`.

Writes, into `--out` (default: `analysis/` next to the input):

- `seed_cells.csv` -- one row per (cell, seed): retention and score, means over tasks
- `seed_spread.csv` -- one row per seeded cell: mean, std, range of retention over
  seeds, and where the default seed's value sits among them
- `comparisons.csv` -- method differences the thesis reports (random projection vs
  random selection, truncation vs both, equal-count vs uniform int), per seed and
  averaged, with how often a single seed gets the sign wrong
- `ranking.csv` -- per backbone, the cell ranking under the default seed against the
  ranking under the seed mean; `winners.csv` -- the best DR method per (dimension,
  quantizer) under each; `frontier.csv` -- budget-frontier cells under each
- `summary.json`, and a printed summary

Retention is in points (x100) in the comparisons and spreads.

    uv run python scripts/seed_check.py storage/seed_check/results.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from geopres_grid import analysis
from geopres_grid.evaluation import TIER_TASKS, task_names
from geopres_grid.identity import DEFAULT_DR_SEED, PCA_METHODS, SEEDED_DR_METHODS

CELL_KEYS = list(analysis.CELL_KEYS)
SIZE_KEYS = ["bits_per_dim", "bytes_per_vector", "compression_factor"]

COMPARISONS: tuple[tuple[str, str, str, str], ...] = (
    ("rp_vs_rs", "dr_method", "random_projection", "random_selection"),
    ("truncate_vs_rs", "dr_method", "truncate", "random_selection"),
    ("truncate_vs_rp", "dr_method", "truncate", "random_projection"),
    ("ec8_vs_int8", "quant_method", "equal_count_8", "int8"),
    ("ec4_vs_int4", "quant_method", "equal_count_4", "int4"),
    ("ec2_vs_int2", "quant_method", "equal_count_2", "int2"),
)
"""(name, column that differs, better, worse): a gain is `better - worse`, matched
on every other cell key. The quantizer comparisons are restricted to seeded methods."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path, help="Tidy results.csv from collect_results.py")
    parser.add_argument("--out", type=Path, help="Output directory (default: <results dir>/analysis)")
    parser.add_argument("--tier", choices=sorted(TIER_TASKS), default="tier1")
    parser.add_argument("--task-type", default="Retrieval")
    parser.add_argument("--pca-mean", choices=("restore", "subtract"), help="Keep the PCA cells of this mode")
    return parser


def seeded_rows(frame: pd.DataFrame, tier: str, task_type: str = "Retrieval", pca_mean: str | None = None) -> pd.DataFrame:
    """`analysis.scored_rows` for one tier's tasks, with `dr_seed` resolved: the
    default seed's rows carry `DEFAULT_DR_SEED` instead of an empty seed, every
    other method too (it has one transform, so one seed)."""
    frame = frame[frame["task"].isin(task_names(tier))]
    if pca_mean is None:
        frame = frame[~frame["dr_method"].isin(PCA_METHODS)]
    rows = analysis.scored_rows(frame, task_type=task_type, pca_mean=pca_mean)
    rows["dr_seed"] = rows["dr_seed"].fillna(DEFAULT_DR_SEED).astype(int)
    return rows


def per_seed_cells(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per (cell, seed): means over tasks, as `analysis.cell_summary`."""
    return (
        rows.groupby(CELL_KEYS + SIZE_KEYS + ["dr_seed"], dropna=False)
        .agg(score=("main_score", "mean"), retention=("retention", "mean"), n_tasks=("task", "nunique"))
        .reset_index()
    )


def seed_mean_cells(per_seed: pd.DataFrame) -> pd.DataFrame:
    """One row per cell, averaged over its seeds; deterministic cells are unchanged."""
    cells = per_seed.groupby(CELL_KEYS + SIZE_KEYS, dropna=False)[["score", "retention"]].mean().reset_index()
    cells["label"] = [analysis.cell_label(*v) for v in cells[["dr_method", "target_dim", "quant_method"]].itertuples(index=False)]
    return cells


def default_seed_cells(per_seed: pd.DataFrame) -> pd.DataFrame:
    """One row per cell under the default seed: what a single-seed grid reports."""
    cells = per_seed[per_seed["dr_seed"] == DEFAULT_DR_SEED].drop(columns=["dr_seed", "n_tasks"]).reset_index(drop=True)
    cells["label"] = [analysis.cell_label(*v) for v in cells[["dr_method", "target_dim", "quant_method"]].itertuples(index=False)]
    return cells


def seed_spread(per_seed: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Per seeded cell: spread of retention over seeds (points), and the default seed's place.

    `std` is the sample std of the task-mean retention over seeds (what a single
    seed's result scatters by); `sem` its standard error of the mean. `task_std` is
    the median over tasks of the per-task std over seeds -- single 50-query
    NanoBEIR tasks move more than their mean. `default_rank` is 1 when the default
    seed scores best of all seeds.
    """
    seeded = per_seed[per_seed["dr_method"].isin(SEEDED_DR_METHODS)].copy()
    seeded["retention"] *= 100
    grouped = seeded.groupby(CELL_KEYS)["retention"]
    spread = (
        seeded.groupby(CELL_KEYS)
        .agg(
            n_seeds=("retention", "count"),
            mean=("retention", "mean"),
            std=("retention", "std"),
            min=("retention", "min"),
            max=("retention", "max"),
            n_tasks_min=("n_tasks", "min"),
        )
        .reset_index()
    )
    spread["range"] = spread["max"] - spread["min"]
    spread["sem"] = spread["std"] / np.sqrt(spread["n_seeds"])
    default = seeded[seeded["dr_seed"] == DEFAULT_DR_SEED].set_index(CELL_KEYS)["retention"].rename("default")
    spread = spread.join(default, on=CELL_KEYS)
    spread["default_minus_mean"] = spread["default"] - spread["mean"]
    spread["default_z"] = spread["default_minus_mean"] / spread["std"]
    seeded["rank"] = grouped.rank(ascending=False, method="min")
    rank = seeded[seeded["dr_seed"] == DEFAULT_DR_SEED].set_index(CELL_KEYS)["rank"].rename("default_rank")
    spread = spread.join(rank, on=CELL_KEYS)
    per_task = rows[rows["dr_method"].isin(SEEDED_DR_METHODS)].groupby(CELL_KEYS + ["task"])["retention"].std() * 100
    return spread.join(per_task.groupby(CELL_KEYS).median().rename("task_std"), on=CELL_KEYS)


def seeded_gain(per_seed: pd.DataFrame, name: str, column: str, better: str, worse: str) -> pd.DataFrame:
    """`better - worse` in points per seed, then summarised over seeds.

    A side with a single seed (a deterministic method) is compared with every seed
    of the other. Seeds pair by number; for the DR comparisons they are unrelated
    draws anyway, for the quantizer comparisons both cells share the transform.
    `sign_flips` counts the seeds whose gain has the opposite sign of the mean gain;
    `default_flips` is True when the default seed alone gets the sign wrong.
    """
    keys = [key for key in CELL_KEYS if key != column]
    frame = per_seed
    if column == "quant_method":
        frame = frame[frame["dr_method"].isin(SEEDED_DR_METHODS)]

    def side(value: str) -> pd.DataFrame:
        part = frame[frame[column] == value]
        return part.pivot_table(index=keys, columns="dr_seed", values="retention") * 100

    left, right = side(better), side(worse)
    if left.empty or right.empty:
        return pd.DataFrame()
    if left.shape[1] == 1 and right.shape[1] > 1:
        left = pd.DataFrame({seed: left.iloc[:, 0] for seed in right.columns})
    if right.shape[1] == 1 and left.shape[1] > 1:
        right = pd.DataFrame({seed: right.iloc[:, 0] for seed in left.columns})
    seeds = left.columns.intersection(right.columns)
    gain = (left[seeds] - right[seeds]).dropna(how="all")
    mean = gain.mean(axis=1)
    out = pd.DataFrame(
        {
            "comparison": name,
            "better": better,
            "worse": worse,
            "n_seeds": gain.count(axis=1),
            "gain_mean": mean,
            "gain_default": gain.get(DEFAULT_DR_SEED),
            "gain_std": gain.std(axis=1),
            "gain_min": gain.min(axis=1),
            "gain_max": gain.max(axis=1),
            "sign_flips": (np.sign(gain).ne(np.sign(mean), axis=0) & gain.notna()).sum(axis=1),
        }
    )
    out["default_flips"] = np.sign(out["gain_default"]) != np.sign(out["gain_mean"])
    out["resolved"] = out["gain_mean"].abs() > 2 * out["gain_std"] / np.sqrt(out["n_seeds"])
    return out.reset_index()


def comparisons(per_seed: pd.DataFrame) -> pd.DataFrame:
    parts = [seeded_gain(per_seed, *comparison) for comparison in COMPARISONS]
    return pd.concat([part for part in parts if not part.empty], ignore_index=True)


def ranking(default: pd.DataFrame, mean: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    """Per backbone, the cell ranking by retention under the default seed vs the seed mean.

    Over all cells (the deterministic ones keep their score, so they anchor the
    ranking) and over the seeded cells alone. `max_shift` is the largest move of
    one cell in rank; `top_overlap` the share of the top `top` kept.
    """
    joined = default.merge(mean, on=CELL_KEYS, suffixes=("_default", "_mean"), validate="one_to_one")
    records = []
    for backbone, group in joined.groupby("backbone"):
        for scope, part in (("all", group), ("seeded", group[group["dr_method"].isin(SEEDED_DR_METHODS)])):
            rank_default = part["retention_default"].rank(ascending=False)
            rank_mean = part["retention_mean"].rank(ascending=False)
            k = min(top, len(part))
            top_default = set(part.nlargest(k, "retention_default")["label_default"])
            top_mean = set(part.nlargest(k, "retention_mean")["label_default"])
            records.append(
                {
                    "backbone": backbone,
                    "scope": scope,
                    "n_cells": len(part),
                    "spearman": rank_default.corr(rank_mean),
                    "max_shift": int((rank_default - rank_mean).abs().max()),
                    "mean_shift": float((rank_default - rank_mean).abs().mean()),
                    "top_overlap": len(top_default & top_mean) / k,
                }
            )
    return pd.DataFrame(records)


def winners(default: pd.DataFrame, mean: pd.DataFrame) -> pd.DataFrame:
    """Best DR method per (backbone, dimension, quantizer), under each seed choice."""
    keys = ["backbone", "target_dim", "quant_method"]

    def best(cells: pd.DataFrame) -> pd.Series:
        reduced = cells[cells["dr_method"] != "none"]
        return reduced.loc[reduced.groupby(keys)["retention"].idxmax()].set_index(keys)["dr_method"]

    table = pd.DataFrame({"winner_default": best(default), "winner_mean": best(mean)}).reset_index()
    table["changed"] = table["winner_default"] != table["winner_mean"]
    return table


def frontier(default: pd.DataFrame, mean: pd.DataFrame) -> pd.DataFrame:
    """Budget-frontier cells (`analysis.budget_frontier`) under each seed choice."""
    parts = []
    for name, cells in (("default", default), ("mean", mean)):
        front = analysis.budget_frontier(cells)[["backbone", "label", "bytes_per_vector", "score"]]
        front["seeds"] = name
        parts.append(front)
    table = pd.concat(parts, ignore_index=True)
    table["in_both"] = table.duplicated(["backbone", "label"], keep=False)
    return table


def noise_floors(rows: pd.DataFrame, default: pd.DataFrame) -> dict[str, float]:
    """The near-lossless reference, in points: per task (`analysis.noise_floor`) and
    per cell (|fp16 - int8| task-mean retention at the same reduction)."""
    task = analysis.noise_floor(rows)
    pair = default[default["quant_method"].isin(["fp16", "int8"])].pivot_table(
        index=["backbone", "dr_method", "target_dim"], columns="quant_method", values="retention"
    )
    cell = ((pair["fp16"] - pair["int8"]).abs() * 100).dropna()
    return {
        "task_max_abs_delta": task["max_abs_delta"] * 100,
        "task_q95_abs_delta": task["q95_abs_delta"] * 100,
        "cell_fp16_int8_max": float(cell.max()),
        "cell_fp16_int8_median": float(cell.median()),
    }


def summarise(
    seeds: list[int],
    spread: pd.DataFrame,
    gains: pd.DataFrame,
    ranks: pd.DataFrame,
    wins: pd.DataFrame,
    front: pd.DataFrame,
    floors: dict[str, float],
) -> dict:
    by_method = spread.groupby("dr_method")["std"].agg(["median", "max"]).round(3).to_dict("index")
    by_quant = spread.groupby("quant_method")["std"].median().round(3).to_dict()
    by_dim = spread.groupby("target_dim")["std"].median().round(3).to_dict()
    material = gains[gains["gain_mean"].abs() >= 1.0]
    return {
        "seeds": [int(seed) for seed in seeds],
        "n_seeded_cells": int(len(spread)),
        "n_seeds_min": int(spread["n_seeds"].min()),
        "n_tasks_min": int(spread["n_tasks_min"].min()),
        "std_points": {
            "median": float(spread["std"].median()),
            "q90": float(spread["std"].quantile(0.9)),
            "max": float(spread["std"].max()),
            "by_method": by_method,
            "by_quant_median": by_quant,
            "by_dim_median": {int(k): v for k, v in by_dim.items()},
        },
        "range_points_max": float(spread["range"].max()),
        "task_std_points_median": float(spread["task_std"].median()),
        "default_seed": {
            "abs_minus_mean_median": float(spread["default_minus_mean"].abs().median()),
            "abs_minus_mean_max": float(spread["default_minus_mean"].abs().max()),
            "abs_z_median": float(spread["default_z"].abs().median()),
            "share_abs_z_over_2": float((spread["default_z"].abs() > 2).mean()),
            "mean_signed_minus_mean": float(spread["default_minus_mean"].mean()),
            "rank_mean": float(spread["default_rank"].mean()),
        },
        "noise_floor_points": floors,
        "comparisons": {
            name: {
                "n": int(len(group)),
                "abs_gain_mean_median": float(group["gain_mean"].abs().median()),
                "gain_std_median": float(group["gain_std"].median()),
                "default_flips": int(group["default_flips"].sum()),
                "default_flips_if_abs_gain_ge_1pt": int(group.loc[group["gain_mean"].abs() >= 1.0, "default_flips"].sum()),
                "unresolved": int((~group["resolved"]).sum()),
            }
            for name, group in gains.groupby("comparison")
        },
        "material_comparisons_flipped_by_default_seed": int(material["default_flips"].sum()),
        "ranking": ranks.round(4).to_dict("records"),
        "winners_changed": int(wins["changed"].sum()),
        "winners_total": int(len(wins)),
        "frontier_cells_not_in_both": int((~front["in_both"]).sum()),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = args.out or args.results.parent / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    rows = seeded_rows(pd.read_csv(args.results, low_memory=False), args.tier, args.task_type, args.pca_mean)
    per_seed = per_seed_cells(rows)
    seeds = sorted(per_seed.loc[per_seed["dr_method"].isin(SEEDED_DR_METHODS), "dr_seed"].unique())
    if len(seeds) < 2:
        print(f"Only seed(s) {seeds} in {args.results}; nothing to compare", file=sys.stderr)
        return 1
    default, mean = default_seed_cells(per_seed), seed_mean_cells(per_seed)
    spread = seed_spread(per_seed, rows)
    gains = comparisons(per_seed)
    ranks, wins, front = ranking(default, mean), winners(default, mean), frontier(default, mean)
    floors = noise_floors(rows, default)

    per_seed.to_csv(out / "seed_cells.csv", index=False)
    spread.to_csv(out / "seed_spread.csv", index=False)
    gains.to_csv(out / "comparisons.csv", index=False)
    ranks.to_csv(out / "ranking.csv", index=False)
    wins.to_csv(out / "winners.csv", index=False)
    front.to_csv(out / "frontier.csv", index=False)
    summary = summarise(seeds, spread, gains, ranks, wins, front, floors)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    std = summary["std_points"]
    print(f"{summary['n_seeded_cells']} seeded cells, seeds {summary['seeds']}, >= {summary['n_seeds_min']} seeds "
          f"and >= {summary['n_tasks_min']} tasks per cell")
    print(f"Std of task-mean retention over seeds (points): median {std['median']:.3f}, 90th pct {std['q90']:.3f}, "
          f"max {std['max']:.3f}; per single task, median {summary['task_std_points_median']:.3f}")
    for method, values in std["by_method"].items():
        print(f"  {method}: median {values['median']:.3f}, max {values['max']:.3f}")
    print(f"  by dim (median): {std['by_dim_median']}")
    print(f"  by quantizer (median): {std['by_quant_median']}")
    print("Noise floor (points): " + ", ".join(f"{k} {v:.3f}" for k, v in floors.items()))
    d = summary["default_seed"]
    print(f"Default seed {DEFAULT_DR_SEED} vs seed mean: |diff| median {d['abs_minus_mean_median']:.3f}, "
          f"max {d['abs_minus_mean_max']:.3f} pts; |z| > 2 in {d['share_abs_z_over_2']:.1%} of cells; "
          f"signed mean {d['mean_signed_minus_mean']:+.3f}; mean rank {d['rank_mean']:.1f} of {len(seeds)}")
    print("Comparisons (points):")
    for name, values in summary["comparisons"].items():
        print(f"  {name}: n={values['n']}, |gain| median {values['abs_gain_mean_median']:.2f}, "
              f"gain std median {values['gain_std_median']:.3f}, default seed flips sign in "
              f"{values['default_flips']} ({values['default_flips_if_abs_gain_ge_1pt']} with |gain| >= 1), "
              f"unresolved by the mean {values['unresolved']}")
    print("Ranking, default seed vs seed mean:")
    for record in summary["ranking"]:
        print(f"  {record['backbone']} {record['scope']}: Spearman {record['spearman']:.4f}, "
              f"max shift {record['max_shift']}, top-10 overlap {record['top_overlap']:.1f}")
    print(f"Best DR method per (backbone, dim, quantizer) changed in {summary['winners_changed']} of "
          f"{summary['winners_total']}; frontier cells not in both: {summary['frontier_cells_not_in_both']}")
    print(f"Tables in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
