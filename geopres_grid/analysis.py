"""Analysis tables over the tidy results frame (`results.load_results` / `results.csv`).

Every number in the tier-1 analysis (logging §23) comes from a function here, so a
later tier is analysed the same way. All of them take the tidy frame, or the
per-cell summary `cell_summary` makes from it, and return plain DataFrames;
`plots.py` draws them and `scripts/analyse_results.py` writes both.

Retention is a cell's score divided by its backbone's fp32 baseline on the same
(task, split, subset), averaged over tasks -- the mean of ratios, so a task with a
low baseline weighs as much as one with a high one. Differences between cells are
reported in points of retention (x100). Every relative table has an absolute twin
(mean main score, nDCG@10 for retrieval) or carries the absolute scores as columns,
so a reader sees both.

`centring_check` is separate: it reads cached embeddings, not results, and tests
one explanation for PCA losing to truncation at high dimension (§23.2).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from geopres_grid.backbones import all_backbones
from geopres_grid.identity import PCA_METHODS
from geopres_grid.reducers import PCA, RandomSelection, Truncate

QUANT_ORDER: tuple[str, ...] = (
    "none",
    "fp16",
    "int8",
    "int4",
    "int2",
    "binary",
    "equal_count_8",
    "equal_count_4",
    "equal_count_2",
)
"""Grid quantizers by method family, then falling bit width."""

DR_ORDER: tuple[str, ...] = ("none", "truncate", "pca", "pca_ror", "random_projection", "random_selection")

CELL_KEYS: tuple[str, ...] = ("backbone", "dr_method", "target_dim", "quant_method")
SCORE_KEYS: tuple[str, ...] = ("backbone", "task", "split", "subset")

REPRESENTATIVE_CELLS: dict[str, tuple[str, int, str]] = {
    "full/binary": ("none", 0, "binary"),
    "full/int2": ("none", 0, "int2"),
    "full/EC-2": ("none", 0, "equal_count_2"),
    "pca-128": ("pca", 128, "none"),
    "pca-64": ("pca", 64, "none"),
    "truncate-128": ("truncate", 128, "none"),
    "truncate-256/int4": ("truncate", 256, "int4"),
    "pca_ror-128/EC-4": ("pca_ror", 128, "equal_count_4"),
}
"""Cells followed per task (`task_deltas`, `task_losses`): quantization alone, DR
alone and two combinations, as (dr_method, target_dim, quant_method)."""


def backbone_order(frame: pd.DataFrame) -> list[str]:
    """Backbones present in `frame`, in registry order, so colours follow the model."""
    present = set(frame["backbone"])
    ordered = [backbone.key for backbone in all_backbones() if backbone.key in present]
    return ordered + sorted(present - set(ordered))


def read_results(paths: Iterable[str | Path]) -> pd.DataFrame:
    """Tidy results tables read as one. A row in two of them (same run, task, split,
    subset: a baseline next to a rerun that also carries it) is kept once."""
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    return frame.drop_duplicates(["run_id", "task", "split", "subset"], ignore_index=True)


def one_pca_mean(frame: pd.DataFrame, pca_mean: str | None = None) -> pd.DataFrame:
    """Keep the PCA cells of one `pca_mean` mode (§29), and every other row.

    Every table here keys a cell on (backbone, method, dimension, quantizer), so
    restored and subtracted PCA cells side by side would be averaged into one.
    Without `pca_mean`, a frame holding both modes is refused. A table from before
    the column existed, or a PCA row without a mode, is a subtracted PCA cell.
    """
    frame = frame.copy()
    is_pca = frame["dr_method"].isin(PCA_METHODS)
    if "pca_mean" not in frame.columns:
        frame["pca_mean"] = None
    frame["pca_mean"] = frame["pca_mean"].astype(object)
    frame.loc[is_pca & frame["pca_mean"].isna(), "pca_mean"] = "subtract"
    modes = set(frame.loc[is_pca, "pca_mean"].dropna())
    if pca_mean is None:
        if len(modes) > 1:
            raise ValueError(f"PCA cells in both pca_mean modes {sorted(modes)}; choose one")
        return frame
    return frame[~is_pca | (frame["pca_mean"] == pca_mean)]


def scored_rows(frame: pd.DataFrame, task_type: str = "Retrieval", pca_mean: str | None = None) -> pd.DataFrame:
    """Rows of one task type, each with its baseline score, delta and retention.

    `target_dim` becomes 0 for the no-DR rows, so it can be a groupby key. A run
    with no baseline for its (backbone, task, split, subset) raises: retention
    against a missing baseline would silently drop the row. PCA cells of one
    `pca_mean` mode only (`one_pca_mean`).
    """
    frame = one_pca_mean(frame, pca_mean)
    rows = frame[frame["task_type"] == task_type].copy()
    rows["target_dim"] = rows["target_dim"].fillna(0).astype(int)
    keys = list(SCORE_KEYS)
    baselines = rows[rows["kind"] == "baseline"]
    duplicated = baselines.duplicated(keys, keep=False)
    if duplicated.any():
        raise ValueError(f"more than one baseline for {baselines.loc[duplicated, keys].drop_duplicates().to_dict('records')}")
    baselines = baselines[keys + ["main_score"]].rename(columns={"main_score": "baseline"})
    rows = rows.merge(baselines, on=keys, how="left", validate="many_to_one")
    missing = rows["baseline"].isna()
    if missing.any():
        raise ValueError(f"no baseline for {rows.loc[missing, keys].drop_duplicates().to_dict('records')}")
    rows["delta"] = rows["main_score"] - rows["baseline"]
    rows["retention"] = rows["main_score"] / rows["baseline"]
    return rows


def cell_summary(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per (backbone, DR, dimension, quantizer): means over tasks.

    The baseline appears as the `none`/0/`none` cell with retention 1. `n_tasks`
    is kept so a cell with a failed task is visible rather than averaged over fewer,
    and `pca_mean` so a PCA cell says which mode it is.
    """
    keys = list(CELL_KEYS) + ["bits_per_dim", "bytes_per_vector", "compression_factor"]
    if "pca_mean" in rows.columns:  # one mode per table (`one_pca_mean`); kept so it is visible
        keys.append("pca_mean")
    cells = (
        rows.groupby(keys, dropna=False)
        .agg(
            score=("main_score", "mean"),
            retention=("retention", "mean"),
            delta=("delta", "mean"),
            min_retention=("retention", "min"),
            n_tasks=("task", "nunique"),
        )
        .reset_index()
    )
    cells["label"] = [cell_label(*values) for values in cells[["dr_method", "target_dim", "quant_method"]].itertuples(index=False)]
    return cells


def cell_label(dr_method: str, target_dim: int, quant_method: str) -> str:
    """`pca_ror-256/equal_count_2`; the no-DR cells as `full/<quantizer>`."""
    reduction = "full" if dr_method == "none" else f"{dr_method}-{int(target_dim)}"
    return f"{reduction}/{quant_method}"


def baselines(rows: pd.DataFrame) -> pd.DataFrame:
    """fp32 baseline per task (rows) and backbone (columns), plus a `mean` row."""
    table = rows[rows["kind"] == "baseline"].pivot_table(index="task", columns="backbone", values="main_score")
    table = table[backbone_order(rows)]
    table.loc["mean"] = table.mean()
    return table


def noise_floor(rows: pd.DataFrame, quant_methods: Iterable[str] = ("fp16", "int8")) -> dict[str, float]:
    """How far near-lossless cells move a single task: |delta| without DR.

    fp16 and int8 change scores only through rounding, so their spread is the
    arithmetic floor. int4 / equal_count_4 are also near-lossless on the task mean
    but already move single small tasks by several points -- the sampling noise of
    50-query NanoBEIR tasks, not a property of the quantizer.
    """
    selected = rows[(rows["dr_method"] == "none") & rows["quant_method"].isin(list(quant_methods))]
    return {
        "max_abs_delta": float(selected["delta"].abs().max()),
        "q95_abs_delta": float(selected["delta"].abs().quantile(0.95)),
        "n": int(len(selected)),
    }


def native_dims(cells: pd.DataFrame) -> pd.Series:
    """Each backbone's native dimension, from its fp32 baseline cell (4 bytes a coordinate)."""
    baseline = cells[(cells["dr_method"] == "none") & (cells["quant_method"] == "none")]
    return (baseline.set_index("backbone")["bytes_per_vector"] // 4).astype(int).rename("native_dim")


def quantization_only(cells: pd.DataFrame, value: str = "retention") -> pd.DataFrame:
    """Every quantizer at full dimension: backbone x quantizer.

    `value="retention"` (default) or `"score"` for the mean nDCG@10; the `none`
    column is the fp32 baseline either way.
    """
    table = cells[cells["dr_method"] == "none"].pivot(index="backbone", columns="quant_method", values=value)
    return table.loc[backbone_order(cells), [q for q in QUANT_ORDER if q in table.columns]]


def reduction_only(cells: pd.DataFrame, value: str = "retention") -> pd.DataFrame:
    """Every DR method by dimension at fp32: (backbone, method) x dimension.

    Each row ends at its backbone's native dimension with the fp32 baseline (no
    reduction): retention 1, or the baseline score. The native column is empty for
    the backbones whose native dimension is another one.
    """
    selected = cells[(cells["quant_method"] == "none") & (cells["dr_method"] != "none")]
    table = selected.pivot_table(index=["backbone", "dr_method"], columns="target_dim", values=value)
    baseline = cells[(cells["dr_method"] == "none") & (cells["quant_method"] == "none")].set_index("backbone")[value]
    natives = native_dims(cells)
    for backbone, method in table.index:
        table.loc[(backbone, method), natives[backbone]] = baseline[backbone]
    return table[sorted(table.columns)]


def all_cells(cells: pd.DataFrame, value: str = "retention") -> pd.DataFrame:
    """Every cell of the grid: (backbone, DR method, dimension) x quantizer.

    Rows per backbone start with no reduction (`none`, 0), then `DR_ORDER` by
    dimension; columns follow `QUANT_ORDER`. `value="retention"` (default) or
    `"score"` for the mean nDCG@10. A cell the table does not hold stays empty.
    """
    table = cells.pivot_table(index=["backbone", "dr_method", "target_dim"], columns="quant_method", values=value)
    methods = [m for m in DR_ORDER if m in set(cells["dr_method"])] + sorted(set(cells["dr_method"]) - set(DR_ORDER))
    order = [
        (backbone, method, dim)
        for backbone in backbone_order(cells)
        for method in methods
        for dim in sorted(cells.loc[(cells["backbone"] == backbone) & (cells["dr_method"] == method), "target_dim"].unique())
    ]
    return table.reindex(index=pd.MultiIndex.from_tuples(order, names=table.index.names), columns=[q for q in QUANT_ORDER if q in table.columns])


def method_gain(cells: pd.DataFrame, column: str, better: str, worse: str, keys: Iterable[str]) -> pd.DataFrame:
    """`better - worse` in points of retention, matched on every other cell key.

    `column` is what differs between the two (`dr_method` or `quant_method`);
    `keys` are the remaining cell keys to match on. `better_score` and
    `worse_score` are the two cells' mean scores (nDCG@10), so the gain can be
    read against where both stand.
    """
    keys = list(keys)
    left = cells[cells[column] == better].set_index(keys)
    right = cells[cells[column] == worse].set_index(keys)
    gain = pd.DataFrame(
        {
            "gain": (left["retention"] - right["retention"]) * 100,
            "better_score": left["score"],
            "worse_score": right["score"],
        }
    )
    return gain.dropna().reset_index()


def ror_gain(cells: pd.DataFrame) -> pd.DataFrame:
    """PCA+ROR minus PCA per backbone, dimension and quantizer (points).

    Exactly zero at fp32 -- an orthogonal rotation preserves cosine -- which makes
    the `none` column a correctness check on the reducers as much as a result.
    """
    return method_gain(cells, "dr_method", "pca_ror", "pca", ["backbone", "target_dim", "quant_method"])


def equal_count_gain(cells: pd.DataFrame) -> pd.DataFrame:
    """Kisako et al.'s global equal-count table minus the per-dimension uniform
    quantizer at the same bit width, per backbone, DR method and dimension (points)."""
    parts = []
    for bits in (8, 4, 2):
        gain = method_gain(
            cells, "quant_method", f"equal_count_{bits}", f"int{bits}", ["backbone", "dr_method", "target_dim"]
        )
        gain["bits"] = bits
        parts.append(gain)
    return pd.concat(parts, ignore_index=True)


def prefix_gap(cells: pd.DataFrame, quant_method: str = "none") -> pd.DataFrame:
    """Truncation minus random coordinate selection, per backbone and dimension (points).

    The grid's version of the MRL probe (logging §12): both keep k original
    coordinates and differ only in which, so a positive gap means the prefix is
    privileged. One random-selection seed, so a gap of a point or two is noise.
    """
    selected = cells[cells["quant_method"] == quant_method]
    return method_gain(selected, "dr_method", "truncate", "random_selection", ["backbone", "target_dim", "quant_method"])


def budget_frontier(cells: pd.DataFrame) -> pd.DataFrame:
    """Per backbone, the cells no cheaper cell beats: the score/bytes Pareto front.

    Sorted by bytes per vector; a cell is kept when its mean score is higher than
    every cell at or below its size. At equal size only the best one is kept.
    """
    fronts = []
    for _, group in cells.groupby("backbone"):
        group = group.sort_values(["bytes_per_vector", "score"], ascending=[True, False])
        group = group.drop_duplicates("bytes_per_vector")
        best_so_far = group["score"].cummax().shift(fill_value=-np.inf)
        fronts.append(group[group["score"] > best_so_far])
    return pd.concat(fronts, ignore_index=True)


def task_losses(rows: pd.DataFrame, selection: dict[str, tuple[str, int, str]]) -> pd.DataFrame:
    """One row per (backbone, task, named cell): baseline, score, delta, retention.

    `selection` maps a name to (dr_method, target_dim, quant_method); use
    target_dim 0 for no DR. The long form behind `task_deltas` and `task_scores`,
    and the per-backbone view of them. Keeps `pca_mean`, so a combined table of two
    tiers still says which PCA cells it holds.
    """
    parts = []
    for name, (dr_method, target_dim, quant_method) in selection.items():
        selected = rows[
            (rows["dr_method"] == dr_method)
            & (rows["target_dim"] == target_dim)
            & (rows["quant_method"] == quant_method)
        ]
        parts.append(selected.assign(config=name))
    columns = ["config", "backbone", "task", "split", "subset", "baseline", "main_score", "delta", "retention"]
    losses = pd.concat(parts, ignore_index=True)
    if "pca_mean" in losses.columns:
        columns.append("pca_mean")
    return losses[columns].rename(columns={"main_score": "score"})


def task_deltas(rows: pd.DataFrame, selection: dict[str, tuple[str, int, str]]) -> pd.DataFrame:
    """Mean delta over backbones (points) per task, for named cells (`task_losses`)."""
    losses = task_losses(rows, selection)
    table = losses.pivot_table(index="task", columns="config", values="delta", aggfunc="mean") * 100
    return table[[name for name in selection if name in table.columns]]


def task_scores(rows: pd.DataFrame, selection: dict[str, tuple[str, int, str]]) -> pd.DataFrame:
    """The absolute twin of `task_deltas`: mean score over backbones per task, with
    the fp32 baseline as the first column."""
    losses = task_losses(rows, selection)
    table = losses.pivot_table(index="task", columns="config", values="score", aggfunc="mean")
    table = table[[name for name in selection if name in table.columns]]
    baseline = rows[rows["kind"] == "baseline"].groupby("task")["main_score"].mean()
    table.insert(0, "fp32", baseline)
    return table


# --- Corpus size -------------------------------------------------------------------


def corpus_sizes(splits: dict[str, str]) -> pd.DataFrame:
    """Documents and queries per task, from MTEB's packaged descriptive statistics.

    `splits` maps a task name to its evaluated split (`task_splits`). Offline: the statistics
    ship with `mteb`, nothing is downloaded. An aggregate task (CQADupstackRetrieval)
    is scored per sub-task, each against its own corpus, and its score is the mean
    over them, so its size is the mean of its sub-tasks' sizes. A task without
    statistics gets NaN rather than a guess.
    """
    import mteb

    def count(values, side):
        # Older statistics (Touche2020) have no num_documents; the text statistics
        # still give the count as total length / average length.
        if values.get(f"num_{side}") is not None:
            return values[f"num_{side}"]
        text = values.get(f"{side}_text_statistics") or {}
        if text.get("average_text_length"):
            return round(text["total_text_length"] / text["average_text_length"])
        return None

    def stats(task, split):
        values = (task.metadata.descriptive_stats or {}).get(split) or {}
        return count(values, "documents"), count(values, "queries")

    rows = []
    for name, split in splits.items():
        task = mteb.get_task(name)
        parts = list(getattr(task.metadata, "tasks", None) or [task])
        sizes = [stats(part, split) for part in parts]
        documents = [d for d, _ in sizes if d is not None]
        queries = [q for _, q in sizes if q is not None]
        rows.append(
            {
                "task": name,
                "split": split,
                "n_documents": float(np.mean(documents)) if len(documents) == len(parts) else np.nan,
                "n_queries": float(np.mean(queries)) if len(queries) == len(parts) else np.nan,
                "n_subtasks": len(parts),
                "source": "mteb descriptive_stats",
            }
        )
    return pd.DataFrame(rows).set_index("task")


def task_splits(rows: pd.DataFrame) -> dict[str, str]:
    """The evaluated split of every task in `rows` (one per task, or it raises)."""
    splits = rows.groupby("task")["split"].unique()
    several = splits[splits.map(len) > 1]
    if len(several):
        raise ValueError(f"tasks evaluated on more than one split: {several.to_dict()}")
    return {task: values[0] for task, values in splits.items()}


def size_correlation(
    losses: pd.DataFrame,
    sizes: pd.DataFrame,
    value: str = "retention",
    groups: Iterable[str] = ("config",),
) -> pd.DataFrame:
    """Spearman and Pearson (on log10 documents) of a per-task loss against corpus size.

    `losses` from `task_losses` (optionally with a `tier` column), `sizes` from
    `corpus_sizes` indexed by task (with `tier` too when tasks repeat across tiers).
    Per group in `groups`, once on the task mean over backbones (`backbone="mean"`)
    and once per backbone. A group holding PCA cells of both `pca_mean` modes is
    refused: it would correlate two different reductions as one.
    """
    from scipy import stats

    groups = list(groups)
    size_keys = ["task"] + (["tier"] if "tier" in sizes.reset_index().columns and "tier" in losses.columns else [])
    data = losses.merge(sizes.reset_index()[size_keys + ["n_documents"]], on=size_keys, how="left")
    data = data.dropna(subset=["n_documents", value])
    rows = []
    for key, group in data.groupby(groups):
        if "pca_mean" in group.columns and group["pca_mean"].dropna().nunique() > 1:
            raise ValueError(f"{dict(zip(groups, np.atleast_1d(key)))}: PCA cells of both pca_mean modes")
        per_task = group.groupby(size_keys).agg(loss=(value, "mean"), n_documents=("n_documents", "first"))
        parts = [("mean", per_task)] + [
            (backbone, part.groupby(size_keys).agg(loss=(value, "mean"), n_documents=("n_documents", "first")))
            for backbone, part in group.groupby("backbone")
        ]
        for backbone, part in parts:
            size = np.log10(part["n_documents"].to_numpy())
            loss = part["loss"].to_numpy()
            if len(part) < 3 or np.ptp(size) == 0 or np.ptp(loss) == 0:
                rho = p_rho = r = p_r = np.nan
            else:
                rho, p_rho = stats.spearmanr(size, loss)
                r, p_r = stats.pearsonr(size, loss)
            rows.append(
                {
                    **dict(zip(groups, np.atleast_1d(key))),
                    "backbone": backbone,
                    "value": value,
                    "n_tasks": len(part),
                    "spearman": float(rho),
                    "spearman_p": float(p_rho),
                    "pearson_log10": float(r),
                    "pearson_p": float(p_r),
                    "min_documents": float(part["n_documents"].min()),
                    "max_documents": float(part["n_documents"].max()),
                }
            )
    return pd.DataFrame(rows)


def dataset_name(task: str) -> str:
    """`NanoClimateFeverRetrieval` and `ClimateFEVER` both -> `climatefever`: the BEIR
    dataset a task samples, to pair a NanoBEIR task with its full counterpart."""
    return task.removeprefix("Nano").removesuffix("Retrieval").lower()


def paired_tiers(losses: pd.DataFrame, small: str, large: str) -> pd.DataFrame:
    """The same dataset at two corpus sizes: retention in `large` minus in `small`.

    `losses` from `task_losses` with a `tier` column. Rows are matched on backbone,
    named cell and `dataset_name`; `task` is the `large` tier's name, so the result
    joins that tier's `corpus_sizes`. A cell whose PCA mode differs between the two
    is refused (§29).
    """
    keys = ["config", "backbone", "dataset"]
    frame = losses.assign(dataset=losses["task"].map(dataset_name))
    left = frame[frame["tier"] == small].set_index(keys)
    right = frame[frame["tier"] == large].set_index(keys)
    paired = right[["task"]].join(
        pd.DataFrame(
            {
                "retention_small": left["retention"],
                "retention_large": right["retention"],
                "score_small": left["score"],
                "score_large": right["score"],
            }
        ),
        how="inner",
    ).dropna(subset=["retention_small", "retention_large"])
    if "pca_mean" in frame.columns:
        modes = pd.DataFrame({"small": left["pca_mean"], "large": right["pca_mean"]}).loc[paired.index]
        mixed = modes.dropna(how="all")
        mixed = mixed[mixed["small"].ne(mixed["large"])]
        if len(mixed):
            raise ValueError(f"PCA cells in different pca_mean modes: {sorted(set(mixed.index.get_level_values(0)))}")
    paired["retention_change"] = paired["retention_large"] - paired["retention_small"]
    return paired.reset_index()


# --- Tier comparison ---------------------------------------------------------------


def shared_datasets(small: pd.DataFrame, large: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Both tiers' rows restricted to the BEIR datasets they share (`dataset_name`),
    so a difference between them is corpus size, not a different task mix."""
    shared = set(small["task"].map(dataset_name)) & set(large["task"].map(dataset_name))
    return small[small["task"].map(dataset_name).isin(shared)], large[large["task"].map(dataset_name).isin(shared)]


def tier_cells(small: pd.DataFrame, large: pd.DataFrame) -> pd.DataFrame:
    """One row per cell present in both tiers: retention and score in each, and the change.

    `small` and `large` are `cell_summary` tables (of `shared_datasets` rows, for a
    like-for-like comparison). Each must hold one `pca_mean` mode (`one_pca_mean`);
    a PCA cell in different modes in the two tiers is refused (§29).
    """
    keys = list(CELL_KEYS) + ["bytes_per_vector"]
    columns = ["retention", "score", "n_tasks", "label"]
    paired = small.set_index(keys)[columns].join(
        large.set_index(keys)[columns[:-1]], lsuffix="_small", rsuffix="_large", how="inner"
    )
    paired = paired.rename(columns={"label_small": "label"}).reset_index()
    for name, table in (("small", small), ("large", large)):
        if "pca_mean" in table.columns:
            modes = set(table.loc[table["dr_method"].isin(PCA_METHODS), "pca_mean"].dropna())
            paired.attrs[f"pca_mean_{name}"] = sorted(modes)
    if paired.attrs.get("pca_mean_small", []) and paired.attrs.get("pca_mean_large", []):
        if paired.attrs["pca_mean_small"] != paired.attrs["pca_mean_large"]:
            raise ValueError(
                f"PCA cells in pca_mean {paired.attrs['pca_mean_small']} vs {paired.attrs['pca_mean_large']}; "
                "drop them or pick one mode for both"
            )
    paired["retention_change"] = paired["retention_large"] - paired["retention_small"]
    return paired


def rank_transfer(paired: pd.DataFrame) -> pd.DataFrame:
    """Spearman of retention and of score between the tiers, per backbone and pooled.

    The fp32 baseline is left out (retention 1 in both by construction). Pooled
    retention ranks cells of all backbones together; pooled score would mostly rank
    the backbones, so it is reported but says less.
    """
    from scipy import stats

    cells = paired[~((paired["dr_method"] == "none") & (paired["quant_method"] == "none"))]
    rows = []
    for backbone, part in [*cells.groupby("backbone"), ("pooled", cells)]:
        rho, p = stats.spearmanr(part["retention_small"], part["retention_large"])
        rho_score, _ = stats.spearmanr(part["score_small"], part["score_large"])
        rows.append(
            {
                "backbone": backbone,
                "n_cells": len(part),
                "spearman_retention": float(rho),
                "spearman_p": float(p),
                "spearman_score": float(rho_score),
                "mean_change": float(part["retention_change"].mean()),
                "median_change": float(part["retention_change"].median()),
                "q05_change": float(part["retention_change"].quantile(0.05)),
            }
        )
    return pd.DataFrame(rows)


def scale_cost(paired: pd.DataFrame) -> pd.DataFrame:
    """Retention change large - small (points), mean over backbones:
    (DR method, dimension) x quantizer. Negative: the cell costs more at scale."""
    table = paired.pivot_table(
        index=["dr_method", "target_dim"], columns="quant_method", values="retention_change", aggfunc="mean"
    ) * 100
    order = [m for m in DR_ORDER if m in table.index.get_level_values(0)]
    table = table.reindex(sorted(table.index, key=lambda key: (order.index(key[0]), key[1])))
    return table[[q for q in QUANT_ORDER if q in table.columns]]


def budget_choice(paired: pd.DataFrame) -> pd.DataFrame:
    """Would the small tier pick the same cell? Per backbone and storage budget.

    For every budget (each bytes-per-vector size in the grid) the best cell at or
    under it by each tier's mean score; `large_score_of_small_pick` is what the small
    tier's choice scores on the large tier, and `regret` (points of nDCG@10) how much
    below the large tier's own best that is. Ties go to the smaller, then the
    alphabetically first cell, so the choice is deterministic.
    """
    rows = []
    for backbone, group in paired.groupby("backbone"):
        for budget in sorted(group["bytes_per_vector"].unique()):
            within = group[group["bytes_per_vector"] <= budget].sort_values(["bytes_per_vector", "label"])
            small_pick = within.loc[within["score_small"].idxmax()]
            large_pick = within.loc[within["score_large"].idxmax()]
            rows.append(
                {
                    "backbone": backbone,
                    "budget": int(budget),
                    "small_pick": small_pick["label"],
                    "large_pick": large_pick["label"],
                    "same": small_pick["label"] == large_pick["label"],
                    "small_score_of_small_pick": small_pick["score_small"],
                    "large_score_of_small_pick": small_pick["score_large"],
                    "large_best": large_pick["score_large"],
                    "regret": (large_pick["score_large"] - small_pick["score_large"]) * 100,
                }
            )
    return pd.DataFrame(rows)


# --- PCA centring ------------------------------------------------------------------


def _unit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), np.finfo(np.float32).eps)


def top_k(queries: np.ndarray, documents: np.ndarray, k: int) -> np.ndarray:
    """Indices of each query's k nearest documents by cosine."""
    scores = _unit(queries) @ _unit(documents).T
    return np.argsort(-scores, axis=1, kind="stable")[:, :k]


def top_k_overlap(found: np.ndarray, reference: np.ndarray) -> float:
    """Mean share of each query's reference top-k that `found` also returns."""
    k = reference.shape[1]
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(found, reference)]))


def centring_check(
    queries: np.ndarray,
    documents: np.ndarray,
    dims: Iterable[int],
    *,
    k: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Top-k agreement with the fp32 ranking, with and without PCA's mean shift.

    Per dimension: the repo's `PCA` (centred, as the grid runs it), the same
    projection applied to uncentred vectors (components fitted the same way, mean
    not subtracted), truncation, and random selection. Fitting is on queries and
    documents together, as `evaluate_cell` does. At the native dimension the
    centred PCA is a rotation plus a shift, so anything below 1.0 there is the
    shift alone.

    A proxy, not nDCG: it needs no qrels and isolates geometry. It says how much of
    the fp32 ranking a reduction keeps, not whether what changed was relevant.
    """
    queries = _unit(queries)
    documents = _unit(documents)
    both = np.vstack([documents, queries])
    reference = top_k(queries, documents, k)
    rows = []
    for dim in dims:
        pca = PCA(dim, seed=seed)
        pca.fit(both)
        centred = top_k(pca.transform(queries), pca.transform(documents), k)
        uncentred = top_k(queries @ pca.components_.T, documents @ pca.components_.T, k)
        truncate = Truncate(dim)
        truncate.fit(both)
        selection = RandomSelection(dim, seed=seed)
        selection.fit(both)
        rows.append(
            {
                "target_dim": dim,
                "pca_centred": top_k_overlap(centred, reference),
                "pca_uncentred": top_k_overlap(uncentred, reference),
                "truncate": top_k_overlap(top_k(truncate.transform(queries), truncate.transform(documents), k), reference),
                "random_selection": top_k_overlap(
                    top_k(selection.transform(queries), selection.transform(documents), k), reference
                ),
            }
        )
    table = pd.DataFrame(rows).set_index("target_dim")
    table.attrs["mean_norm"] = float(np.linalg.norm(both.mean(axis=0)))
    return table
