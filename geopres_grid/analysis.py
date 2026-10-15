"""Analysis tables over the tidy results frame (`results.load_results` / `results.csv`).

Every number in the tier-1 analysis (logging §23) comes from a function here, so a
later tier is analysed the same way. All of them take the tidy frame, or the
per-cell summary `cell_summary` makes from it, and return plain DataFrames;
`plots.py` draws them and `scripts/analyse_results.py` writes both.

Retention is a cell's score divided by its backbone's fp32 baseline on the same
(task, split, subset), averaged over tasks -- the mean of ratios, so a task with a
low baseline weighs as much as one with a high one. Differences between cells are
reported in points of retention (x100).

`centring_check` is separate: it reads cached embeddings, not results, and tests
one explanation for PCA losing to truncation at high dimension (§23.2).
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from geopres_grid.backbones import all_backbones
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


def backbone_order(frame: pd.DataFrame) -> list[str]:
    """Backbones present in `frame`, in registry order, so colours follow the model."""
    present = set(frame["backbone"])
    ordered = [backbone.key for backbone in all_backbones() if backbone.key in present]
    return ordered + sorted(present - set(ordered))


def scored_rows(frame: pd.DataFrame, task_type: str = "Retrieval") -> pd.DataFrame:
    """Rows of one task type, each with its baseline score, delta and retention.

    `target_dim` becomes 0 for the no-DR rows, so it can be a groupby key. A run
    with no baseline for its (backbone, task, split, subset) raises: retention
    against a missing baseline would silently drop the row.
    """
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
    is kept so a cell with a failed task is visible rather than averaged over fewer.
    """
    keys = list(CELL_KEYS) + ["bits_per_dim", "bytes_per_vector", "compression_factor"]
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


def quantization_only(cells: pd.DataFrame) -> pd.DataFrame:
    """Retention of every quantizer at full dimension: backbone x quantizer."""
    table = cells[cells["dr_method"] == "none"].pivot(index="backbone", columns="quant_method", values="retention")
    return table.loc[backbone_order(cells), [q for q in QUANT_ORDER if q in table.columns]]


def reduction_only(cells: pd.DataFrame) -> pd.DataFrame:
    """fp32 retention of every DR method by dimension: (backbone, method) x dimension."""
    selected = cells[(cells["quant_method"] == "none") & (cells["dr_method"] != "none")]
    return selected.pivot_table(index=["backbone", "dr_method"], columns="target_dim", values="retention")


def method_gain(cells: pd.DataFrame, column: str, better: str, worse: str, keys: Iterable[str]) -> pd.DataFrame:
    """`better - worse` in points of retention, matched on every other cell key.

    `column` is what differs between the two (`dr_method` or `quant_method`);
    `keys` are the remaining cell keys to match on.
    """
    keys = list(keys)
    left = cells[cells[column] == better].set_index(keys)["retention"]
    right = cells[cells[column] == worse].set_index(keys)["retention"]
    gain = ((left - right) * 100).dropna().rename("gain").reset_index()
    return gain


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


def task_deltas(rows: pd.DataFrame, selection: dict[str, tuple[str, int, str]]) -> pd.DataFrame:
    """Mean delta over backbones (points) per task, for named cells.

    `selection` maps a column name to (dr_method, target_dim, quant_method); use
    target_dim 0 for no DR.
    """
    columns = {}
    for name, (dr_method, target_dim, quant_method) in selection.items():
        selected = rows[
            (rows["dr_method"] == dr_method)
            & (rows["target_dim"] == target_dim)
            & (rows["quant_method"] == quant_method)
        ]
        columns[name] = selected.groupby("task")["delta"].mean() * 100
    return pd.DataFrame(columns)


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
