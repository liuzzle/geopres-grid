#!/usr/bin/env python
"""Thesis figures and tables from the frozen result tables (thesis plan §3.3, §5).

Reads only `storage/frozen/` (data freeze of 3 Nov 2026, logging §100), checks the
files against `SHA256SUMS`, applies the row filter that defines each evaluation set
(the README there), and writes into `--out` (default `storage/thesis/`):

- `tables/T<n>.tex`: booktabs tabulars to `\\input` inside a `table` environment, and
  the unformatted numbers behind each as `T<n>.csv`;
- `figures/F<n>.pdf`: vector, 15 cm wide, sans-serif 9 pt, no titles (the caption is
  the title), plus a PNG preview of each;
- `numbers.tex`: one `\\newcommand` per headline number, and `numbers.csv`, the ledger
  behind it (macro, value, file, filter, aggregation) for `numbers.md`.

Thesis names and orders (plan §5.3) are fixed here once; a code name never reaches a
label. Formats (§5.4): nDCG@10 x 100 with one decimal, retention in percent with one
decimal, differences in pp with a sign, Spearman with two decimals, budgets in bytes.
Random DR methods at fp32 are the seed mean ± std; quantized cells are the default
seed (§3.2). BEIR supplies every effect size; NanoBEIR the 10-seed checks and the
transfer; Impresso the domain check.

    uv run --extra analysis python scripts/thesis_figures.py
    uv run python scripts/thesis_figures.py --no-figures      # tables and numbers only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from geopres_grid import analysis
from geopres_grid.backbones import all_backbones
from geopres_grid.evaluation import TIER_TASKS
from geopres_grid.identity import DEFAULT_DR_SEED

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "storage" / "frozen"
IMPRESSO_DATA = ROOT / "Impresso_IR_test_set_eval_eval_format"

FILES = {
    "nanobeir": "nanobeir_results_restore.csv",
    "beir": "beir_results.csv",
    "impresso": "impresso_collect_results.csv",
    "seeds": "seed_check_results.csv",
}
FILTERS = {
    "nanobeir": "task_type Retrieval; pca_mean in {restore, empty}",
    "beir": "task_type Retrieval (all rows; PCA cells are restore)",
    "impresso": "task starts with Impresso (3,780 rows)",
    "seeds": "tasks of the set; dr_seed empty = default seed 42; PCA cells restore",
}

# --- Thesis names and orders (plan §5.3) ---------------------------------------------

BACKBONES = {
    "mgte": "mGTE",
    "mdenseon": "mDenseOn",
    "embeddinggemma-2": "EmbeddingGemma-2",
    "harrier-270m": "Harrier-270M",
    "harrier-06b": "Harrier-0.6B",
}
BACKBONE_MACRO = {
    "mgte": "Mgte",
    "mdenseon": "Mdenseon",
    "embeddinggemma-2": "Gemma",
    "harrier-270m": "HarrierSmall",
    "harrier-06b": "HarrierLarge",
}
DR = {
    "none": "Full",
    "truncate": "Truncation",
    "pca": "PCA",
    "pca_ror": "PCA+ROR",
    "random_projection": "RP",
    "random_selection": "RS",
}
DR_LONG = {**DR, "random_projection": "Random projection (RP)", "random_selection": "Random selection (RS)"}
QUANT = {
    "none": "fp32",
    "fp16": "fp16",
    "int8": "int8",
    "equal_count_8": "EC-8",
    "int4": "int4",
    "equal_count_4": "EC-4",
    "int2": "int2",
    "equal_count_2": "EC-2",
    "binary": "binary",
}
QUANT_BY_BITS = list(QUANT)  # fp32, fp16, int8, EC-8, int4, EC-4, int2, EC-2, binary
SETS = {"nanobeir": "NanoBEIR", "beir": "BEIR", "impresso": "Impresso"}
IMPRESSO_TASKS = {
    "ImpressoCrosslingualDE": "DE (EN queries)",
    "ImpressoMultilingualDE": "DE (DE queries)",
    "ImpressoCrosslingualFR": "FR (EN queries)",
    "ImpressoMultilingualFR": "FR (FR queries)",
}
IMPRESSO_FOLDERS = {
    "ImpressoCrosslingualDE": ("crosslingual", "DE"),
    "ImpressoMultilingualDE": ("multilingual", "DE"),
    "ImpressoCrosslingualFR": ("crosslingual", "FR"),
    "ImpressoMultilingualFR": ("multilingual", "FR"),
}
DIMS = (64, 128, 256, 512)
BUDGETS = (8, 16, 32, 64, 128, 256)
RANDOM_METHODS = ("random_projection", "random_selection")

REFERENCE: dict[str, tuple[str, int, str]] = {
    "Full / fp16": ("none", 0, "fp16"),
    "Full / int4": ("none", 0, "int4"),
    "Full / EC-2": ("none", 0, "equal_count_2"),
    "Full / binary": ("none", 0, "binary"),
    "PCA-256 / fp32": ("pca", 256, "none"),
    "PCA-128 / EC-4": ("pca", 128, "equal_count_4"),
    "PCA+ROR-256 / EC-2": ("pca_ror", 256, "equal_count_2"),
    "Truncation-128 / fp32": ("truncate", 128, "none"),
}
"""The eight reference configurations (plan §3.3). Fixed once; every 'selected
configurations' figure or table uses exactly these."""

# Okabe-Ito. A backbone keeps its colour everywhere; nothing else is coloured.
COLOURS = {
    "mgte": "#0072B2",
    "mdenseon": "#D55E00",
    "embeddinggemma-2": "#009E73",
    "harrier-270m": "#CC79A7",
    "harrier-06b": "#E69F00",
}
MARKERS = {"none": "*", "truncate": "^", "pca": "s", "pca_ror": "P", "random_projection": "D", "random_selection": "v"}
INK = "#111111"
GRAY = "#8a8a8a"
LIGHT = "#cfcfcf"
WIDE = 15 / 2.54
NARROW = 7.3 / 2.54


def config_name(dr_method: str, target_dim: int, quant_method: str) -> str:
    """`PCA+ROR-256 / EC-2`; at native dimension `Full / EC-2` (plan §5.3)."""
    reduction = "Full" if dr_method == "none" else f"{DR[dr_method]}-{int(target_dim)}"
    return f"{reduction} / {QUANT[quant_method]}"


def ordered_backbones(frame: pd.DataFrame) -> list[str]:
    present = set(frame["backbone"])
    return [key for key in BACKBONES if key in present]


# --- Frozen data ---------------------------------------------------------------------


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_frozen(frozen: Path) -> dict[str, str]:
    """Every file named in SHA256SUMS still hashes the same; returns name -> hash."""
    recorded = {}
    for line in (frozen / "SHA256SUMS").read_text().splitlines():
        digest, _, name = line.partition("  ")
        recorded[name.strip()] = digest.strip()
    for name in FILES.values():
        if name not in recorded:
            raise SystemExit(f"{name} is not in {frozen / 'SHA256SUMS'}")
        actual = sha256(frozen / name)
        if actual != recorded[name]:
            raise SystemExit(f"{name} changed since the freeze: {actual} != {recorded[name]}")
    return {name: recorded[name] for name in FILES.values()}


class Frozen:
    """The frozen tables, read once, with the row filter that defines each set."""

    def __init__(self, frozen: Path) -> None:
        self.frozen = frozen
        self.hashes = check_frozen(frozen)
        self._frames: dict[str, pd.DataFrame] = {}

    def frame(self, name: str) -> pd.DataFrame:
        if name not in self._frames:
            self._frames[name] = analysis.read_results([self.frozen / FILES[name]])
        return self._frames[name]

    def rows(self, evaluation_set: str, metric: str | None = None) -> pd.DataFrame:
        """`analysis.scored_rows` of one evaluation set (retrieval, PCA restore)."""
        frame = self.frame(evaluation_set)
        if evaluation_set == "impresso":
            frame = frame[frame["task"].str.startswith("Impresso")].copy()
            frame["task_type"] = "Retrieval"
            frame["main_score_name"] = "ndcg_at_10"
        if metric:
            frame = frame.assign(main_score=frame[metric]).dropna(subset=["main_score"])
        rows = analysis.scored_rows(frame, pca_mean="restore")
        rows = rows[rows["dr_seed"].isna() | (rows["dr_seed"] == DEFAULT_DR_SEED)]
        return rows

    def seed_cells(self, evaluation_set: str) -> pd.DataFrame:
        """One row per (cell, seed) of the seed file for one set: means over tasks."""
        tier = {"nanobeir": "tier1", "beir": "tier2"}[evaluation_set]
        frame = self.frame("seeds")
        frame = frame[frame["task"].isin(TIER_TASKS[tier])]
        rows = analysis.scored_rows(frame, pca_mean="restore")
        rows = rows.assign(dr_seed=rows["dr_seed"].fillna(DEFAULT_DR_SEED).astype(int))
        keys = list(analysis.CELL_KEYS) + ["bits_per_dim", "bytes_per_vector", "compression_factor", "dr_seed"]
        cells = (
            rows.groupby(keys, dropna=False)
            .agg(score=("main_score", "mean"), retention=("retention", "mean"), n_tasks=("task", "nunique"))
            .reset_index()
        )
        full = rows["task"].nunique()
        short = cells[cells["n_tasks"] < full]
        if len(short):
            print(f"warning: {len(short)} seeded {evaluation_set} cells cover fewer than {full} tasks", file=sys.stderr)
        return cells


# --- Formats (plan §5.4) -------------------------------------------------------------


def score(value: float) -> str:
    return f"{value * 100:.1f}"


def percent(value: float) -> str:
    return f"{value * 100:.1f}"


def pp(value: float) -> str:
    """Points, with a sign; `value` is already in points."""
    return f"{value:+.1f}"


def rho(value: float) -> str:
    return f"{value:.2f}"


def rho3(value: float) -> str:
    """Three decimals, for correlations so close to 1 that two would print 1.00."""
    return f"{value:.3f}"


def integer(value: float) -> str:
    return f"{int(round(value)):,}"


def pm(mean: float, std: float) -> str:
    return f"{mean:+.1f} $\\pm$ {std:.1f}"


def span(mean: float, low: float, high: float) -> str:
    return f"{mean:+.1f} ({low:+.1f} to {high:+.1f})"


def tex(text: str) -> str:
    for char in ("&", "%", "#", "_"):
        text = text.replace(char, "\\" + char)
    return text


def bold_best(values: Iterable[float], formatted: list[str], *, tie: float = 0.5, higher: bool = True) -> list[str]:
    """Bold the best entry, unless the runner-up is within `tie` (in the formatted unit)."""
    values = np.asarray(list(values), dtype=float)
    if np.isnan(values).all() or len(values) < 2:
        return formatted
    order = np.argsort(-values if higher else values)
    best, second = order[0], order[1]
    if abs(values[best] - values[second]) < tie:
        return formatted
    out = list(formatted)
    out[best] = f"\\textbf{{{out[best]}}}"
    return out


# --- LaTeX ---------------------------------------------------------------------------


def latex_table(
    header: list[str],
    rows: list[list[str]],
    column_format: str,
    *,
    groups: list[tuple[str, int]] | None = None,
    midrule_before: Iterable[int] = (),
    notes: Iterable[str] = (),
) -> str:
    """A booktabs tabular. `groups` is an optional first header line of
    (text, span); `notes` become LaTeX comments at the top (for the caption)."""
    lines = [f"% {note}" for note in notes]
    lines += [f"\\begin{{tabular}}{{{column_format}}}", "\\toprule"]
    if groups:
        cells, rules, column = [], [], 1
        for text, width in groups:
            cells.append(f"\\multicolumn{{{width}}}{{c}}{{{text}}}" if text else "")
            if text:
                rules.append(f"\\cmidrule(lr){{{column}-{column + width - 1}}}")
            column += width
        lines.append(" & ".join(cells) + " \\\\")
        lines.append(" ".join(rules))
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")
    breaks = set(midrule_before)
    for index, row in enumerate(rows):
        if index in breaks and index:
            lines.append("\\midrule")
        lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


class Output:
    def __init__(self, out: Path) -> None:
        self.out = out
        (out / "tables").mkdir(parents=True, exist_ok=True)
        (out / "figures").mkdir(parents=True, exist_ok=True)
        self.written: list[str] = []

    def table(self, name: str, latex: str, data: pd.DataFrame) -> None:
        (self.out / "tables" / f"{name}.tex").write_text(latex)
        data.to_csv(self.out / "tables" / f"{name}.csv", index=not isinstance(data.index, pd.RangeIndex))
        self.written.append(f"tables/{name}.tex")

    def figure(self, name: str, figure) -> None:
        import matplotlib.pyplot as plt

        for suffix in ("pdf", "png"):
            figure.savefig(self.out / "figures" / f"{name}.{suffix}")
        plt.close(figure)
        self.written.append(f"figures/{name}.pdf")


# --- Numbers ledger (plan §5.4.5) ----------------------------------------------------

DIGITS = {str(i): word for i, word in enumerate("Zero One Two Three Four Five Six Seven Eight Nine".split())}


def macro_name(*parts: str) -> str:
    text = "".join(part[:1].upper() + part[1:] for part in parts if part)
    return "num" + "".join(DIGITS[c] if c.isdigit() else c for c in text if c.isalnum())


class Ledger:
    def __init__(self) -> None:
        self.rows: list[dict[str, str]] = []
        self.names: set[str] = set()

    def add(self, parts: tuple[str, ...], value: str, file: str, filter_: str, aggregation: str) -> str:
        name = macro_name(*parts)
        if name in self.names:
            raise ValueError(f"duplicate macro {name}")
        self.names.add(name)
        self.rows.append({"macro": name, "value": value, "file": file, "filter": filter_, "aggregation": aggregation})
        return name

    def write(self, out: Path) -> None:
        lines = ["% Generated by scripts/thesis_figures.py from storage/frozen/. Do not edit; rerun."]
        for row in self.rows:
            lines.append(f"\\newcommand{{\\{row['macro']}}}{{{row['value']}}}  % {row['file']}: {row['aggregation']}")
        (out / "numbers.tex").write_text("\n".join(lines) + "\n")
        pd.DataFrame(self.rows).to_csv(out / "numbers.csv", index=False)


# --- Tables --------------------------------------------------------------------------


def table_backbones(output: Output) -> None:
    """T2: the five backbones, from the registry."""
    rows, data = [], []
    registry = {backbone.key: backbone for backbone in all_backbones()}
    for key in BACKBONES:
        backbone = registry[key]
        if backbone.mrl_dims:
            dims = sorted(backbone.mrl_dims)
            step = dims[1] - dims[0] if len(dims) > 2 and len(set(np.diff(dims))) == 1 else None
            mrl = f"every {step} up to {dims[-1]}" if step else ", ".join(str(d) for d in dims)
        elif backbone.mrl_checked:
            mrl = "not documented"
        else:
            mrl = "unknown"
        data.append(
            {
                "backbone": BACKBONES[backbone.key],
                "model_id": backbone.model_id,
                "revision": backbone.revision[:7],
                "native_dim": backbone.native_dim,
                "context_window": backbone.context_window,
                "mrl_dimensions": mrl,
            }
        )
        rows.append(
            [
                BACKBONES[backbone.key],
                f"\\texttt{{{tex(backbone.model_id)}}}",
                str(backbone.native_dim),
                integer(backbone.context_window),
                mrl,
            ]
        )
    latex = latex_table(
        ["Backbone", "Hugging Face model", "$d$", "Context (tokens)", "MRL dimensions (documented)"],
        rows,
        "llrrl",
        notes=["T2: backbones. Revisions pinned in geopres_grid/backbones.py; 'not documented' = checked, no MRL training claim."],
    )
    output.table("T2", latex, pd.DataFrame(data))


def table_storage(output: Output, cells: pd.DataFrame) -> None:
    """T3: bytes per vector for (k, b), and T3b: compression factor of the reference
    configurations per backbone (from the grid's own bookkeeping)."""
    bits = (32, 16, 8, 4, 2, 1)
    natives = analysis.native_dims(cells)
    dims = sorted(set(DIMS) | set(int(d) for d in natives))
    rows, data = [], []
    for k in dims:
        values = [k * b / 8 for b in bits]
        label = f"{k}" + ("" if k in DIMS else " (native)")
        rows.append([label] + [f"{v:g}" for v in values])
        data.append({"k": k, **{f"b={b}": v for b, v in zip(bits, values)}})
    latex = latex_table(
        ["$k$"] + [f"$b={b}$" for b in bits],
        rows,
        "l" + "r" * len(bits),
        midrule_before=[len(DIMS)],
        notes=["T3: bytes per stored vector B = k*b/8. Compression factor r = 4d / B."],
    )
    output.table("T3", latex, pd.DataFrame(data))

    backbones = ordered_backbones(cells)
    rows, data = [], []
    for name, (dr_method, dim, quant) in REFERENCE.items():
        line = [name]
        record = {"configuration": name}
        for backbone in backbones:
            cell = cells[
                (cells["backbone"] == backbone)
                & (cells["dr_method"] == dr_method)
                & (cells["target_dim"] == dim)
                & (cells["quant_method"] == quant)
            ]
            factor = float(cell["compression_factor"].iloc[0]) if len(cell) else np.nan
            line.append(f"{factor:g}$\\times$" if not np.isnan(factor) else "--")
            record[backbone] = factor
        rows.append(line)
        data.append(record)
    latex = latex_table(
        ["Configuration"] + [BACKBONES[b] for b in backbones],
        rows,
        "l" + "r" * len(backbones),
        notes=["T3b: compression factor of the reference configurations relative to the fp32 baseline, per backbone (d differs)."],
    )
    output.table("T3b", latex, pd.DataFrame(data))


def impresso_counts() -> dict[str, tuple[int, int]]:
    counts = {}
    for task, (form, language) in IMPRESSO_FOLDERS.items():
        folder = IMPRESSO_DATA / form / language
        if not folder.exists():
            counts[task] = (np.nan, np.nan)
            continue
        docs = pd.read_csv(folder / "docs.csv")
        queries = pd.read_csv(folder / "queries.csv")
        counts[task] = (len(docs), len(queries))
    return counts


def table_evaluation_sets(output: Output, frozen: Frozen) -> pd.DataFrame:
    """T4: every task of the three sets: documents, queries, split, metric."""
    records = []
    beir_names = {analysis.dataset_name(task): task for task in TIER_TASKS["tier2"]}
    for evaluation_set in ("nanobeir", "beir"):
        rows = frozen.rows(evaluation_set)
        sizes = analysis.corpus_sizes(analysis.task_splits(rows))
        for task, size in sizes.iterrows():
            records.append(
                {
                    "set": SETS[evaluation_set],
                    "task": beir_names.get(analysis.dataset_name(task), task) if evaluation_set == "nanobeir" else task,
                    "task_code": task,
                    "n_documents": size["n_documents"],
                    "n_queries": size["n_queries"],
                    "n_subtasks": int(size["n_subtasks"]),
                    "split": size["split"],
                    "metric": "nDCG@10",
                }
            )
    impresso = frozen.rows("impresso")
    splits = analysis.task_splits(impresso)
    for task, (documents, queries) in impresso_counts().items():
        records.append(
            {
                "set": "Impresso",
                "task": IMPRESSO_TASKS[task],
                "task_code": task,
                "n_documents": documents,
                "n_queries": queries,
                "n_subtasks": 1,
                "split": splits.get(task, "test"),
                "metric": "nDCG@10, R@5",
            }
        )
    data = pd.DataFrame(records)
    rows, breaks, previous = [], [], None
    for index, record in data.iterrows():
        if record["set"] != previous:
            breaks.append(index)
            previous = record["set"]
        task = tex(record["task"]) + (f" ({record['n_subtasks']} sub-tasks, mean)" if record["n_subtasks"] > 1 else "")
        rows.append(
            [
                record["set"] if index in breaks else "",
                task,
                integer(record["n_documents"]) if pd.notna(record["n_documents"]) else "--",
                integer(record["n_queries"]) if pd.notna(record["n_queries"]) else "--",
                record["split"],
                record["metric"],
            ]
        )
    latex = latex_table(
        ["Set", "Task", "Documents", "Queries", "Split", "Metric"],
        rows,
        "llrrll",
        midrule_before=breaks,
        notes=["T4: evaluation sets. NanoBEIR/BEIR sizes from MTEB's descriptive statistics; Impresso counted from the test-set files."],
    )
    output.table("T4", latex, data)
    return data


def table_baselines(output: Output, ledger: Ledger, rows: dict[str, pd.DataFrame], impresso_r5: pd.DataFrame) -> pd.DataFrame:
    """T5: fp32 baselines, backbones x sets (nDCG@10 x 100; Impresso also R@5)."""
    columns = {}
    for evaluation_set, frame in rows.items():
        columns[(SETS[evaluation_set], "nDCG@10")] = analysis.baselines(frame).loc["mean"]
    columns[("Impresso", "R@5")] = analysis.baselines(impresso_r5).loc["mean"]
    table = pd.DataFrame(columns)
    table = table.loc[[b for b in BACKBONES if b in table.index]]
    formatted = pd.DataFrame({column: bold_best(table[column] * 100, [score(v) for v in table[column]]) for column in table.columns}, index=table.index)
    body = [[BACKBONES[backbone]] + list(formatted.loc[backbone]) for backbone in table.index]
    latex = latex_table(
        ["Backbone"] + [f"{metric}" for _, metric in table.columns],
        body,
        "l" + "r" * len(table.columns),
        groups=[("", 1)] + [(name, 1) for name, _ in table.columns],
        notes=[
            "T5: fp32 baselines, mean over the tasks of each set (13 NanoBEIR, 15 BEIR with CQADupstack once, 4 Impresso). Bold: best per column unless within 0.5.",
        ],
    )
    output.table("T5", latex, table.rename(columns=lambda c: f"{c[0]} {c[1]}"))
    for (set_name, metric), column in table.items():
        for backbone, value in column.items():
            ledger.add(
                ("baseline", set_name, {"nDCG@10": "Ndcg", "R@5": "RecallFive"}[metric], BACKBONE_MACRO[backbone]),
                score(value),
                FILES["impresso" if set_name == "Impresso" else set_name.lower()],
                FILTERS["impresso" if set_name == "Impresso" else set_name.lower()] + ("; main_score = recall_at_5" if metric == "R@5" else ""),
                f"kind == baseline; mean of main_score over tasks; x100",
            )
    return table


def prefix_gap_seeds(cells: pd.DataFrame, seeds: pd.DataFrame) -> pd.DataFrame:
    """Truncation minus random selection at fp32, per seed of the random side,
    summarised over seeds (points of retention). `resolved`: |mean| > 2 SEM."""
    truncation = cells[(cells["dr_method"] == "truncate") & (cells["quant_method"] == "none")].set_index(["backbone", "target_dim"])["retention"]
    selection = seeds[(seeds["dr_method"] == "random_selection") & (seeds["quant_method"] == "none")]
    records = []
    for (backbone, dim), group in selection.groupby(["backbone", "target_dim"]):
        gaps = (truncation.loc[(backbone, dim)] - group["retention"].to_numpy()) * 100
        mean, std = float(np.mean(gaps)), float(np.std(gaps, ddof=1))
        sem = std / np.sqrt(len(gaps))
        records.append(
            {
                "backbone": backbone,
                "target_dim": int(dim),
                "n_seeds": len(gaps),
                "gap_mean": mean,
                "gap_std": std,
                "gap_sem": sem,
                "resolved": abs(mean) > 2 * sem,
            }
        )
    return pd.DataFrame(records)


def table_prefix_gap(output: Output, ledger: Ledger, gap: pd.DataFrame) -> None:
    """T6: prefix gap (truncation - random selection) as the seed mean ± std, BEIR fp32."""
    backbones = [b for b in BACKBONES if b in set(gap["backbone"])]
    rows = []
    for backbone in backbones:
        line = [BACKBONES[backbone]]
        for dim in DIMS:
            cell = gap[(gap["backbone"] == backbone) & (gap["target_dim"] == dim)].iloc[0]
            text = pm(cell["gap_mean"], cell["gap_std"])
            line.append(text if cell["resolved"] else text + "$^{\\dagger}$")
        rows.append(line)
    n_seeds = int(gap["n_seeds"].min())
    latex = latex_table(
        ["Backbone"] + [f"$k={d}$" for d in DIMS],
        rows,
        "l" + "r" * len(DIMS),
        notes=[
            f"T6: truncation minus random selection at fp32, BEIR, mean +- std over {n_seeds} random-selection seeds (pp of retention).",
            "dagger: not resolved (|mean| <= 2 SEM).",
        ],
    )
    output.table("T6", latex, gap)
    for _, cell in gap[gap["target_dim"] == 64].iterrows():
        ledger.add(
            ("prefixGap", "BEIR", "k64", BACKBONE_MACRO[cell["backbone"]]),
            pm(cell["gap_mean"], cell["gap_std"]),
            FILES["beir"] + " + " + FILES["seeds"],
            FILTERS["seeds"],
            f"truncate-64/fp32 retention minus random_selection-64/fp32 per seed; mean +- std over {int(cell['n_seeds'])} seeds; x100",
        )


def table_equal_count(output: Output, ledger: Ledger, cells: pd.DataFrame) -> pd.DataFrame:
    """T7: equal-count minus uniform at 8, 4, 2 bits, per DR method (BEIR, pp)."""
    gain = analysis.equal_count_gain(cells)
    methods = [m for m in DR if m in set(gain["dr_method"])]
    rows, data = [], []
    for method in methods:
        line = [DR_LONG[method] if method != "none" else "Full (native $d$)"]
        record = {"dr_method": method}
        for bits in (8, 4, 2):
            part = gain[(gain["dr_method"] == method) & (gain["bits"] == bits)]["gain"]
            line.append(span(part.mean(), part.min(), part.max()))
            record.update({f"mean_{bits}": part.mean(), f"min_{bits}": part.min(), f"max_{bits}": part.max(), f"n_{bits}": len(part)})
        rows.append(line)
        data.append(record)
    latex = latex_table(
        ["DR method", "8 bits", "4 bits", "2 bits"],
        rows,
        "lrrr",
        midrule_before=[1],
        notes=[
            "T7: EC-b minus int-b in pp of retention, BEIR. Mean over the five backbones (and over k = 64..512 for the DR rows), with the range in parentheses.",
        ],
    )
    output.table("T7", latex, pd.DataFrame(data))
    full = gain[(gain["dr_method"] == "none") & (gain["bits"] == 2)]["gain"]
    ledger.add(("ecGain", "BEIR", "full", "twoBits", "mean"), pp(full.mean()), FILES["beir"], FILTERS["beir"], "equal_count_2 minus int2 retention at native d, mean over backbones; x100")
    reduced = gain[(gain["dr_method"] != "none") & (gain["bits"] == 2)]
    by_method = reduced.groupby("dr_method")["gain"].mean()
    ledger.add(("ecGain", "BEIR", "afterDR", "twoBits", "min"), pp(by_method.min()), FILES["beir"], FILTERS["beir"], "EC-2 minus int2, mean over backbones and k per DR method; smallest method mean; x100")
    ledger.add(("ecGain", "BEIR", "afterDR", "twoBits", "max"), pp(by_method.max()), FILES["beir"], FILTERS["beir"], "EC-2 minus int2, mean over backbones and k per DR method; largest method mean; x100")
    return gain


def ror_gain_seeds(seeds: pd.DataFrame, quantizers: Iterable[str]) -> pd.DataFrame:
    """PCA+ROR minus PCA per (backbone, k, quantizer), paired by seed, mean over seeds."""
    keys = ["backbone", "target_dim", "quant_method"]
    part = seeds[seeds["quant_method"].isin(list(quantizers)) & seeds["dr_method"].isin(["pca", "pca_ror"])]
    wide = part.pivot_table(index=keys + ["dr_seed"], columns="dr_method", values="retention")
    wide = wide.dropna()
    wide["gain"] = (wide["pca_ror"] - wide["pca"]) * 100
    per_cell = wide.groupby(keys)["gain"].agg(["mean", "std", "count"]).reset_index()
    return per_cell.rename(columns={"mean": "gain", "std": "gain_std", "count": "n_seeds"})


def table_ror(output: Output, ledger: Ledger, cells: dict[str, pd.DataFrame], nano_seeds: pd.DataFrame) -> pd.DataFrame:
    """T8: PCA+ROR minus PCA by k x quantizer x set (pp); NanoBEIR as the 10-seed mean."""
    quantizers = ("binary", "equal_count_2", "int2", "equal_count_4")
    gains = {
        "nanobeir": ror_gain_seeds(nano_seeds, quantizers),
        "beir": analysis.ror_gain(cells["beir"]),
        "impresso": analysis.ror_gain(cells["impresso"]),
    }
    rows, data, breaks = [], [], []
    for quant in quantizers:
        breaks.append(len(rows))
        for dim in DIMS:
            line = [QUANT[quant] if dim == DIMS[0] else "", str(dim)]
            record = {"quant_method": quant, "target_dim": dim}
            for evaluation_set, gain in gains.items():
                part = gain[(gain["quant_method"] == quant) & (gain["target_dim"] == dim)]["gain"]
                line.append(span(part.mean(), part.min(), part.max()) if len(part) else "--")
                record.update({f"{evaluation_set}_mean": part.mean(), f"{evaluation_set}_min": part.min(), f"{evaluation_set}_max": part.max()})
            rows.append(line)
            data.append(record)
    n_seeds = int(gains["nanobeir"]["n_seeds"].min())
    latex = latex_table(
        ["Quantizer", "$k$", "NanoBEIR", "BEIR", "Impresso"],
        rows,
        "llrrr",
        midrule_before=breaks,
        notes=[
            f"T8: PCA+ROR minus PCA in pp of retention: mean over the five backbones, range in parentheses. NanoBEIR: each backbone's value is the mean over {n_seeds} seeds (paired by seed); BEIR and Impresso: default seed.",
        ],
    )
    output.table("T8", latex, pd.DataFrame(data))
    beir = gains["beir"]
    for dim in (64, 512):
        part = beir[(beir["quant_method"] == "binary") & (beir["target_dim"] == dim)]["gain"]
        ledger.add(("rorGain", "BEIR", "binary", f"k{dim}", "min"), pp(part.min()), FILES["beir"], FILTERS["beir"], f"pca_ror-{dim}/binary minus pca-{dim}/binary, smallest backbone; x100")
        ledger.add(("rorGain", "BEIR", "binary", f"k{dim}", "max"), pp(part.max()), FILES["beir"], FILTERS["beir"], f"pca_ror-{dim}/binary minus pca-{dim}/binary, largest backbone; x100")
    return pd.DataFrame(data)


def best_per_budget(cells: pd.DataFrame) -> pd.DataFrame:
    records = []
    for backbone in ordered_backbones(cells):
        own = cells[cells["backbone"] == backbone]
        for budget in BUDGETS:
            within = own[own["bytes_per_vector"] <= budget]
            if within.empty:
                continue
            best = within.sort_values(["score", "bytes_per_vector", "label"], ascending=[False, True, True]).iloc[0]
            records.append(
                {
                    "backbone": backbone,
                    "budget": budget,
                    "configuration": config_name(best["dr_method"], best["target_dim"], best["quant_method"]),
                    "bytes_per_vector": int(best["bytes_per_vector"]),
                    "retention": float(best["retention"]),
                    "score": float(best["score"]),
                }
            )
    return pd.DataFrame(records)


def table_budget(output: Output, ledger: Ledger, cells: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """T9: the best configuration at each storage budget per backbone, BEIR and Impresso."""
    beir = best_per_budget(cells["beir"]).set_index(["backbone", "budget"])
    impresso = best_per_budget(cells["impresso"]).set_index(["backbone", "budget"])
    joined = beir.join(impresso, lsuffix="_beir", rsuffix="_impresso", how="left").reset_index()
    rows, breaks, previous = [], [], None
    for index, record in joined.iterrows():
        if record["backbone"] != previous:
            breaks.append(index)
            previous = record["backbone"]
        rows.append(
            [
                BACKBONES[record["backbone"]] if index in breaks else "",
                str(record["budget"]),
                record["configuration_beir"],
                percent(record["retention_beir"]),
                score(record["score_beir"]),
                record["configuration_impresso"] if isinstance(record["configuration_impresso"], str) else "--",
                percent(record["retention_impresso"]) if pd.notna(record["retention_impresso"]) else "--",
            ]
        )
    latex = latex_table(
        ["Backbone", "Budget (B)", "Configuration", "Ret. (\\%)", "nDCG@10", "Configuration", "Ret. (\\%)"],
        rows,
        "lrlrrlr",
        groups=[("", 2), ("BEIR", 3), ("Impresso", 2)],
        midrule_before=breaks,
        notes=[
            "T9: best configuration by mean nDCG@10 among those at or under each budget in bytes per vector, with its retention; mean over the set's tasks.",
        ],
    )
    output.table("T9", latex, joined)
    at64 = joined[joined["budget"] == 64]
    ledger.add(("budget64", "BEIR", "retention", "min"), percent(at64["retention_beir"].min()), FILES["beir"], FILTERS["beir"], "best cell with bytes_per_vector <= 64 per backbone; smallest retention; x100")
    ledger.add(("budget64", "BEIR", "retention", "max"), percent(at64["retention_beir"].max()), FILES["beir"], FILTERS["beir"], "best cell with bytes_per_vector <= 64 per backbone; largest retention; x100")
    return joined


def table_scale_cost(output: Output, ledger: Ledger, scale: pd.DataFrame, transfer: pd.DataFrame, paired: pd.DataFrame) -> None:
    """T10a: retention change BEIR - NanoBEIR at fp32 by DR x k; T10b: at native d by quantizer."""
    fp32 = scale["none"].unstack("target_dim")
    methods = [m for m in DR if m not in ("none", "pca_ror") and m in fp32.index]
    names = {**DR_LONG, "pca": "PCA (and PCA+ROR)"}
    rows = [[names[m]] + [pp(fp32.loc[m, d]) for d in DIMS] for m in methods]
    latex = latex_table(
        ["DR method"] + [f"$k={d}$" for d in DIMS],
        rows,
        "l" + "r" * len(DIMS),
        notes=["T10a: retention on BEIR minus retention on NanoBEIR (pp), fp32, mean over the five backbones, 13 shared datasets. PCA+ROR equals PCA at fp32."],
    )
    output.table("T10a", latex, fp32.loc[methods, list(DIMS)])
    native = scale.loc[("none", 0)]
    quantizers = [q for q in QUANT_BY_BITS if q != "none" and q in native.index]
    latex = latex_table(
        [QUANT[q] for q in quantizers],
        [[pp(native[q]) for q in quantizers]],
        "r" * len(quantizers),
        notes=["T10b: retention on BEIR minus retention on NanoBEIR (pp) at native dimension, by quantizer, mean over the five backbones."],
    )
    output.table("T10b", latex, native[quantizers].to_frame("change_pp").T)
    for dim in DIMS:
        ledger.add(("scaleCost", "truncation", f"k{dim}"), pp(fp32.loc["truncate", dim]), FILES["beir"] + " + " + FILES["nanobeir"], "13 shared datasets; PCA restore", "retention(BEIR) - retention(NanoBEIR) for truncate-k/fp32, mean over backbones; x100")
    pooled = transfer[transfer["backbone"] == "pooled"].iloc[0]
    rho = rho3  # transfer correlations sit above 0.99; two decimals would print 1.00
    ledger.add(("transfer", "spearman", "pooled"), rho(pooled["spearman_retention"]), FILES["beir"] + " + " + FILES["nanobeir"], "13 shared datasets; baseline cell excluded", "Spearman of cell retention NanoBEIR vs BEIR, all backbones pooled")
    per_backbone = transfer[transfer["backbone"] != "pooled"]["spearman_retention"]
    ledger.add(("transfer", "spearman", "min"), rho(per_backbone.min()), FILES["beir"] + " + " + FILES["nanobeir"], "13 shared datasets; baseline cell excluded", "Spearman of cell retention NanoBEIR vs BEIR per backbone; smallest")
    ledger.add(("transfer", "spearman", "max"), rho(per_backbone.max()), FILES["beir"] + " + " + FILES["nanobeir"], "13 shared datasets; baseline cell excluded", "Spearman of cell retention NanoBEIR vs BEIR per backbone; largest")
    for subset, selected in transfer_by_family(paired).items():
        ledger.add(("transfer", "spearman", subset, "min"), rho(selected.min()), FILES["beir"] + " + " + FILES["nanobeir"], f"13 shared datasets; {subset} cells; baseline excluded", "Spearman of cell retention NanoBEIR vs BEIR per backbone; smallest")
        ledger.add(("transfer", "spearman", subset, "max"), rho(selected.max()), FILES["beir"] + " + " + FILES["nanobeir"], f"13 shared datasets; {subset} cells; baseline excluded", "Spearman of cell retention NanoBEIR vs BEIR per backbone; largest")


def transfer_by_family(paired: pd.DataFrame) -> dict[str, pd.Series]:
    """Spearman of retention between the tiers per backbone, for the PCA-based cells
    and the rest separately (the plan quotes them apart)."""
    from scipy import stats

    cells = paired[~((paired["dr_method"] == "none") & (paired["quant_method"] == "none"))]
    is_pca = cells["dr_method"].isin(analysis.PCA_METHODS)
    out = {}
    for name, part in (("nonPCA", cells[~is_pca]), ("PCA", cells[is_pca])):
        out[name] = part.groupby("backbone").apply(
            lambda g: stats.spearmanr(g["retention_small"], g["retention_large"])[0], include_groups=False
        )
    return out


def table_impresso(output: Output, impresso: pd.DataFrame, impresso_r5: pd.DataFrame) -> None:
    """T11: Impresso fp32 baselines per task and backbone, nDCG@10 and R@5."""
    ndcg = analysis.baselines(impresso)
    recall = analysis.baselines(impresso_r5)
    backbones = [b for b in BACKBONES if b in ndcg.columns]
    rows = []
    for task in list(IMPRESSO_TASKS) + ["mean"]:
        line = [IMPRESSO_TASKS.get(task, "Mean")]
        for backbone in backbones:
            line += [score(ndcg.loc[task, backbone]), score(recall.loc[task, backbone])]
        rows.append(line)
    latex = latex_table(
        ["Task"] + ["nDCG@10", "R@5"] * len(backbones),
        rows,
        "l" + "rr" * len(backbones),
        groups=[("", 1)] + [(BACKBONES[b], 2) for b in backbones],
        midrule_before=[len(IMPRESSO_TASKS)],
        notes=["T11: Impresso fp32 baselines per task (x100). Cross-lingual = English queries; same-language = queries in the document language."],
    )
    data = pd.concat({"ndcg_at_10": ndcg, "recall_at_5": recall}, axis=1)
    output.table("T11", latex, data)


# --- Figures -------------------------------------------------------------------------


def pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.edgecolor": "#bbbbbb",
            "axes.labelcolor": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#e4e4e4",
            "grid.linewidth": 0.5,
            "legend.frameon": False,
            "lines.linewidth": 1.4,
            "lines.markersize": 4.5,
            "lines.markeredgewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.dpi": 200,
            "figure.facecolor": "white",
        }
    )
    return plt


def panel_grid(plt, n_panels: int, *, height: float, ncols: int = 3, sharey: bool = True):
    nrows = int(np.ceil((n_panels + 1) / ncols))  # one spare cell for the legend
    figure, axes = plt.subplots(nrows, ncols, figsize=(WIDE, height), sharey=sharey, squeeze=False, layout="constrained")
    flat = list(axes.flat)
    for axis in flat[n_panels:]:
        axis.axis("off")
    return figure, flat[:n_panels], flat[n_panels]


def corner_label(axis, text: str, corner: str = "upper left") -> None:
    x, y, ha, va = {"upper left": (0.02, 0.97, "left", "top"), "lower right": (0.98, 0.03, "right", "bottom")}[corner]
    axis.text(x, y, text, transform=axis.transAxes, ha=ha, va=va, fontsize=8.5, color=INK)


def figure_quantization(cells: pd.DataFrame, ledger: Ledger):
    """F2: quantization alone at native d, BEIR: retention and absolute nDCG@10."""
    plt = pyplot()
    retention = analysis.quantization_only(cells)
    scores = analysis.quantization_only(cells, "score")
    quantizers = [q for q in QUANT_BY_BITS if q != "none" and q in retention.columns]
    position = {"fp16": 0.0, "int8": 0.8, "equal_count_8": 1.2, "int4": 1.8, "equal_count_4": 2.2, "int2": 2.8, "equal_count_2": 3.2, "binary": 4.0}
    backbones = [b for b in BACKBONES if b in retention.index]
    jitter = np.linspace(-0.09, 0.09, len(backbones))
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(WIDE, 3.6), sharex=True, layout="constrained", height_ratios=[1.2, 1])
    for offset, backbone in zip(jitter, backbones):
        x = [position[q] + offset for q in quantizers]
        top.plot(x, retention.loc[backbone, quantizers] * 100, linestyle="none", marker="o", color=COLOURS[backbone], label=BACKBONES[backbone])
        bottom.plot(x, scores.loc[backbone, quantizers] * 100, linestyle="none", marker="o", color=COLOURS[backbone])
        bottom.axhline(scores.loc[backbone, "none"] * 100, color=COLOURS[backbone], linewidth=0.7, linestyle=(0, (3, 2)))
    top.axhline(100, color=GRAY, linewidth=0.8)
    for boundary in (0.4, 1.5, 2.5, 3.6):
        for axis in (top, bottom):
            axis.axvline(boundary, color="#e4e4e4", linewidth=0.6)
    top.set_ylabel("Retention (%)")
    bottom.set_ylabel("nDCG@10")
    bottom.set_xticks([position[q] for q in quantizers])
    bottom.set_xticklabels([QUANT[q] for q in quantizers])
    bottom.set_xlabel("Quantizer at native dimension (bits: 16, 8, 4, 2, 1; dashed: fp32 baseline)")
    top.legend(ncol=len(backbones), loc="lower left", bbox_to_anchor=(0.0, 0.0), handletextpad=0.2, columnspacing=1.0)
    for quant in quantizers:
        ledger.add(("quantRetention", "BEIR", QUANT[quant], "min"), percent(retention[quant].min()), FILES["beir"], FILTERS["beir"], f"full/{quant} retention, smallest backbone; x100")
        ledger.add(("quantRetention", "BEIR", QUANT[quant], "max"), percent(retention[quant].max()), FILES["beir"], FILTERS["beir"], f"full/{quant} retention, largest backbone; x100")
    return figure


def figure_reduction(cells: pd.DataFrame, seeds: pd.DataFrame, gap: pd.DataFrame):
    """F3: DR alone at fp32, BEIR: one panel per backbone, random methods as seed mean ± std."""
    plt = pyplot()
    retention = analysis.reduction_only(cells)
    baseline = cells[(cells["dr_method"] == "none") & (cells["quant_method"] == "none")].set_index("backbone")["score"]
    natives = analysis.native_dims(cells)
    random = seeds[seeds["quant_method"].eq("none") & seeds["dr_method"].isin(RANDOM_METHODS)]
    summary = random.groupby(["backbone", "dr_method", "target_dim"])["retention"].agg(["mean", "std", "count"])
    backbones = [b for b in BACKBONES if b in retention.index.get_level_values(0)]
    figure, panels, legend_axis = panel_grid(plt, len(backbones), height=4.2)
    for axis, backbone in zip(panels, backbones):
        native = int(natives[backbone])
        for method in ("truncate", "pca"):
            line = retention.loc[(backbone, method)]
            axis.plot(list(DIMS), line[list(DIMS)] * 100, marker=MARKERS[method], color=INK, label=DR_LONG[method], linestyle="-" if method == "pca" else "--")
        for method in RANDOM_METHODS:
            part = summary.loc[(backbone, method)]
            axis.errorbar(part.index, part["mean"] * 100, yerr=part["std"] * 100, marker=MARKERS[method], color=GRAY, linestyle=":" if method == "random_selection" else "-.", capsize=2, label=DR_LONG[method])
        axis.plot([native], [100], marker=MARKERS["none"], color=INK, markersize=8, linestyle="none", label="fp32, no reduction")
        axis.axhline(100, color=LIGHT, linewidth=0.8)
        axis.set_xscale("log", base=2)
        axis.set_xticks(list(DIMS) + [native])
        axis.set_xticklabels([str(d) for d in DIMS] + [f"\n{native}"])
        axis.minorticks_off()
        axis.tick_params(axis="x", labelsize=7)
        corner_label(axis, f"{BACKBONES[backbone]}\nfp32 nDCG@10 {score(baseline[backbone])}", "lower right")
        axis.set_xlabel("$k$")
    for axis in panels[::3]:
        axis.set_ylabel("Retention (%)")
    handles, labels = panels[0].get_legend_handles_labels()
    n_seeds = int(summary["count"].min())
    legend_axis.legend(handles, labels, loc="center left", title=f"RP and RS: mean $\\pm$ std over {n_seeds} seeds", title_fontsize=8, alignment="left")
    return figure


def figure_budget(cells: pd.DataFrame):
    """F4: budget frontier per backbone, BEIR: retention (left) and nDCG@10 (right) vs bytes."""
    plt = pyplot()
    frontier = analysis.budget_frontier(cells)
    baselines = cells[(cells["dr_method"] == "none") & (cells["quant_method"] == "none")].set_index("backbone")
    backbones = ordered_backbones(cells)
    low = float(cells["retention"].min() * 100) - 2
    figure, panels, legend_axis = panel_grid(plt, len(backbones), height=4.6)
    sizes = sorted(cells["bytes_per_vector"].unique())
    ticks = [s for s in (8, 32, 128, 512, 2048) if min(sizes) <= s <= max(sizes)]
    for axis, backbone in zip(panels, backbones):
        own = cells[cells["backbone"] == backbone]
        front = frontier[frontier["backbone"] == backbone].sort_values("bytes_per_vector")
        base = float(baselines.loc[backbone, "score"])
        axis.scatter(own["bytes_per_vector"], own["retention"] * 100, s=5, color=LIGHT, linewidths=0, zorder=1)
        axis.step(front["bytes_per_vector"], front["retention"] * 100, where="post", color=COLOURS[backbone], linewidth=1.4, zorder=2)
        for method, part in front.groupby("dr_method"):
            axis.plot(part["bytes_per_vector"], part["retention"] * 100, linestyle="none", marker=MARKERS[method], color=COLOURS[backbone], markersize=4 if method != "none" else 7, zorder=3)
        for quant, label in (("fp16", "fp16"), ("int8", "int8")):
            cell = own[(own["dr_method"] == "none") & (own["quant_method"] == quant)].iloc[0]
            axis.plot([cell["bytes_per_vector"]], [cell["retention"] * 100], marker="o", markerfacecolor="white", markeredgecolor=COLOURS[backbone], linestyle="none", zorder=4)
        axis.plot([baselines.loc[backbone, "bytes_per_vector"]], [100], marker="*", markersize=9, markerfacecolor="white", markeredgecolor=COLOURS[backbone], linestyle="none", zorder=5)
        axis.set_xscale("log", base=2)
        axis.set_xticks(ticks)
        axis.set_xticklabels([str(t) for t in ticks], fontsize=7)
        axis.minorticks_off()
        axis.set_ylim(low, 104)
        twin = axis.twinx()
        twin.set_ylim(low * base, 104 * base)
        twin.grid(False)
        twin.spines["right"].set_visible(True)
        twin.tick_params(axis="y", labelsize=7, colors=GRAY)
        corner_label(axis, BACKBONES[backbone], "lower right")
        axis.set_xlabel("Bytes per vector")
    for axis in panels[::3]:
        axis.set_ylabel("Retention (%)")
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], color=INK, marker=MARKERS[m], linestyle="none", label=DR_LONG[m]) for m in DR if m != "none"]
    handles.append(Line2D([], [], color=INK, marker="*", markersize=9, markerfacecolor="white", linestyle="none", label="fp32 baseline"))
    handles.append(Line2D([], [], color=INK, marker="o", markerfacecolor="white", linestyle="none", label="fp16 and int8 at native $d$"))
    handles.append(Line2D([], [], color=LIGHT, marker="o", markersize=3, linestyle="none", label="every configuration"))
    legend_axis.legend(handles=handles, loc="center left", title="Line: no cheaper configuration scores higher.\nRight axis: nDCG@10 of that backbone.", title_fontsize=7.5, alignment="left")
    return figure


def figure_transfer(paired: pd.DataFrame, transfer: pd.DataFrame, paired_tasks: pd.DataFrame, sizes: pd.DataFrame, correlation: pd.DataFrame):
    """F5: (a) NanoBEIR vs BEIR retention per configuration; (b) retention change vs corpus size."""
    plt = pyplot()
    figure, (left, right) = plt.subplots(1, 2, figsize=(WIDE, 3.0), layout="constrained")
    cells = paired[~((paired["dr_method"] == "none") & (paired["quant_method"] == "none"))]
    for backbone in ordered_backbones(cells):
        own = cells[cells["backbone"] == backbone]
        for method, part in own.groupby("dr_method"):
            left.scatter(part["retention_small"] * 100, part["retention_large"] * 100, s=9, marker=MARKERS[method], color=COLOURS[backbone], alpha=0.8, linewidths=0)
    low = float(min(cells["retention_small"].min(), cells["retention_large"].min()) * 100) - 2
    left.plot([low, 101], [low, 101], color=GRAY, linewidth=0.8, zorder=0)
    left.set_xlabel("Retention on NanoBEIR (%)")
    left.set_ylabel("Retention on BEIR (%)")
    pooled = transfer.set_index("backbone").loc["pooled", "spearman_retention"]
    corner_label(left, f"(a)  $\\rho$ = {pooled:.3f}, {len(cells)} configurations")
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], color=COLOURS[b], marker="o", linestyle="none", label=BACKBONES[b]) for b in ordered_backbones(cells)]
    handles += [Line2D([], [], color=INK, marker=MARKERS[m], linestyle="none", label=DR_LONG[m]) for m in DR if m != "none"]
    left.legend(handles=handles, loc="lower right", fontsize=6.5, ncol=2, handletextpad=0.1, columnspacing=0.6)

    configs = [name for name, (dr, _, _) in REFERENCE.items() if dr != "none"]
    styles = {
        "PCA-256 / fp32": {"marker": "s", "markerfacecolor": INK},
        "PCA-128 / EC-4": {"marker": "s", "markerfacecolor": "white"},
        "PCA+ROR-256 / EC-2": {"marker": "P", "markerfacecolor": INK},
        "Truncation-128 / fp32": {"marker": "^", "markerfacecolor": INK},
    }
    per_task = paired_tasks.groupby(["config", "task"])["retention_change"].mean().reset_index()
    per_task = per_task.join(sizes["n_documents"], on="task")
    spearman = correlation[correlation["backbone"] == "mean"].set_index("config")["spearman"]
    for config in configs:
        part = per_task[per_task["config"] == config]
        right.plot(part["n_documents"], part["retention_change"] * 100, linestyle="none", color=INK, markersize=5, label=f"{config} ($\\rho$ = {spearman[config]:.2f})", **styles[config])
    right.axhline(0, color=GRAY, linewidth=0.8)
    right.set_xscale("log")
    right.set_xlabel("Documents in the BEIR corpus")
    right.set_ylabel("Retention, BEIR $-$ NanoBEIR (pp)")
    corner_label(right, "(b)")
    right.legend(loc="lower left", fontsize=6.5)
    return figure


def figure_impresso(impresso_losses: pd.DataFrame, beir_losses: pd.DataFrame):
    """F6: retention of the reference configurations per Impresso task, next to BEIR."""
    plt = pyplot()
    configs = list(REFERENCE)
    x = np.arange(len(configs))
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(WIDE, 3.8), sharex=True, layout="constrained", height_ratios=[1.2, 1])
    styles = {
        "ImpressoCrosslingualDE": {"color": INK, "linestyle": "-", "marker": "o", "markerfacecolor": INK},
        "ImpressoMultilingualDE": {"color": INK, "linestyle": "-", "marker": "o", "markerfacecolor": "white"},
        "ImpressoCrosslingualFR": {"color": INK, "linestyle": "--", "marker": "s", "markerfacecolor": INK},
        "ImpressoMultilingualFR": {"color": INK, "linestyle": "--", "marker": "s", "markerfacecolor": "white"},
    }
    per_task = impresso_losses.groupby(["config", "task"]).agg(retention=("retention", "mean"), score=("score", "mean"))
    for task, style in styles.items():
        part = per_task.xs(task, level="task").loc[configs]
        top.plot(x, part["retention"] * 100, label=IMPRESSO_TASKS[task], **style)
        bottom.plot(x, part["score"] * 100, **style)
    beir = beir_losses.groupby("config").agg(retention=("retention", "mean"), score=("score", "mean")).loc[configs]
    top.plot(x, beir["retention"] * 100, color=GRAY, linestyle=":", marker="D", label="BEIR (15 tasks)")
    bottom.plot(x, beir["score"] * 100, color=GRAY, linestyle=":", marker="D")
    top.axhline(100, color=LIGHT, linewidth=0.8)
    top.set_ylabel("Retention (%)")
    bottom.set_ylabel("nDCG@10")
    bottom.set_xticks(x)
    bottom.set_xticklabels(configs, rotation=20, ha="right", fontsize=7.5)
    top.legend(ncol=3, loc="lower left", fontsize=7)
    return figure


# --- Main ----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frozen", type=Path, default=FROZEN)
    parser.add_argument("--out", type=Path, default=ROOT / "storage" / "thesis")
    parser.add_argument("--no-figures", action="store_true", help="Tables and numbers only (no matplotlib)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    frozen = Frozen(args.frozen)
    output = Output(args.out)
    ledger = Ledger()

    rows = {name: frozen.rows(name) for name in ("nanobeir", "beir", "impresso")}
    impresso_r5 = frozen.rows("impresso", metric="recall_at_5")
    cells = {name: analysis.cell_summary(frame) for name, frame in rows.items()}
    beir_seeds = frozen.seed_cells("beir")
    nano_seeds = frozen.seed_cells("nanobeir")
    for name, frame in rows.items():
        print(f"{name}: {len(frame)} rows, {frame['task'].nunique()} tasks, {len(cells[name])} cells")

    table_backbones(output)
    table_storage(output, cells["beir"])
    table_evaluation_sets(output, frozen)
    table_baselines(output, ledger, rows, impresso_r5)
    gap = prefix_gap_seeds(cells["beir"], beir_seeds)
    table_prefix_gap(output, ledger, gap)
    table_equal_count(output, ledger, cells["beir"])
    table_ror(output, ledger, cells, nano_seeds)
    table_budget(output, ledger, cells)

    small, large = analysis.shared_datasets(rows["nanobeir"], rows["beir"])
    paired = analysis.tier_cells(analysis.cell_summary(small), analysis.cell_summary(large))
    transfer = analysis.rank_transfer(paired)
    scale = analysis.scale_cost(paired)
    table_scale_cost(output, ledger, scale, transfer, paired)
    table_impresso(output, rows["impresso"], impresso_r5)

    losses = pd.concat(
        [analysis.task_losses(small, REFERENCE).assign(tier="nanobeir"), analysis.task_losses(large, REFERENCE).assign(tier="beir")],
        ignore_index=True,
    )
    paired_tasks = analysis.paired_tiers(losses, "nanobeir", "beir")
    sizes = analysis.corpus_sizes(analysis.task_splits(large))
    correlation = analysis.size_correlation(paired_tasks, sizes, value="retention_change")
    for config, value in correlation[correlation["backbone"] == "mean"].set_index("config")["spearman"].items():
        if REFERENCE[config][0] != "none":
            ledger.add(("sizeCorrelation", config), rho(value), FILES["beir"] + " + " + FILES["nanobeir"], "13 shared datasets; mean over backbones per dataset", "Spearman of retention change (BEIR - NanoBEIR) against log10 BEIR documents")
    paired_tasks.to_csv(args.out / "tables" / "F5b_paired_tasks.csv", index=False)
    correlation.to_csv(args.out / "tables" / "F5b_correlation.csv", index=False)
    paired.to_csv(args.out / "tables" / "F5a_cells.csv", index=False)

    impresso_losses = analysis.task_losses(rows["impresso"], REFERENCE)
    beir_losses = analysis.task_losses(rows["beir"], REFERENCE)
    reference_table = pd.concat(
        [
            impresso_losses.groupby(["config", "task"])[["retention", "score"]].mean().reset_index().assign(set="impresso"),
            beir_losses.groupby("config")[["retention", "score"]].mean().reset_index().assign(task="all", set="beir"),
        ],
        ignore_index=True,
    )
    reference_table.to_csv(args.out / "tables" / "F6_reference.csv", index=False)

    if not args.no_figures:
        output.figure("F2", figure_quantization(cells["beir"], ledger))
        output.figure("F3", figure_reduction(cells["beir"], beir_seeds, gap))
        output.figure("F4", figure_budget(cells["beir"]))
        output.figure("F5", figure_transfer(paired, transfer, paired_tasks, sizes, correlation))
        output.figure("F6", figure_impresso(impresso_losses, beir_losses))

    ledger.write(args.out)
    manifest = {
        "frozen": str(args.frozen),
        "hashes": frozen.hashes,
        "filters": FILTERS,
        "reference_configurations": {name: list(value) for name, value in REFERENCE.items()},
        "written": output.written,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"{len(output.written)} files -> {args.out}; {len(ledger.rows)} numbers in numbers.tex")
    return 0


if __name__ == "__main__":
    sys.exit(main())
