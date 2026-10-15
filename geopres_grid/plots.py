"""Figures for the tables in `analysis.py`. Needs the `analysis` extra (matplotlib).

One function per figure, each taking the analysis table it draws and returning the
matplotlib Figure; `save` writes PNG and PDF side by side. Colours follow the
entity, never its rank: a backbone keeps its colour in every figure, set by
registry order, and a DR method likewise. Palette from the dataviz reference
instance, validated for colour-vision deficiency on adjacent pairs; three of the
hues are below 3:1 on white, so every series also carries its own marker and a
legend, and the numbers are in the CSVs next to the figures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from geopres_grid.analysis import QUANT_ORDER, backbone_order

CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
DIVERGING = ("#b8322f", "#e34948", "#f0efec", "#2a78d6", "#1c5cab")
"""Red (worse) through a neutral gray to blue (better)."""

DR_SERIES = ("truncate", "pca", "random_projection", "random_selection", "pca_ror")
"""DR methods in colour order. `pca_ror` last: it is identical to `pca` at fp32
and only drawn where quantization separates them."""

QUANT_LABELS = {
    "none": "fp32",
    "fp16": "fp16",
    "int8": "int8",
    "int4": "int4",
    "int2": "int2",
    "binary": "binary",
    "equal_count_8": "EC-8",
    "equal_count_4": "EC-4",
    "equal_count_2": "EC-2",
}


def _pyplot():
    try:
        import matplotlib
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError(
            "Plotting needs matplotlib: uv sync --extra analysis "
            "(or uv run --extra analysis ...)"
        ) from error
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.edgecolor": "#c3c2b7",
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlesize": 10,
            "axes.titlecolor": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
        }
    )
    return plt


def series_style(names: list[str]) -> dict[str, dict[str, Any]]:
    """Fixed colour and marker per series name, by position in `names`."""
    if len(names) > len(CATEGORICAL):
        raise ValueError(f"{len(names)} series; the palette has {len(CATEGORICAL)} -- facet instead")
    return {name: {"color": CATEGORICAL[i], "marker": MARKERS[i]} for i, name in enumerate(names)}


def _diverging_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("geopres_diverging", DIVERGING)


def save(figure, directory: str | Path, name: str) -> list[Path]:
    """Write `name.png` and `name.pdf` into `directory`."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = [directory / f"{name}.{suffix}" for suffix in ("png", "pdf")]
    for path in paths:
        figure.savefig(path)
    _pyplot().close(figure)
    return paths


def _dim_axis(axis, dims) -> None:
    axis.set_xscale("log", base=2)
    axis.set_xticks(list(dims))
    axis.set_xticklabels([str(int(d)) for d in dims])
    axis.minorticks_off()


def quantization_only(table: pd.DataFrame):
    """Dot plot: retention per quantizer at full dimension, one marker per backbone."""
    plt = _pyplot()
    quantizers = [q for q in QUANT_ORDER if q in table.columns and q != "none"]
    styles = series_style(list(table.index))
    figure, axis = plt.subplots(figsize=(7.0, 3.4))
    offsets = np.linspace(-0.24, 0.24, len(table.index))
    x = np.arange(len(quantizers))
    for offset, backbone in zip(offsets, table.index):
        axis.scatter(
            x + offset,
            table.loc[backbone, quantizers].to_numpy(),
            s=30,
            label=backbone,
            edgecolors="white",
            linewidths=0.8,
            zorder=3,
            **styles[backbone],
        )
    axis.axhline(1.0, color=MUTED, linewidth=1, zorder=2)
    axis.set_xticks(x)
    axis.set_xticklabels([QUANT_LABELS[q] for q in quantizers])
    axis.set_ylabel("Retention (score / fp32 baseline)")
    axis.grid(axis="x", visible=False)
    axis.legend(
        ncol=len(table.index), loc="lower left", bbox_to_anchor=(0.0, 1.0), fontsize=8, handletextpad=0.2, columnspacing=1.0
    )
    axis.set_title("Quantization alone, full dimension", loc="left", pad=22)
    return figure


def reduction_only(table: pd.DataFrame):
    """Small multiples: fp32 retention by dimension, one panel per backbone, a line per DR method.

    `pca_ror` is omitted: at fp32 it equals `pca` exactly (rotation preserves cosine).
    """
    plt = _pyplot()
    backbones = [b for b in backbone_order(table.reset_index()) if b in table.index.get_level_values(0)]
    methods = [m for m in DR_SERIES if m != "pca_ror" and m in table.index.get_level_values(1)]
    styles = series_style(list(DR_SERIES))
    dims = list(table.columns)
    figure, axes = plt.subplots(1, len(backbones), figsize=(2.2 * len(backbones), 2.9), sharey=True)
    axes = np.atleast_1d(axes)
    for axis, backbone in zip(axes, backbones):
        for method in methods:
            if (backbone, method) not in table.index:
                continue
            axis.plot(
                dims,
                table.loc[(backbone, method)].to_numpy(),
                linewidth=2,
                markersize=4.5,
                label=method,
                **styles[method],
            )
        axis.axhline(1.0, color=MUTED, linewidth=1)
        _dim_axis(axis, dims)
        axis.set_title(backbone, loc="left")
        axis.set_xlabel("Target dimension")
    axes[0].set_ylabel("Retention, fp32")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, ncol=len(methods), loc="upper left", bbox_to_anchor=(0.0, 1.08), fontsize=8)
    figure.tight_layout()
    return figure


def budget_frontier(cells: pd.DataFrame, frontier: pd.DataFrame):
    """Best mean score at each storage size: every cell faint, each backbone's front as a line."""
    plt = _pyplot()
    backbones = backbone_order(cells)
    styles = series_style(backbones)
    figure, axis = plt.subplots(figsize=(7.0, 4.0))
    axis.scatter(cells["bytes_per_vector"], cells["score"], s=6, color="#c3c2b7", alpha=0.5, linewidths=0, zorder=1)
    for backbone in backbones:
        front = frontier[frontier["backbone"] == backbone].sort_values("bytes_per_vector")
        axis.plot(
            front["bytes_per_vector"],
            front["score"],
            linewidth=2,
            markersize=4.5,
            drawstyle="steps-post",
            label=backbone,
            zorder=3,
            **styles[backbone],
        )
    sizes = sorted(cells["bytes_per_vector"].unique())
    ticks = [s for s in (8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096) if sizes[0] <= s <= sizes[-1]]
    axis.set_xscale("log", base=2)
    axis.set_xticks(ticks)
    axis.set_xticklabels([str(t) for t in ticks])
    axis.minorticks_off()
    axis.set_xlabel("Bytes per vector")
    axis.set_ylabel("Mean nDCG@10")
    axis.set_title("Best configuration per storage budget (gray: every cell)", loc="left")
    axis.legend(loc="lower right", fontsize=8)
    return figure


def _heatmap(table: pd.DataFrame, *, title: str, xlabel: str, ylabel: str, unit: str = "pts", limit: float | None = None):
    plt = _pyplot()
    values = table.to_numpy(dtype=float)
    limit = limit or max(float(np.nanmax(np.abs(values))), 1e-9)
    height = 0.75 + 0.32 * len(table.index)
    figure, axis = plt.subplots(figsize=(1.2 + 0.85 * len(table.columns), height))
    image = axis.imshow(values, cmap=_diverging_cmap(), vmin=-limit, vmax=limit, aspect="auto")
    axis.grid(False)
    axis.set_xticks(range(len(table.columns)))
    axis.set_xticklabels([str(c) for c in table.columns])
    axis.set_yticks(range(len(table.index)))
    axis.set_yticklabels([str(i) for i in table.index])
    axis.tick_params(length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)
    for (row, column), value in np.ndenumerate(values):
        if np.isnan(value):
            continue
        strong = abs(value) > 0.55 * limit
        axis.text(column, row, f"{value:+.1f}", ha="center", va="center", fontsize=8, color="white" if strong else INK)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.04, pad=0.02)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(colors=MUTED, labelcolor=INK_SECONDARY, labelsize=8)
    colorbar.set_label(unit, color=INK_SECONDARY)
    axis.set_title(title, loc="left")
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    return figure


def ror_gain(gain: pd.DataFrame):
    """Heatmap: PCA+ROR minus PCA (points of retention), mean over backbones."""
    table = gain.pivot_table(index="quant_method", columns="target_dim", values="gain", aggfunc="mean")
    table = table.loc[[q for q in QUANT_ORDER if q in table.index and q not in ("none", "fp16")]]
    table.index = [QUANT_LABELS[q] for q in table.index]
    return _heatmap(
        table,
        title="PCA+ROR minus PCA, mean over backbones",
        xlabel="Target dimension",
        ylabel="Quantizer",
    )


def equal_count_gain(gain: pd.DataFrame):
    """Heatmap: equal-count minus uniform int at the same bits (points), mean over backbones and dimensions."""
    table = gain.pivot_table(index="dr_method", columns="bits", values="gain", aggfunc="mean")
    table = table.loc[[m for m in ("none",) + DR_SERIES if m in table.index], sorted(table.columns, reverse=True)]
    table.columns = [f"{bits} bit" for bits in table.columns]
    return _heatmap(
        table,
        title="Equal-count minus int, same bits",
        xlabel="Bits per dimension",
        ylabel="DR method",
    )


def prefix_gap(gap: pd.DataFrame):
    """Lines: truncation minus random selection (points) by dimension, one per backbone."""
    plt = _pyplot()
    backbones = backbone_order(gap)
    styles = series_style(backbones)
    figure, axis = plt.subplots(figsize=(5.2, 3.4))
    dims = sorted(gap["target_dim"].unique())
    for backbone in backbones:
        part = gap[gap["backbone"] == backbone].sort_values("target_dim")
        axis.plot(part["target_dim"], part["gain"], linewidth=2, markersize=4.5, label=backbone, **styles[backbone])
    axis.axhline(0.0, color=MUTED, linewidth=1)
    _dim_axis(axis, dims)
    axis.set_xlabel("Kept dimensions")
    axis.set_ylabel("Truncate minus random selection (pts)")
    axis.set_title("Is the prefix privileged? (MRL, fp32)", loc="left")
    axis.legend(fontsize=8, loc="lower right")
    return figure


def task_deltas(table: pd.DataFrame):
    """Heatmap: mean delta per task (points of nDCG@10) for selected cells."""
    table = table.sort_values(table.columns[0])
    table.index = [task.removeprefix("Nano").removesuffix("Retrieval") for task in table.index]
    return _heatmap(
        table,
        title="Score change per task, mean over backbones",
        xlabel="Configuration",
        ylabel="Task",
        unit="nDCG@10 pts",
    )


def centring_check(table: pd.DataFrame, *, title: str):
    """Lines: top-k agreement with fp32 by dimension, per reduction."""
    plt = _pyplot()
    names = list(table.columns)
    # Same colours as the DR methods elsewhere; uncentred PCA gets a slot of its own.
    dr_styles = series_style(list(DR_SERIES))
    styles = {
        "pca_centred": dr_styles["pca"],
        "pca_uncentred": {"color": CATEGORICAL[6], "marker": MARKERS[6]},
        **{name: dr_styles[name] for name in names if name in dr_styles},
    }
    figure, axis = plt.subplots(figsize=(5.2, 3.4))
    dims = list(table.index)
    for name in names:
        axis.plot(dims, table[name], linewidth=2, markersize=4.5, label=name.replace("_", " "), **styles[name])
    axis.axhline(1.0, color=MUTED, linewidth=1)
    _dim_axis(axis, dims)
    axis.set_xlabel("Target dimension")
    axis.set_ylabel("Top-10 overlap with fp32")
    axis.set_title(title, loc="left")
    axis.legend(fontsize=8, loc="lower right")
    return figure
