"""Figures for the tables in `analysis.py`. Needs the `analysis` extra (matplotlib).

One function per figure, each taking the analysis table it draws and returning the
matplotlib Figure; `save` writes PNG and PDF side by side. Colours follow the
entity, never its rank: a backbone keeps its colour in every figure, set by
registry order, and a DR method likewise. Palette from the dataviz reference
instance, validated for colour-vision deficiency on adjacent pairs; three of the
hues are below 3:1 on white, so every series also carries its own marker and a
legend, and the numbers are in the CSVs next to the figures.

A relative view (retention, delta, gain) shows the absolute score next to it, as a
second row of panels or in the labels; the per-backbone figures (`*_by_backbone`)
facet what the others average over backbones.
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

BASELINE_STYLE = {"color": INK, "marker": "*", "markersize": 9, "linestyle": "none"}
"""The no-reduction fp32 point wherever a curve ends at full precision."""

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


def quantization_only(table: pd.DataFrame, scores: pd.DataFrame | None = None):
    """Dot plot: retention per quantizer at full dimension, one marker per backbone.

    With `scores` (`analysis.quantization_only(cells, "score")`) a second panel
    shows the absolute mean nDCG@10, fp32 first.
    """
    plt = _pyplot()
    styles = series_style(list(table.index))
    panels = [(table, "Retention (score / fp32 baseline)", True)]
    if scores is not None:
        panels.append((scores, "Mean nDCG@10", False))
    quantizers = [q for q in QUANT_ORDER if q in table.columns and (q != "none" or scores is not None)]
    figure, axes = plt.subplots(len(panels), 1, figsize=(7.0, 3.0 * len(panels) + 0.4), sharex=True, squeeze=False)
    offsets = np.linspace(-0.24, 0.24, len(table.index))
    x = np.arange(len(quantizers))
    for axis, (values, ylabel, reference) in zip(axes[:, 0], panels):
        for offset, backbone in zip(offsets, table.index):
            axis.scatter(
                x + offset,
                values.loc[backbone, quantizers].to_numpy(),
                s=30,
                label=backbone,
                edgecolors="white",
                linewidths=0.8,
                zorder=3,
                **styles[backbone],
            )
        if reference:
            axis.axhline(1.0, color=MUTED, linewidth=1, zorder=2)
        axis.set_ylabel(ylabel)
        axis.grid(axis="x", visible=False)
    axes[-1, 0].set_xticks(x)
    axes[-1, 0].set_xticklabels([QUANT_LABELS[q] for q in quantizers])
    axes[0, 0].legend(
        ncol=len(table.index), loc="lower left", bbox_to_anchor=(0.0, 1.0), fontsize=8, handletextpad=0.2, columnspacing=1.0
    )
    axes[0, 0].set_title("Quantization alone, full dimension", loc="left", pad=22)
    figure.tight_layout()
    return figure


def reduction_only(table: pd.DataFrame, scores: pd.DataFrame | None = None):
    """Small multiples: fp32 retention by dimension, one panel per backbone, a line per DR method.

    Each line ends, dotted, at the backbone's native dimension: the fp32 baseline
    with no reduction (the star), the last column `analysis.reduction_only` fills
    for that backbone. With `scores` a second row shows mean nDCG@10. `pca_ror` is
    omitted: at fp32 it equals `pca` exactly (rotation preserves cosine).
    """
    plt = _pyplot()
    backbones = [b for b in backbone_order(table.reset_index()) if b in table.index.get_level_values(0)]
    methods = [m for m in DR_SERIES if m != "pca_ror" and m in table.index.get_level_values(1)]
    styles = series_style(list(DR_SERIES))
    panels = [(table, "Retention, fp32")] + ([(scores, "Mean nDCG@10, fp32")] if scores is not None else [])
    figure, axes = plt.subplots(
        len(panels), len(backbones), figsize=(2.3 * len(backbones), 2.7 * len(panels) + 0.3), sharey="row", squeeze=False
    )
    for row, (values, ylabel) in enumerate(panels):
        for axis, backbone in zip(axes[row], backbones):
            filled = values.loc[backbone].dropna(axis=1, how="all")
            native = max(filled.columns)
            dims = [d for d in filled.columns if d != native]
            for method in methods:
                if method not in filled.index:
                    continue
                line = filled.loc[method]
                axis.plot(dims, line[dims].to_numpy(), linewidth=2, markersize=4.5, label=method, **styles[method])
                axis.plot(
                    [dims[-1], native], line[[dims[-1], native]].to_numpy(), linewidth=1.2, linestyle=":",
                    color=styles[method]["color"],
                )
            axis.plot([native], [filled[native].iloc[0]], zorder=4, label="fp32, no reduction", **BASELINE_STYLE)
            if row == 0:
                axis.axhline(1.0, color=MUTED, linewidth=1)
                axis.set_title(backbone, loc="left")
            _dim_axis(axis, dims + [native])
            # The native tick a line lower: 640 sits too close to 512 on a log axis.
            axis.set_xticklabels([str(int(d)) for d in dims] + [f"\n{native}"])
            axis.tick_params(axis="x", labelsize=7.5)
            if row == len(panels) - 1:
                axis.set_xlabel("Target dimension")
        axes[row, 0].set_ylabel(ylabel)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, ncol=len(labels), loc="upper left", bbox_to_anchor=(0.0, 1.05), fontsize=8)
    figure.tight_layout()
    return figure


def _bytes_axis(axis, sizes) -> None:
    ticks = [s for s in (8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096) if min(sizes) <= s <= max(sizes)]
    axis.set_xscale("log", base=2)
    axis.set_xticks(ticks)
    axis.set_xticklabels([str(t) for t in ticks])
    axis.minorticks_off()


def _baseline_cells(cells: pd.DataFrame) -> pd.DataFrame:
    return cells[(cells["dr_method"] == "none") & (cells["quant_method"] == "none")].set_index("backbone")


def _front_line(axis, front: pd.DataFrame, style: dict[str, Any], label: str | None) -> None:
    front = front.sort_values("bytes_per_vector")
    axis.plot(
        front["bytes_per_vector"], front["score"], linewidth=2, markersize=4.5, drawstyle="steps-post",
        label=label, zorder=3, **style,
    )


def _baseline_point(axis, baseline: pd.Series, style: dict[str, Any], label: str | None = None) -> None:
    """fp32 reference: a large hollow marker in the backbone's colour, at its bytes per vector."""
    axis.plot(
        [baseline["bytes_per_vector"]], [baseline["score"]], marker=style["marker"], markersize=11,
        markerfacecolor="none", markeredgecolor=style["color"], markeredgewidth=1.6, linestyle="none",
        zorder=4, label=label,
    )


def budget_frontier(cells: pd.DataFrame, frontier: pd.DataFrame):
    """Best mean score at each storage size: every cell faint, each backbone's front as a line.

    Each backbone's uncompressed fp32 baseline is the large hollow marker at its own
    bytes per vector (4 x native dimension), whether or not it is on the front.
    """
    plt = _pyplot()
    backbones = backbone_order(cells)
    styles = series_style(backbones)
    baselines = _baseline_cells(cells)
    figure, axis = plt.subplots(figsize=(7.0, 4.0))
    axis.scatter(cells["bytes_per_vector"], cells["score"], s=6, color="#c3c2b7", alpha=0.5, linewidths=0, zorder=1)
    for backbone in backbones:
        _front_line(axis, frontier[frontier["backbone"] == backbone], styles[backbone], backbone)
        _baseline_point(axis, baselines.loc[backbone], styles[backbone])
    axis.plot([], [], marker="o", markersize=9, markerfacecolor="none", markeredgecolor=INK_SECONDARY,
              linestyle="none", label="fp32 baseline (hollow)")
    _bytes_axis(axis, cells["bytes_per_vector"])
    axis.set_xlabel("Bytes per vector")
    axis.set_ylabel("Mean nDCG@10")
    axis.set_title("Best configuration per storage budget (gray: every cell)", loc="left")
    axis.legend(loc="lower right", fontsize=8)
    return figure


def budget_frontier_by_backbone(cells: pd.DataFrame, frontier: pd.DataFrame):
    """One panel per backbone: its cells, its front with each front cell named, its fp32 baseline."""
    plt = _pyplot()
    backbones = backbone_order(cells)
    styles = series_style(backbones)
    baselines = _baseline_cells(cells)
    ncols = min(3, len(backbones))
    nrows = int(np.ceil(len(backbones) / ncols))
    figure, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.6 * nrows), sharey=True, squeeze=False)
    for axis, backbone in zip(axes.flat, backbones):
        own = cells[cells["backbone"] == backbone]
        front = frontier[frontier["backbone"] == backbone].sort_values("bytes_per_vector")
        axis.scatter(own["bytes_per_vector"], own["score"], s=7, color="#c3c2b7", alpha=0.7, linewidths=0, zorder=1)
        _front_line(axis, front, styles[backbone], None)
        _baseline_point(axis, baselines.loc[backbone], styles[backbone])
        for _, cell in front.iterrows():
            axis.annotate(
                cell["label"].replace("random_", "r").replace("equal_count_", "EC-"), (cell["bytes_per_vector"], cell["score"]),
                xytext=(2, -4), textcoords="offset points", fontsize=5.5, color=INK_SECONDARY, rotation=-60,
                ha="left", va="top", zorder=5,
            )
        _bytes_axis(axis, cells["bytes_per_vector"])
        axis.set_title(f"{backbone}  (fp32 {baselines.loc[backbone, 'score']:.3f})", loc="left")
        axis.set_xlabel("Bytes per vector")
    for axis in axes.flat[len(backbones):]:
        axis.set_visible(False)
    for axis in axes[:, 0]:
        axis.set_ylabel("Mean nDCG@10")
    figure.tight_layout()
    return figure


def _heatmap_on(axis, table: pd.DataFrame, limit: float, scores: pd.DataFrame | None = None, fontsize: float = 8):
    """Draw `table` as an annotated diverging heatmap on `axis`; returns the image.

    With `scores` (same shape, mean nDCG@10) each cell also carries the absolute
    score under its difference, smaller, so a loss reads against where it lands.
    """
    values = table.to_numpy(dtype=float)
    image = axis.imshow(values, cmap=_diverging_cmap(), vmin=-limit, vmax=limit, aspect="auto")
    axis.grid(False)
    axis.set_xticks(range(len(table.columns)))
    labels = [str(c) for c in table.columns]
    if max(map(len, labels)) > 8:  # configuration names: slanted, or neighbours overlap
        axis.set_xticklabels(labels, rotation=35, ha="right", rotation_mode="anchor")
    else:
        axis.set_xticklabels(labels)
    axis.set_yticks(range(len(table.index)))
    axis.set_yticklabels([str(i) for i in table.index])
    axis.tick_params(length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)
    absolute = None if scores is None else scores.reindex(index=table.index, columns=table.columns).to_numpy(dtype=float)
    for (row, column), value in np.ndenumerate(values):
        if np.isnan(value):
            continue
        color = "white" if abs(value) > 0.55 * limit else INK
        if absolute is None or np.isnan(absolute[row, column]):
            axis.text(column, row, f"{value:+.1f}", ha="center", va="center", fontsize=fontsize, color=color)
            continue
        axis.text(column, row - 0.17, f"{value:+.1f}", ha="center", va="center", fontsize=fontsize, color=color)
        axis.text(column, row + 0.22, f"{absolute[row, column]:.3f}", ha="center", va="center",
                  fontsize=fontsize - 2, color=color, alpha=0.85)
    return image


def _colorbar(figure, image, axes, unit: str) -> None:
    colorbar = figure.colorbar(image, ax=axes, fraction=0.04, pad=0.02)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(colors=MUTED, labelcolor=INK_SECONDARY, labelsize=8)
    colorbar.set_label(unit, color=INK_SECONDARY)


def _limit(tables: list[pd.DataFrame]) -> float:
    return max([float(np.nanmax(np.abs(t.to_numpy(dtype=float)))) for t in tables if t.size] + [1e-9])


def _heatmap(
    table: pd.DataFrame,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    unit: str = "pts",
    limit: float | None = None,
    scores: pd.DataFrame | None = None,
):
    plt = _pyplot()
    limit = limit or _limit([table])
    height = 0.75 + (0.32 if scores is None else 0.45) * len(table.index)
    figure, axis = plt.subplots(figsize=(1.2 + 0.85 * len(table.columns), height))
    image = _heatmap_on(axis, table, limit, scores)
    _colorbar(figure, image, axis, unit)
    axis.set_title(title, loc="left")
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    return figure


def _heatmap_grid(
    tables: dict[str, pd.DataFrame],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    unit: str = "pts",
    ncols: int = 3,
    scores: dict[str, pd.DataFrame] | None = None,
):
    """One heatmap per backbone on a shared colour scale (so panels compare), one colorbar.
    `scores` (same keys) adds each cell's absolute score, as in `_heatmap_on`."""
    plt = _pyplot()
    limit = _limit(list(tables.values()))
    first = next(iter(tables.values()))
    ncols = min(ncols, len(tables))
    nrows = int(np.ceil(len(tables) / ncols))
    width = 1.6 + (0.62 if scores is None else 0.7) * len(first.columns)
    height = 0.9 + (0.3 if scores is None else 0.42) * len(first.index)
    figure, axes = plt.subplots(nrows, ncols, figsize=(width * ncols + 0.8, height * nrows + 0.4), squeeze=False)
    for axis, (name, table) in zip(axes.flat, tables.items()):
        image = _heatmap_on(axis, table, limit, None if scores is None else scores[name])
        axis.set_title(name, loc="left")
    for axis in axes.flat[len(tables):]:
        axis.set_visible(False)
    for axis in axes[-1]:
        axis.set_xlabel(xlabel)
    for axis in axes[:, 0]:
        axis.set_ylabel(ylabel)
    figure.suptitle(title, x=0.01, ha="left", fontsize=10, color=INK)
    top = 1 - 0.35 / figure.get_figheight()  # room for the suptitle on tall figures
    figure.tight_layout(rect=(0, 0, 0.93, top))
    _colorbar(figure, image, list(axes.flat[: len(tables)]), unit)
    return figure


def ror_gain(gain: pd.DataFrame):
    """Heatmap: PCA+ROR minus PCA (points of retention), mean over backbones, with
    PCA+ROR's mean nDCG@10 under each difference."""
    return _heatmap(
        _ror_table(gain),
        title="PCA+ROR minus PCA, mean over backbones (small: PCA+ROR nDCG@10)",
        xlabel="Target dimension",
        ylabel="Quantizer",
        scores=_ror_table(gain, "better_score"),
    )


def _ror_table(gain: pd.DataFrame, value: str = "gain") -> pd.DataFrame:
    table = gain.pivot_table(index="quant_method", columns="target_dim", values=value, aggfunc="mean")
    table = table.loc[[q for q in QUANT_ORDER if q in table.index and q not in ("none", "fp16")]]
    table.index = [QUANT_LABELS[q] for q in table.index]
    return table


def ror_gain_by_backbone(gain: pd.DataFrame):
    """`ror_gain` per backbone, on one colour scale."""
    backbones = backbone_order(gain)
    return _heatmap_grid(
        {b: _ror_table(gain[gain["backbone"] == b]) for b in backbones},
        title="PCA+ROR minus PCA, per backbone (pts of retention; small: PCA+ROR nDCG@10)",
        xlabel="Target dimension",
        ylabel="Quantizer",
        scores={b: _ror_table(gain[gain["backbone"] == b], "better_score") for b in backbones},
    )


def _equal_count_table(gain: pd.DataFrame, value: str = "gain") -> pd.DataFrame:
    table = gain.pivot_table(index="dr_method", columns="bits", values=value, aggfunc="mean")
    table = table.loc[[m for m in ("none",) + DR_SERIES if m in table.index], sorted(table.columns, reverse=True)]
    table.columns = [f"{bits} bit" for bits in table.columns]
    return table


def equal_count_gain_by_backbone(gain: pd.DataFrame):
    """`equal_count_gain` per backbone (mean over dimensions), on one colour scale."""
    backbones = backbone_order(gain)
    return _heatmap_grid(
        {b: _equal_count_table(gain[gain["backbone"] == b]) for b in backbones},
        title="Equal-count minus int at the same bits, per backbone (pts, mean over dimensions; small: equal-count nDCG@10)",
        xlabel="Bits per dimension",
        ylabel="DR method",
        ncols=5,
        scores={b: _equal_count_table(gain[gain["backbone"] == b], "better_score") for b in backbones},
    )


def equal_count_gain(gain: pd.DataFrame):
    """Heatmap: equal-count minus uniform int at the same bits (points), mean over backbones and dimensions."""
    return _heatmap(
        _equal_count_table(gain),
        title="Equal-count minus int, same bits (small: equal-count nDCG@10)",
        xlabel="Bits per dimension",
        ylabel="DR method",
        scores=_equal_count_table(gain, "better_score"),
    )


def _all_cells_panel(retention: pd.DataFrame, score: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One backbone's (or the mean's) block of `analysis.all_cells`, as points of
    retention lost against fp32 and the absolute score, with readable row and column names."""
    loss = (retention - 1) * 100
    names = ["full" if method == "none" else f"{method}-{int(dim)}" for method, dim in loss.index]
    columns = [QUANT_LABELS.get(q, q) for q in loss.columns]
    loss, score = loss.copy(), score.reindex(index=retention.index, columns=retention.columns).copy()
    for table in (loss, score):
        table.index, table.columns = names, columns
    return loss, score


def _method_separators(axes, index: list[str]) -> None:
    """A white rule between DR methods, so the 21 rows read as six blocks."""
    methods = [name.rsplit("-", 1)[0] for name in index]
    for row in range(1, len(methods)):
        if methods[row] != methods[row - 1]:
            for axis in axes:
                axis.axhline(row - 0.5, color="white", linewidth=2.5)


def all_cells(retention: pd.DataFrame, score: pd.DataFrame):
    """Heatmap of the whole grid, mean over backbones: every (DR, dimension) by every
    quantizer, coloured by points of retention against fp32, with the mean nDCG@10
    under each. Takes `analysis.all_cells` for retention and for score."""
    mean_retention = retention.groupby(level=["dr_method", "target_dim"], sort=False).mean()
    mean_score = score.groupby(level=["dr_method", "target_dim"], sort=False).mean()
    loss, absolute = _all_cells_panel(mean_retention, mean_score)
    figure = _heatmap(
        loss,
        title="Every configuration, mean over backbones: pts of retention vs fp32 (small: nDCG@10)",
        xlabel="Quantizer",
        ylabel="Reduction",
        unit="pts of retention",
        scores=absolute,
    )
    _method_separators(figure.axes[:1], list(loss.index))
    return figure


def all_cells_by_backbone(retention: pd.DataFrame, score: pd.DataFrame, ncols: int = 3):
    """`all_cells` per backbone on one colour scale."""
    tables, scores = {}, {}
    for backbone in retention.index.get_level_values("backbone").unique():
        tables[backbone], scores[backbone] = _all_cells_panel(retention.loc[backbone], score.loc[backbone])
    figure = _heatmap_grid(
        tables,
        title="Every configuration per backbone: pts of retention vs fp32 (small: nDCG@10)",
        xlabel="Quantizer",
        ylabel="Reduction",
        unit="pts of retention",
        ncols=ncols,
        scores=scores,
    )
    for axis, table in zip(figure.axes, tables.values()):
        _method_separators([axis], list(table.index))
    return figure


def all_cells_per_backbone(retention: pd.DataFrame, score: pd.DataFrame) -> dict[str, Any]:
    """`all_cells_by_backbone` split into one figure per backbone (one page each),
    still on one colour scale so figures compare. Keyed by backbone."""
    panels = {
        backbone: _all_cells_panel(retention.loc[backbone], score.loc[backbone])
        for backbone in retention.index.get_level_values("backbone").unique()
    }
    limit = _limit([loss for loss, _ in panels.values()])
    figures = {}
    for backbone, (loss, absolute) in panels.items():
        figure = _heatmap(
            loss,
            title=f"{backbone}: every configuration, pts of retention vs fp32 (small: nDCG@10)",
            xlabel="Quantizer",
            ylabel="Reduction",
            unit="pts of retention",
            limit=limit,
            scores=absolute,
        )
        _method_separators(figure.axes[:1], list(loss.index))
        figures[backbone] = figure
    return figures


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
    axis.set_title("Comparison: keep first k vs. k random coordinates  (MRL, fp32)", loc="left")
    axis.legend(fontsize=8, loc="lower right")
    return figure


def _task_name(task: str) -> str:
    return task.removeprefix("Nano").removesuffix("Retrieval")


def task_deltas(table: pd.DataFrame, baseline: pd.Series | None = None):
    """Heatmap: mean delta per task (points of nDCG@10) for selected cells.

    With `baseline` (task -> mean fp32 nDCG@10, the `fp32` column of
    `analysis.task_scores`) each task's label carries its absolute starting point.
    """
    table = table.sort_values(table.columns[0])
    labels = [_task_name(task) for task in table.index]
    if baseline is not None:
        labels = [f"{label} ({baseline[task]:.2f})" for label, task in zip(labels, table.index)]
    table.index = labels
    return _heatmap(
        table,
        title="Score change per task, mean over backbones (fp32 nDCG@10 in brackets)"
        if baseline is not None
        else "Score change per task, mean over backbones",
        xlabel="Configuration",
        ylabel="Task",
        unit="nDCG@10 pts",
    )


def task_deltas_by_backbone(losses: pd.DataFrame, configs: list[str] | None = None):
    """`task_deltas` per backbone (points of nDCG@10), tasks in one order across panels."""
    configs = configs or list(dict.fromkeys(losses["config"]))
    tables = {}
    order = None
    for backbone in backbone_order(losses):
        part = losses[losses["backbone"] == backbone]
        table = part.pivot_table(index="task", columns="config", values="delta", aggfunc="mean")[configs] * 100
        order = order if order is not None else table.sort_values(configs[0]).index
        table = table.reindex(order)
        table.index = [_task_name(task) for task in table.index]
        tables[backbone] = table
    return _heatmap_grid(
        tables,
        title="Score change per task, per backbone (nDCG@10 pts)",
        xlabel="Configuration",
        ylabel="Task",
        unit="nDCG@10 pts",
        ncols=3,
    )


def corpus_size_loss(
    losses: pd.DataFrame,
    sizes: pd.DataFrame,
    correlation: pd.DataFrame | None = None,
    value: str = "retention",
    ylabel: str | None = None,
):
    """Small multiples, one per named cell: per-task loss against corpus size (log).

    Points are the task mean over backbones, the vertical bar spans the backbones.
    With a `tier` column in `losses` (and `sizes`), each tier gets its own marker.
    `correlation` (`analysis.size_correlation` on the same data, backbone "mean",
    no tier column) puts Spearman's rho in each title.
    """
    plt = _pyplot()
    configs = list(dict.fromkeys(losses["config"]))
    keys = ["task"] + (["tier"] if "tier" in losses.columns else [])
    sized = losses.merge(sizes.reset_index()[keys + ["n_documents"]], on=keys, how="left").dropna(subset=["n_documents"])
    tiers = list(dict.fromkeys(sized["tier"])) if "tier" in sized.columns else [None]
    styles = series_style([str(t) for t in tiers])
    ncols = min(4, len(configs))
    nrows = int(np.ceil(len(configs) / ncols))
    figure, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.6 * nrows + 0.4), sharex=True, sharey=True, squeeze=False)
    for axis, config in zip(axes.flat, configs):
        part = sized[sized["config"] == config]
        for tier in tiers:
            own = part if tier is None else part[part["tier"] == tier]
            stats = own.groupby("task").agg(
                mean=(value, "mean"), low=(value, "min"), high=(value, "max"), n_documents=("n_documents", "first")
            )
            style = styles[str(tier)]
            axis.vlines(stats["n_documents"], stats["low"], stats["high"], color=style["color"], alpha=0.35, linewidth=1.2)
            axis.scatter(
                stats["n_documents"], stats["mean"], s=22, edgecolors="white", linewidths=0.6, zorder=3,
                label=None if tier is None else str(tier), **style,
            )
        axis.axhline(1.0 if value == "retention" else 0.0, color=MUTED, linewidth=1)
        title = config
        if correlation is not None:
            match = correlation[(correlation["config"] == config) & (correlation["backbone"] == "mean")]
            if "tier" in match.columns:
                match = match[match["tier"].isna()]
            if len(match) and not np.isnan(match["spearman"].iloc[0]):
                title += f"\nrho = {match['spearman'].iloc[0]:+.2f} (p = {match['spearman_p'].iloc[0]:.2g}, n = {match['n_tasks'].iloc[0]})"
        axis.set_title(title, loc="left", fontsize=8.5)
        axis.set_xscale("log")
    for axis in axes.flat[len(configs):]:
        axis.set_visible(False)
    for index, axis in enumerate(axes.flat[: len(configs)]):
        # The bottom axis of every column, also when the panel under it is empty.
        if index + ncols >= len(configs):
            axis.xaxis.set_tick_params(labelbottom=True)
            axis.set_xlabel("Documents in corpus")
    for axis in axes[:, 0]:
        axis.set_ylabel(ylabel or ("Retention" if value == "retention" else value))
    if tiers != [None]:
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(handles, labels, ncol=len(tiers), loc="upper left", bbox_to_anchor=(0.0, 1.02), fontsize=8)
    figure.tight_layout(rect=(0, 0, 1, 0.97 if tiers != [None] else 1))
    return figure


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


# --- Tier comparison ---------------------------------------------------------------


def tier_transfer(paired: pd.DataFrame, transfer: pd.DataFrame, small: str = "tier 1", large: str = "tier 2"):
    """Every cell's retention (top) and mean nDCG@10 (bottom) in both tiers, one panel per backbone.

    Colour and marker by DR method; the diagonal is "costs the same in both". Points
    below it cost more on the large tier. Spearman of retention in the titles.
    """
    plt = _pyplot()
    backbones = backbone_order(paired)
    styles = {"none": {"color": INK, "marker": "o"}, **series_style(list(DR_SERIES))}
    rho = transfer.set_index("backbone")["spearman_retention"]
    figure, axes = plt.subplots(2, len(backbones), figsize=(2.6 * len(backbones), 5.6), squeeze=False)
    for column, backbone in enumerate(backbones):
        part = paired[paired["backbone"] == backbone]
        for row, value in enumerate(("retention", "score")):
            axis = axes[row, column]
            for method in ["none", *DR_SERIES]:
                own = part[part["dr_method"] == method]
                if own.empty:
                    continue
                axis.scatter(
                    own[f"{value}_small"], own[f"{value}_large"], s=12, alpha=0.75, linewidths=0,
                    label="no DR" if method == "none" else method, **styles[method],
                )
            low = min(part[f"{value}_small"].min(), part[f"{value}_large"].min())
            high = max(part[f"{value}_small"].max(), part[f"{value}_large"].max())
            axis.plot([low, high], [low, high], color=MUTED, linewidth=1, zorder=0)
            axis.set_xlabel(f"{'Retention' if value == 'retention' else 'nDCG@10'}, {small}")
            if column == 0:
                axis.set_ylabel(f"{'Retention' if value == 'retention' else 'Mean nDCG@10'}, {large}")
        axes[0, column].set_title(f"{backbone}\nrho = {rho[backbone]:.3f}", loc="left")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, ncol=len(labels), loc="upper left", bbox_to_anchor=(0.0, 1.04), fontsize=8)
    figure.tight_layout()
    return figure


def scale_cost(table: pd.DataFrame, small: str = "tier 1", large: str = "tier 2"):
    """Heatmap: retention change large - small (points), mean over backbones, per DR cell and quantizer."""
    table = table.copy()
    table.index = ["full" if method == "none" else f"{method}-{dim}" for method, dim in table.index]
    table.columns = [QUANT_LABELS[q] for q in table.columns]
    return _heatmap(
        table,
        title=f"Retention {large} minus {small}, mean over backbones (pts)",
        xlabel="Quantizer",
        ylabel="DR method and dimension",
    )


def budget_choice(choice: pd.DataFrame, small: str = "tier 1", large: str = "tier 2"):
    """Per backbone: the large tier's best mean nDCG@10 per budget, and what the small
    tier's pick scores there. The gap between the two lines is the regret."""
    plt = _pyplot()
    backbones = backbone_order(choice)
    styles = series_style(backbones)
    ncols = min(3, len(backbones))
    nrows = int(np.ceil(len(backbones) / ncols))
    figure, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.2 * nrows), sharey=True, squeeze=False)
    for axis, backbone in zip(axes.flat, backbones):
        part = choice[choice["backbone"] == backbone].sort_values("budget")
        color = styles[backbone]["color"]
        axis.plot(part["budget"], part["large_best"], drawstyle="steps-post", color=color, linewidth=2,
                  label=f"best on {large}")
        axis.plot(part["budget"], part["large_score_of_small_pick"], drawstyle="steps-post", color=INK,
                  linewidth=1.2, linestyle="--", label=f"{small}'s pick, scored on {large}")
        differ = part[~part["same"]]
        axis.scatter(differ["budget"], differ["large_score_of_small_pick"], s=14, color=INK, zorder=3,
                     label="different pick")
        _bytes_axis(axis, part["budget"])
        same = part["same"].mean()
        axis.set_title(f"{backbone}  (same pick {same:.0%}, max regret {part['regret'].max():.1f} pts)",
                       loc="left", fontsize=9)
        axis.set_xlabel("Budget, bytes per vector")
    for axis in axes.flat[len(backbones):]:
        axis.set_visible(False)
    for axis in axes[:, 0]:
        axis.set_ylabel(f"Mean nDCG@10, {large}")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, ncol=3, loc="upper left", bbox_to_anchor=(0.0, 1.04), fontsize=8)
    figure.tight_layout()
    return figure
