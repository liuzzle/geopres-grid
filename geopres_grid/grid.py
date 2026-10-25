"""The grid of post-processing cells evaluated for every backbone.

One place for the cells, so tier 1 is run the same way for all five backbones and
a rerun after a timeout covers exactly what the first run did. `scripts/run_grid.py`
sweeps it; `scripts/evaluate.py` still scores any single cell outside it.
"""

from __future__ import annotations

from typing import Iterable

from geopres_grid.backbones import Backbone
from geopres_grid.identity import DEFAULT_DR_SEED, PCA_METHODS, SEEDED_DR_METHODS, PostProcConfig

GRID_TARGET_DIMS: tuple[int, ...] = (64, 128, 256, 512)
"""Shared across backbones, so the compression factor differs per model (768, 640,
1024 native). 128/256/512 are where all three MRL backbones are trained (README,
"Matryoshka support"); 64 adds the high-compression end. A dimension not below a
backbone's native one is skipped for it."""

GRID_DR_METHODS: tuple[str, ...] = (
    "truncate",
    "pca",
    "pca_ror",
    "random_projection",
    "random_selection",
)
"""Every reducer except `geopres`, which needs trained weights per backbone and
dimension and is run separately. `none` is always in the grid, once, without a
target dimension."""

GRID_QUANT_METHODS: tuple[str, ...] = (
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
"""The `uint*` quantizers are left out: under fake quantization they score exactly
as their `int*` counterparts (logging §15.4) and differ only for WP-G."""


def grid_cells(
    backbone: Backbone,
    *,
    dr_methods: Iterable[str] = GRID_DR_METHODS,
    target_dims: Iterable[int] = GRID_TARGET_DIMS,
    quant_methods: Iterable[str] = GRID_QUANT_METHODS,
    pca_mean: str | None = None,
    dr_seeds: Iterable[int | None] = (None,),
) -> list[PostProcConfig]:
    """Cells for `backbone`, in a fixed order: no DR first, then each method by
    dimension (and seed), every quantizer within. The identity cell (no DR, no
    quantization) is included; `evaluate_cell` scores it as the baseline without a
    second slot. `pca_mean` applies to the PCA methods only; None is their default,
    `"restore"`. `dr_seeds` multiplies `SEEDED_DR_METHODS` only, one cell per seed;
    every other method keeps its single cell. None and `DEFAULT_DR_SEED` are one seed
    (`PostProcConfig`), so the default grid keeps the run ids it was scored under."""
    dims = [dim for dim in target_dims if dim < backbone.native_dim]
    seeds = list(dict.fromkeys(None if seed == DEFAULT_DR_SEED else seed for seed in dr_seeds))
    reductions: list[tuple[str, int | None, int | None]] = [("none", None, None)]
    reductions += [
        (method, dim, seed)
        for method in dr_methods
        if method != "none"
        for dim in dims
        for seed in (seeds if method in SEEDED_DR_METHODS else [None])
    ]
    return [
        PostProcConfig(
            dr_method=method,
            target_dim=dim,
            dr_seed=seed,
            quant_method=quant,
            pca_mean=pca_mean if method in PCA_METHODS else None,
        )
        for method, dim, seed in reductions
        for quant in quant_methods
    ]


def shard(cells: list[PostProcConfig], index: int, count: int) -> list[PostProcConfig]:
    """Every `count`-th cell from `index`, so each shard gets a mix of methods and
    dimensions rather than one slow block."""
    if not 0 <= index < count:
        raise ValueError(f"shard index {index} is outside 0..{count - 1}")
    return cells[index::count]
