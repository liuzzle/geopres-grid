import pytest

from geopres_grid.backbones import all_backbones, get
from geopres_grid.grid import GRID_QUANT_METHODS, grid_cells, shard


def test_every_backbone_gets_the_same_cells_below_its_native_dimension():
    for backbone in all_backbones():
        cells = grid_cells(backbone)
        assert len(cells) == len(set(cells))
        assert all(c.target_dim is None or c.target_dim < backbone.native_dim for c in cells)
        assert sum(c.is_identity for c in cells) == 1
        assert not any(c.dr_method == "geopres" for c in cells)
        assert not any(c.quant_method.startswith("uint") for c in cells)


def test_a_dimension_not_below_the_native_one_is_skipped():
    harrier = get("harrier-270m")  # 640-d
    dims = {c.target_dim for c in grid_cells(harrier, target_dims=(128, 640, 768))}
    assert dims == {None, 128}


def test_grid_size():
    # (no DR + 5 methods x 4 dims) x 9 quantizers, for every backbone (all have d > 512).
    assert len(grid_cells(get("mgte"))) == (1 + 5 * 4) * len(GRID_QUANT_METHODS) == 189


def test_shards_partition_the_grid():
    cells = grid_cells(get("harrier-06b"))
    parts = [shard(cells, index, 8) for index in range(8)]
    assert sorted(map(str, (c for part in parts for c in part))) == sorted(map(str, cells))
    assert max(map(len, parts)) - min(map(len, parts)) <= 1
    with pytest.raises(ValueError, match="outside"):
        shard(cells, 8, 8)


def test_seeds_multiply_only_the_seeded_methods():
    mgte = get("mgte")
    cells = grid_cells(mgte, dr_seeds=(42, 1, 2))
    assert len(cells) == len(set(cells))
    for method in ("none", "truncate"):
        assert all(c.dr_seed is None for c in cells if c.dr_method == method)
        assert sum(c.dr_method == method for c in cells) == sum(c.dr_method == method for c in grid_cells(mgte))
    for method in ("pca", "pca_ror", "random_projection", "random_selection"):
        assert {c.dr_seed for c in cells if c.dr_method == method} == {None, 1, 2}
    # 1 + 1 method x 4 dims + 4 seeded methods x 4 dims x 3 seeds, 9 quantizers each.
    assert len(cells) == (1 + 4 + 4 * 4 * 3) * len(GRID_QUANT_METHODS)


def test_seed_42_keeps_the_run_ids_and_other_seeds_change_them():
    mgte = get("mgte")
    default = grid_cells(mgte)
    assert grid_cells(mgte, dr_seeds=(42,)) == default
    assert [c.hash for c in grid_cells(mgte, dr_seeds=(42,))] == [c.hash for c in default]
    random = dict(dr_methods=("random_projection",), target_dims=(128,), quant_methods=("none",))
    seeded = grid_cells(mgte, dr_seeds=(42, 42, 1, 2), **random)
    assert [c.dr_seed for c in seeded] == [None, None, 1, 2]  # the no-DR cell, then one per seed
    assert len({c.hash for c in seeded}) == 4
    assert seeded[1].hash == grid_cells(mgte, **random)[1].hash


def test_run_grid_lists_one_cell_per_seed(capsys):
    from run_grid import main

    argv = ["--backbone", "mgte", "--list", "--dr-method", "random_selection", "--target-dim", "64"]
    assert main(argv + ["--quant-method", "int8", "--dr-seed", "1", "--dr-seed", "2"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[:3] == ["none None int8", "random_selection 64 int8 seed=1", "random_selection 64 int8 seed=2"]
    assert main(argv) == 0
    assert sum("seed=42" in line for line in capsys.readouterr().out.splitlines()) == len(GRID_QUANT_METHODS)
