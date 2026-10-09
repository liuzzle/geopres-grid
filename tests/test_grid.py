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
