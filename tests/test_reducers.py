import numpy as np
import pytest

from geopres_grid.identity import PostProcConfig, weights_id
from geopres_grid.reducers import GeoPres
from geopres_grid.reducers import Identity
from geopres_grid.reducers import PCA
from geopres_grid.reducers import PCARotation
from geopres_grid.reducers import PostProcessingPipeline
from geopres_grid.reducers import RandomProjection
from geopres_grid.reducers import RandomSelection
from geopres_grid.reducers import Truncate
from geopres_grid.reducers import apply_reduction
from geopres_grid.reducers import reducer_from_config


def test_truncate_is_a_slice():
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    reducer = Truncate(2)
    reducer.fit(values)

    result = reducer.transform(values)

    np.testing.assert_array_equal(result, values[:, :2])
    assert reducer.overhead_class == "slice"


def test_projection_reducers_fit_transform_and_reload(tmp_path):
    values = np.arange(30, dtype=np.float32).reshape(6, 5)

    reducers = [
        PCA(2),
        RandomProjection(2, seed=7),
        RandomSelection(2, seed=7),
    ]
    for reducer in reducers:
        reducer.fit(values)
        expected = reducer.transform(values)
        path = tmp_path / f"{type(reducer).__name__}.npz"
        reducer.save(path)
        loaded = type(reducer).load(path)
        np.testing.assert_allclose(loaded.transform(values), expected)


def test_geopres_projects_with_weight_rows_and_reloads(tmp_path):
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    weights = np.array([[1, 0, 0, 0], [0, 0, 1, 0]], dtype=np.float32)
    reducer = GeoPres(weights=weights)
    reducer.fit(values)
    expected = values[:, [0, 2]]
    np.testing.assert_array_equal(reducer.transform(values), expected)

    path = tmp_path / "geopres.npy"
    reducer.save(path)
    np.testing.assert_array_equal(GeoPres.load(path).transform(values), expected)


def test_pca_rotation_reloads_without_changing_output(tmp_path):
    values = np.arange(30, dtype=np.float32).reshape(6, 5)
    reducer = PCARotation(2, seed=7)
    reducer.fit(values)
    expected = reducer.transform(values)

    path = tmp_path / "pca_ror.npz"
    reducer.save(path)

    np.testing.assert_allclose(PCARotation.load(path).transform(values), expected)


def test_reducer_from_config_selects_configured_methods():
    assert type(reducer_from_config(PostProcConfig(), 8)) is Identity
    assert type(reducer_from_config(PostProcConfig(dr_method="truncate", target_dim=2), 8)) is Truncate
    assert type(reducer_from_config(PostProcConfig(dr_method="pca_ror", target_dim=2), 8)) is PCARotation
    assert type(reducer_from_config(PostProcConfig(dr_method="random_selection", target_dim=2), 8)).__name__ == "RandomSelection"

    try:
        reducer_from_config(PostProcConfig(dr_method="geopres", target_dim=2, dr_weights_id="0" * 16), 8)
    except ValueError as error:
        assert "geopres_weights" in str(error)
    else:
        raise AssertionError("GeoPres configuration without weights must fail")


def test_apply_reduction_normalizes_both_sides():
    values = np.array([[3, 4, 0], [0, 0, 2]], dtype=np.float32)

    result = apply_reduction(values, Identity())

    np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1.0)
    np.testing.assert_allclose(result, [[0.6, 0.8, 0], [0, 0, 1]])


def test_post_processing_pipeline_fits_on_normalized_values():
    values = np.array([[3, 4, 0], [0, 0, 2], [5, 0, 0]], dtype=np.float32)
    pipeline = PostProcessingPipeline(
        PostProcConfig(dr_method="truncate", target_dim=2), source_dim=3
    )
    pipeline.fit(values)

    result = pipeline.transform(values)

    np.testing.assert_allclose(
        result,
        [[0.6, 0.8], [0, 0], [1, 0]],
    )


def test_post_processing_pipeline_requires_fit():
    pipeline = PostProcessingPipeline(
        PostProcConfig(dr_method="pca", target_dim=2), source_dim=3
    )

    try:
        pipeline.transform(np.ones((2, 3), dtype=np.float32))
    except RuntimeError as error:
        assert "fitted" in str(error)
    else:
        raise AssertionError("transform before fit must fail")

def test_geopres_weights_must_match_the_configured_weights_id():
    weights = np.arange(16, dtype=np.float32).reshape(2, 8)
    config = PostProcConfig(dr_method="geopres", target_dim=2, dr_weights_id=weights_id(weights))
    assert type(reducer_from_config(config, 8, geopres_weights=weights)) is GeoPres

    with pytest.raises(ValueError, match="dr_weights_id"):
        reducer_from_config(config, 8, geopres_weights=weights + 1)
