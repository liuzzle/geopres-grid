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


def cone(rows=300, dim=16, seed=0):
    """Unit vectors in a narrow cone, like the backbones' (||mean|| well above 0)."""
    generator = np.random.default_rng(seed)
    values = generator.normal(size=(rows, dim)) * 0.4 + 1.0
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def cosine(a, b):
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return a @ b.T


@pytest.mark.parametrize("method", ["pca", "pca_ror"])
def test_restored_pca_scores_in_the_uncentred_projection(method):
    values = cone()
    pipeline = PostProcessingPipeline(PostProcConfig(dr_method=method, target_dim=6), 16)
    pipeline.fit(values)

    scored = pipeline.finish(pipeline.transform(values))

    uncentred = values @ pipeline.reducer.components_.T
    np.testing.assert_allclose(cosine(scored, scored), cosine(uncentred, uncentred), atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(scored, axis=1), 1.0, atol=1e-6)
    centred = PostProcessingPipeline(PostProcConfig(dr_method=method, target_dim=6, pca_mean="subtract"), 16)
    centred.fit(values)
    old = centred.transform(values)
    assert np.abs(cosine(old, old) - cosine(uncentred, uncentred)).max() > 0.1  # what §23.2 found


def test_restored_pca_and_pca_ror_agree_at_fp32():
    values = cone()
    scores = []
    for method in ("pca", "pca_ror"):
        pipeline = PostProcessingPipeline(PostProcConfig(dr_method=method, target_dim=6), 16)
        pipeline.fit(values)
        out = pipeline.finish(pipeline.transform(values))
        scores.append(out @ out.T)
    np.testing.assert_allclose(scores[0], scores[1], atol=1e-5)


@pytest.mark.parametrize("method", ["pca", "pca_ror"])
def test_restored_pca_hands_the_quantizer_centred_values(method):
    values = cone()
    pipeline = PostProcessingPipeline(PostProcConfig(dr_method=method, target_dim=6), 16)
    pipeline.fit(values)

    quantizer_input = pipeline.transform(values)

    np.testing.assert_allclose(quantizer_input.mean(axis=0), 0.0, atol=1e-5)
    np.testing.assert_allclose(
        quantizer_input + pipeline.reducer.projected_mean(),
        pipeline.reducer.transform(values) + pipeline.reducer.projected_mean(),
        atol=1e-6,
    )
    assert not np.allclose(np.linalg.norm(quantizer_input, axis=1), 1.0)  # not renormalised


def test_restore_puts_the_mean_on_the_reconstruction_scale():
    values = cone()
    pipeline = PostProcessingPipeline(PostProcConfig(dr_method="pca", target_dim=6), 16)
    pipeline.fit(values)
    codes = np.sign(pipeline.transform(values))

    restored = pipeline.finish(codes, 0.25)

    expected = 0.25 * codes + pipeline.reducer.projected_mean()
    np.testing.assert_allclose(restored, expected / np.linalg.norm(expected, axis=1, keepdims=True), atol=1e-6)


@pytest.mark.parametrize(
    "config",
    [
        PostProcConfig(dr_method="pca", target_dim=6, pca_mean="subtract"),
        PostProcConfig(dr_method="truncate", target_dim=6),
        PostProcConfig(dr_method="random_projection", target_dim=6),
        PostProcConfig(),
    ],
)
def test_every_other_cell_is_unchanged(config):
    """`transform` is normalize -> reduce -> normalize and `finish` returns its input."""
    values = cone()
    pipeline = PostProcessingPipeline(config, 16)
    pipeline.fit(values)
    dequantized = pipeline.transform(values)

    assert pipeline.finish(dequantized, 0.25) is dequantized
    np.testing.assert_array_equal(dequantized, apply_reduction(values, pipeline.reducer))
