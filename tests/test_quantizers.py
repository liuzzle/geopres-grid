import numpy as np
import pytest

from geopres_grid.identity import BITS_PER_DIM
from geopres_grid.identity import PostProcConfig
from geopres_grid.quantizers import Binary
from geopres_grid.quantizers import CalibratedCompressionWrapper
from geopres_grid.quantizers import EqualCount
from geopres_grid.quantizers import FP16
from geopres_grid.quantizers import Float32
from geopres_grid.quantizers import UniformAffine
from geopres_grid.quantizers import calibration_symmetry_ablation
from geopres_grid.quantizers import fake_quantize
from geopres_grid.quantizers import fit_quantizer
from geopres_grid.quantizers import quantizer_from_config


def test_every_quant_method_builds_a_quantizer_of_the_tabled_width():
    """The config table and the quantizer must agree on what a method stores."""
    for method, bits in BITS_PER_DIM.items():
        quantizer = quantizer_from_config(PostProcConfig(quant_method=method))
        assert quantizer.bits_per_dim == bits, method


def test_none_is_the_fp32_original():
    values = np.array([[0.1234567, -1e-8], [3.0, 1e6]], dtype=np.float32)
    quantizer = quantizer_from_config(PostProcConfig())

    assert type(quantizer) is Float32
    np.testing.assert_array_equal(fake_quantize(values, quantizer), values)


def test_uniform_affine_round_trip_and_reload(tmp_path):
    values = np.array([[-1.0, 0.0], [0.0, 2.0], [1.0, 4.0]], dtype=np.float32)
    quantizer = UniformAffine(8)
    quantizer.fit(values)
    quantized = quantizer.quantize(values)

    assert quantized.dtype == np.uint8
    np.testing.assert_allclose(quantizer.dequantize(quantized), values, atol=0.01)

    path = tmp_path / "int8.npz"
    quantizer.save(path)
    np.testing.assert_allclose(
        UniformAffine.load(path).dequantize(quantized), values, atol=0.01
    )


def test_uniform_affine_two_bit_uses_four_levels_per_dimension():
    values = np.linspace(-1, 1, 40, dtype=np.float32).reshape(20, 2)
    quantizer = UniformAffine(2)
    quantizer.fit(values)

    codes = quantizer.quantize(values)

    assert set(np.unique(codes)) == {0, 1, 2, 3}
    assert quantizer.bytes_per_vector(768) == 192


def test_equal_count_levels_are_bin_means_of_the_pooled_values():
    # 16 distinct scalars, 2 bits -> 4 bins of 4 values each (Kisako et al. §3.4).
    values = np.arange(16, dtype=np.float32).reshape(8, 2)
    quantizer = EqualCount(2)
    quantizer.fit(values)

    np.testing.assert_allclose(quantizer.levels, [1.5, 5.5, 9.5, 13.5])
    np.testing.assert_allclose(quantizer.edges, [3.5, 7.5, 11.5])
    counts = np.bincount(quantizer.quantize(values).ravel(), minlength=4)
    np.testing.assert_array_equal(counts, [4, 4, 4, 4])


def test_equal_count_is_one_table_for_every_coordinate():
    rng = np.random.default_rng(0)
    # Second coordinate has 100x the spread of the first, as after bare PCA.
    values = rng.normal(size=(500, 2)).astype(np.float32) * np.array([0.01, 1.0], dtype=np.float32)
    quantizer = EqualCount(4)
    quantizer.fit(values)

    assert quantizer.levels.shape == (16,)
    assert quantizer.edges.shape == (15,)
    probe = np.array([[0.3, 0.3]], dtype=np.float32)
    codes = quantizer.quantize(probe)
    assert codes[0, 0] == codes[0, 1]
    # The narrow coordinate collapses onto the few central bins of the shared table.
    assert len(np.unique(quantizer.quantize(values)[:, 0])) < 16


def test_equal_count_reloads_and_rejects_too_little_calibration(tmp_path):
    values = np.arange(32, dtype=np.float32).reshape(16, 2)
    quantizer = EqualCount(4)
    quantizer.fit(values)
    path = tmp_path / "eq.npz"
    quantizer.save(path)

    reloaded = EqualCount.load(path)
    np.testing.assert_array_equal(reloaded.quantize(values), quantizer.quantize(values))
    with pytest.raises(ValueError, match="at least 256"):
        EqualCount(8).fit(values)


def test_binary_fp16_and_config_factory():
    values = np.array([[-1.0, 0.0], [1.0, 2.0]], dtype=np.float32)
    binary = Binary()
    binary.fit(values)
    np.testing.assert_array_equal(binary.dequantize(binary.quantize(values)), [[-1, 1], [1, 1]])

    fp16 = FP16()
    fp16.fit(values)
    assert fp16.dequantize(fp16.quantize(values)).dtype == np.float32

    assert type(quantizer_from_config(PostProcConfig(quant_method="int8"))) is UniformAffine
    assert type(quantizer_from_config(PostProcConfig(quant_method="equal_count_2"))) is EqualCount


def test_fit_quantizer_and_fake_path_reuse_shared_calibration():
    calibration = np.array([[-1.0, -1.0], [1.0, 1.0]], dtype=np.float32)
    quantizer = fit_quantizer(PostProcConfig(quant_method="int8"), calibration)
    query = np.array([[0.25, -0.25]], dtype=np.float32)
    document = np.array([[-0.25, 0.25]], dtype=np.float32)

    query_restored = fake_quantize(query, quantizer)
    document_restored = fake_quantize(document, quantizer)

    assert np.allclose(query_restored, query, atol=0.01)
    assert np.allclose(document_restored, document, atol=0.01)


class FakeModel:
    mteb_model_meta = None

    def encode(self, inputs, **kwargs):
        return np.asarray(inputs, dtype=np.float32)


def test_calibrated_wrapper_does_not_refit_per_encode():
    quantizer = UniformAffine(8)
    quantizer.fit(np.array([[-1.0, -1.0], [1.0, 1.0]], dtype=np.float32))
    wrapper = CalibratedCompressionWrapper(FakeModel(), quantizer)

    result = wrapper.encode(np.array([[0.5, -0.5]], dtype=np.float32))

    np.testing.assert_allclose(result, [[0.5, -0.5]], atol=0.01)
    np.testing.assert_array_equal(quantizer.minimum, [-1.0, -1.0])


def test_calibration_symmetry_ablation_scores_against_fp32():
    rng = np.random.default_rng(0)
    queries = rng.normal(size=(20, 16)).astype(np.float32)
    documents = rng.normal(size=(200, 16)).astype(np.float32)

    scores = calibration_symmetry_ablation(
        PostProcConfig(quant_method="int8"), queries, documents, k=10
    )

    assert set(scores) == {"shared_calibration", "per_side_dequantized", "per_side_quantized"}
    for values in scores.values():
        assert -1.0 <= values["spearman"] <= 1.0
        assert 0.0 <= values["recall@10"] <= 1.0
    # 8 bits with a shared table barely moves a ranking.
    assert scores["shared_calibration"]["spearman"] > 0.99
    assert scores["shared_calibration"]["recall@10"] > 0.9


def test_symmetry_ablation_is_exact_without_quantization():
    rng = np.random.default_rng(1)
    queries = rng.normal(size=(5, 8)).astype(np.float32)
    documents = rng.normal(size=(30, 8)).astype(np.float32)

    scores = calibration_symmetry_ablation(PostProcConfig(), queries, documents, k=5)

    for values in scores.values():
        assert values["spearman"] == pytest.approx(1.0)
        assert values["recall@5"] == pytest.approx(1.0)
