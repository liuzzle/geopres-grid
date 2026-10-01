import numpy as np

from geopres_grid.identity import PostProcConfig
from geopres_grid.quantizers import Binary
from geopres_grid.quantizers import CalibratedCompressionWrapper
from geopres_grid.quantizers import EqualCount
from geopres_grid.quantizers import FP16
from geopres_grid.quantizers import UniformAffine
from geopres_grid.quantizers import calibration_symmetry_ablation
from geopres_grid.quantizers import fake_quantize
from geopres_grid.quantizers import fit_quantizer
from geopres_grid.quantizers import quantizer_from_config


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


def test_equal_count_uses_per_dimension_calibration():
    values = np.arange(40, dtype=np.float32).reshape(20, 2)
    quantizer = EqualCount(4)
    quantizer.fit(values)
    quantized = quantizer.quantize(values)
    restored = quantizer.dequantize(quantized)

    assert quantized.shape == values.shape
    assert restored.shape == values.shape
    assert np.all(np.isfinite(restored))
    assert quantizer.bytes_per_vector(2) == 1


def test_binary_fp16_and_config_factory():
    values = np.array([[-1.0, 0.0], [1.0, 2.0]], dtype=np.float32)
    binary = Binary()
    binary.fit(values)
    np.testing.assert_array_equal(binary.dequantize(binary.quantize(values)), [[-1, 1], [1, 1]])

    fp16 = FP16()
    fp16.fit(values)
    assert fp16.dequantize(fp16.quantize(values)).dtype == np.float32

    assert type(quantizer_from_config(PostProcConfig(quant_method="int8"))) is UniformAffine


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


def test_calibration_symmetry_ablation_reports_three_paths():
    queries = np.array([[-1.0, -0.5], [1.0, 0.5]], dtype=np.float32)
    documents = np.array([[-0.5, -1.0], [0.5, 1.0]], dtype=np.float32)

    scores = calibration_symmetry_ablation(
        PostProcConfig(quant_method="int4"), queries, documents
    )

    assert set(scores) == {
        "shared_calibration",
        "per_side_dequantized",
        "per_side_quantized",
    }
    assert all(np.isfinite(score) for score in scores.values())