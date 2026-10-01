import numpy as np

from geopres_grid.identity import PostProcConfig
from geopres_grid.quantizers import Binary
from geopres_grid.quantizers import EqualCount
from geopres_grid.quantizers import FP16
from geopres_grid.quantizers import UniformAffine
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