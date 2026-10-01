"""Fitted fake-quantization methods for post-processed embeddings."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from mteb.models import CompressionWrapper

from geopres_grid.identity import PostProcConfig


class Quantizer(ABC):
    bits_per_dim: int

    @abstractmethod
    def fit(self, values: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def quantize(self, values: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def dequantize(self, quantized: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str | Path) -> None:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "Quantizer":
        raise NotImplementedError

    def bytes_per_vector(self, dimensions: int) -> int:
        return (dimensions * self.bits_per_dim + 7) // 8


class UniformAffine(Quantizer):
    def __init__(self, bits: int) -> None:
        if bits not in (4, 8):
            raise ValueError("UniformAffine supports 4 or 8 bits")
        self.bits_per_dim = bits
        self.minimum: np.ndarray | None = None
        self.scale: np.ndarray | None = None

    @property
    def _levels(self) -> int:
        return 1 << self.bits_per_dim

    def fit(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float32)
        self.minimum = values.min(axis=0)
        maximum = values.max(axis=0)
        self.scale = (maximum - self.minimum) / (self._levels - 1)
        self.scale = np.maximum(self.scale, np.finfo(np.float32).eps)

    def quantize(self, values: np.ndarray) -> np.ndarray:
        if self.minimum is None or self.scale is None:
            raise RuntimeError("quantizer must be fitted before quantize")
        codes = np.rint((np.asarray(values, dtype=np.float32) - self.minimum) / self.scale)
        return np.clip(codes, 0, self._levels - 1).astype(np.uint8)

    def dequantize(self, quantized: np.ndarray) -> np.ndarray:
        if self.minimum is None or self.scale is None:
            raise RuntimeError("quantizer must be fitted before dequantize")
        return np.asarray(quantized, dtype=np.float32) * self.scale + self.minimum

    def save(self, path: str | Path) -> None:
        if self.minimum is None or self.scale is None:
            raise RuntimeError("quantizer must be fitted before save")
        np.savez(path, bits=self.bits_per_dim, minimum=self.minimum, scale=self.scale)

    @classmethod
    def load(cls, path: str | Path) -> "UniformAffine":
        state = np.load(path)
        quantizer = cls(int(state["bits"]))
        quantizer.minimum = state["minimum"]
        quantizer.scale = state["scale"]
        return quantizer


class Binary(Quantizer):
    bits_per_dim = 1

    def fit(self, values: np.ndarray) -> None:
        del values

    def quantize(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values) >= 0).astype(np.int8)

    def dequantize(self, quantized: np.ndarray) -> np.ndarray:
        return np.where(np.asarray(quantized) != 0, 1.0, -1.0).astype(np.float32)

    def save(self, path: str | Path) -> None:
        np.savez(path)

    @classmethod
    def load(cls, path: str | Path) -> "Binary":
        np.load(path)
        return cls()


class FP16(Quantizer):
    bits_per_dim = 16

    def fit(self, values: np.ndarray) -> None:
        del values

    def quantize(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float16)

    def dequantize(self, quantized: np.ndarray) -> np.ndarray:
        return np.asarray(quantized, dtype=np.float32)

    def save(self, path: str | Path) -> None:
        np.savez(path)

    @classmethod
    def load(cls, path: str | Path) -> "FP16":
        np.load(path)
        return cls()


class EqualCount(Quantizer):
    def __init__(self, bits: int) -> None:
        if bits not in (4, 8):
            raise ValueError("EqualCount supports 4 or 8 bits")
        self.bits_per_dim = bits
        self.edges: np.ndarray | None = None
        self.levels: np.ndarray | None = None

    @property
    def _levels_count(self) -> int:
        return 1 << self.bits_per_dim

    def fit(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float32)
        quantiles = np.linspace(0, 1, self._levels_count + 1)[1:-1]
        self.edges = np.quantile(values, quantiles, axis=0).T.astype(np.float32)
        boundaries = np.concatenate(
            [values.min(axis=0)[:, None], self.edges, values.max(axis=0)[:, None]],
            axis=1,
        )
        self.levels = ((boundaries[:, :-1] + boundaries[:, 1:]) / 2).astype(np.float32)

    def quantize(self, values: np.ndarray) -> np.ndarray:
        if self.edges is None:
            raise RuntimeError("quantizer must be fitted before quantize")
        return np.sum(np.asarray(values)[..., None] > self.edges[None, :, :], axis=-1).astype(np.uint8)

    def dequantize(self, quantized: np.ndarray) -> np.ndarray:
        if self.levels is None:
            raise RuntimeError("quantizer must be fitted before dequantize")
        return np.take_along_axis(
            self.levels[None, :, :],
            np.asarray(quantized, dtype=np.intp)[..., None],
            axis=2,
        )[..., 0].astype(np.float32)

    def save(self, path: str | Path) -> None:
        if self.edges is None or self.levels is None:
            raise RuntimeError("quantizer must be fitted before save")
        np.savez(path, bits=self.bits_per_dim, edges=self.edges, levels=self.levels)

    @classmethod
    def load(cls, path: str | Path) -> "EqualCount":
        state = np.load(path)
        quantizer = cls(int(state["bits"]))
        quantizer.edges = state["edges"]
        quantizer.levels = state["levels"]
        return quantizer


def quantizer_from_config(config: PostProcConfig) -> Quantizer:
    """Construct the configured quantizer."""
    methods = {
        "none": FP16,
        "fp16": FP16,
        "int8": lambda: UniformAffine(8),
        "uint8": lambda: UniformAffine(8),
        "int4": lambda: UniformAffine(4),
        "uint4": lambda: UniformAffine(4),
        "binary": Binary,
        "equal_count_8": lambda: EqualCount(8),
        "equal_count_4": lambda: EqualCount(4),
    }
    try:
        return methods[config.quant_method]()
    except KeyError as error:
        raise ValueError(f"Unsupported quant_method: {config.quant_method}") from error


def fit_quantizer(config: PostProcConfig, calibration: np.ndarray) -> Quantizer:
    """Fit one calibration table for all vectors that share a post-processing run."""
    quantizer = quantizer_from_config(config)
    quantizer.fit(calibration)
    return quantizer


def fake_quantize(values: np.ndarray, quantizer: Quantizer) -> np.ndarray:
    """Quantize and immediately restore fp32 values for cosine evaluation."""
    return quantizer.dequantize(quantizer.quantize(values))


def _pairwise_cosine(values1: np.ndarray, values2: np.ndarray) -> float:
    values1 = np.asarray(values1, dtype=np.float32)
    values2 = np.asarray(values2, dtype=np.float32)
    norms1 = np.linalg.norm(values1, axis=1)
    norms2 = np.linalg.norm(values2, axis=1)
    scores = np.sum(values1 * values2, axis=1) / np.maximum(norms1 * norms2, np.finfo(np.float32).eps)
    return float(np.mean(scores))


def calibration_symmetry_ablation(
    config: PostProcConfig,
    queries: np.ndarray,
    documents: np.ndarray,
) -> dict[str, float]:
    """Compare shared and per-side calibration under both score conventions."""
    queries = np.asarray(queries, dtype=np.float32)
    documents = np.asarray(documents, dtype=np.float32)
    if len(queries) != len(documents):
        raise ValueError("queries and documents must have the same number of rows")

    shared = fit_quantizer(config, np.vstack([queries, documents]))
    query_quantizer = fit_quantizer(config, queries)
    document_quantizer = fit_quantizer(config, documents)

    shared_query = shared.quantize(queries)
    shared_document = shared.quantize(documents)
    side_query = query_quantizer.quantize(queries)
    side_document = document_quantizer.quantize(documents)
    return {
        "shared_calibration": _pairwise_cosine(
            shared.dequantize(shared_query), shared.dequantize(shared_document)
        ),
        "per_side_dequantized": _pairwise_cosine(
            query_quantizer.dequantize(side_query),
            document_quantizer.dequantize(side_document),
        ),
        "per_side_quantized": _pairwise_cosine(side_query, side_document),
    }


class CalibratedCompressionWrapper(CompressionWrapper):
    """MTEB-compatible wrapper using a pre-fitted project quantizer."""

    def __init__(self, model: object, quantizer: Quantizer) -> None:
        self.model = model
        self.quantizer = quantizer

    def encode(self, inputs: object, **kwargs: object) -> np.ndarray:
        embeddings = self.model.encode(inputs, **kwargs)
        if hasattr(embeddings, "detach"):
            embeddings = embeddings.detach().cpu().numpy()
        return fake_quantize(np.asarray(embeddings, dtype=np.float32), self.quantizer)