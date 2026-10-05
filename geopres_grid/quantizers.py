"""Fitted fake-quantization methods for post-processed embeddings.

The bit-width grid follows Kisako, Tsukagoshi & Sasano, *When Is 0.1% Enough?*
(arXiv:2606.01074), §3.4: b = 32 is the original fp32 vector, b = 16 a float16
round trip, b = 1 sign quantization, and b in {2, 4, 8} a global equal-count
lookup table. Uniform affine is the comparison point proposed on 16.09 (§5).

Which quantizers share one grid across coordinates matters for the DR side:
`EqualCount` and `Binary` do, `UniformAffine` fits one range per dimension. The
PCA-vs-PCA+ROR argument (Kisako et al. §3.3, meeting 16.09 §3) is an argument
about a shared grid, so its effect is expected under the first two.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from mteb.models import CompressionWrapper
from scipy.stats import rankdata

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


class Float32(Quantizer):
    """No quantization: the fp32 reference (Kisako et al. §3.4, b = 32)."""

    bits_per_dim = 32

    def fit(self, values: np.ndarray) -> None:
        del values

    def quantize(self, values: np.ndarray) -> np.ndarray:
        return np.array(values, dtype=np.float32)

    def dequantize(self, quantized: np.ndarray) -> np.ndarray:
        return np.asarray(quantized, dtype=np.float32)

    def save(self, path: str | Path) -> None:
        np.savez(path)

    @classmethod
    def load(cls, path: str | Path) -> "Float32":
        np.load(path)
        return cls()


class UniformAffine(Quantizer):
    """Uniform affine quantizer with one min/max range per dimension.

    The per-dimension range is the scheme of MTEB's `CompressionWrapper`
    (`_get_min_max_per_dim`), fitted once on calibration data rather than inside
    every `encode` call. Because each coordinate gets its own range, this
    quantizer already absorbs coordinate-wise variance imbalance -- unlike the
    shared grids of `EqualCount` and `Binary`.
    """

    def __init__(self, bits: int) -> None:
        if bits not in (2, 4, 8):
            raise ValueError("UniformAffine supports 2, 4 or 8 bits")
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
    """Sign quantization: +1 for non-negative values, -1 otherwise.

    Kisako et al. §3.4, b = 1. One threshold for every coordinate, so a shared
    grid in the sense of §3.3.
    """

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
    """float16 round trip (Kisako et al. §3.4, b = 16)."""

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


def _equal_count_bounds(size: int, count: int) -> np.ndarray:
    """Chunk boundaries splitting `size` sorted values into `count` near-equal bins.

    The partition of `np.array_split`: the first `size % count` bins hold one
    extra value, so no bin is ever empty.
    """
    base, extra = divmod(size, count)
    sizes = np.full(count, base, dtype=np.int64)
    sizes[:extra] += 1
    return np.concatenate(([0], np.cumsum(sizes)))


class EqualCount(Quantizer):
    """Global equal-count lookup-table quantizer (Kisako et al. §3.4).

    All calibration scalars are flattened into one pool, sorted, and split into
    2**b bins holding equal numbers of values; each bin is represented by the
    mean of its members. A scalar is replaced by the representative of the bin
    it falls into, with bin edges midway between neighbouring bins. One table
    serves every coordinate, which is what makes the variance imbalance of bare
    PCA costly and the random rotation of PCA+ROR worthwhile (§3.3).
    """

    def __init__(self, bits: int) -> None:
        if bits not in (2, 4, 8):
            raise ValueError("EqualCount supports 2, 4 or 8 bits")
        self.bits_per_dim = bits
        self.edges: np.ndarray | None = None
        self.levels: np.ndarray | None = None

    @property
    def _levels_count(self) -> int:
        return 1 << self.bits_per_dim

    def fit(self, values: np.ndarray) -> None:
        flat = np.sort(np.asarray(values, dtype=np.float32).ravel())
        count = self._levels_count
        if flat.size < count:
            raise ValueError(
                f"EqualCount({self.bits_per_dim}) needs at least {count} calibration "
                f"values, got {flat.size}"
            )
        bounds = _equal_count_bounds(flat.size, count)
        sums = np.concatenate(([0.0], np.cumsum(flat, dtype=np.float64)))
        self.levels = ((sums[bounds[1:]] - sums[bounds[:-1]]) / np.diff(bounds)).astype(
            np.float32
        )
        inner = bounds[1:-1]
        self.edges = ((flat[inner - 1].astype(np.float64) + flat[inner]) / 2).astype(
            np.float32
        )

    def quantize(self, values: np.ndarray) -> np.ndarray:
        if self.edges is None:
            raise RuntimeError("quantizer must be fitted before quantize")
        values = np.asarray(values, dtype=np.float32)
        return np.searchsorted(self.edges, values, side="right").astype(np.uint8)

    def dequantize(self, quantized: np.ndarray) -> np.ndarray:
        if self.levels is None:
            raise RuntimeError("quantizer must be fitted before dequantize")
        return self.levels[np.asarray(quantized, dtype=np.intp)]

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
        "none": Float32,
        "fp16": FP16,
        "int8": lambda: UniformAffine(8),
        "uint8": lambda: UniformAffine(8),
        "int4": lambda: UniformAffine(4),
        "uint4": lambda: UniformAffine(4),
        "int2": lambda: UniformAffine(2),
        "uint2": lambda: UniformAffine(2),
        "binary": Binary,
        "equal_count_8": lambda: EqualCount(8),
        "equal_count_4": lambda: EqualCount(4),
        "equal_count_2": lambda: EqualCount(2),
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


def _cosine_matrix(queries: np.ndarray, documents: np.ndarray) -> np.ndarray:
    eps = np.finfo(np.float64).eps
    queries = np.asarray(queries, dtype=np.float64)
    documents = np.asarray(documents, dtype=np.float64)
    queries = queries / np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), eps)
    documents = documents / np.maximum(
        np.linalg.norm(documents, axis=1, keepdims=True), eps
    )
    return queries @ documents.T


def _rank_fidelity(reference: np.ndarray, scores: np.ndarray, k: int) -> dict[str, float]:
    """How well `scores` preserve each query's fp32 ranking.

    Row-wise Spearman correlation, averaged over queries -- the "local" variant of
    GeoPres's intrinsic Spearman metric (`losses._spearman_local`), here with
    exact ranks -- and recall@k of the fp32 top-k documents, the recall of
    ANN-Benchmarks (Aumüller et al., arXiv:1807.05614).
    """
    reference_ranks = rankdata(reference, axis=1)
    ranks = rankdata(scores, axis=1)
    reference_ranks -= reference_ranks.mean(axis=1, keepdims=True)
    ranks -= ranks.mean(axis=1, keepdims=True)
    covariance = np.sum(reference_ranks * ranks, axis=1)
    scale = np.sqrt(np.sum(reference_ranks**2, axis=1) * np.sum(ranks**2, axis=1))
    spearman = np.divide(covariance, scale, out=np.zeros_like(covariance), where=scale > 0)

    k = min(k, reference.shape[1])
    true_top = np.argpartition(-reference, k - 1, axis=1)[:, :k]
    found_top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    recall = [len(np.intersect1d(a, b)) / k for a, b in zip(true_top, found_top)]
    return {"spearman": float(np.mean(spearman)), f"recall@{k}": float(np.mean(recall))}


def calibration_symmetry_ablation(
    config: PostProcConfig,
    queries: np.ndarray,
    documents: np.ndarray,
    *,
    k: int = 10,
) -> dict[str, dict[str, float]]:
    """Measure what each calibration choice costs against the fp32 scores (WP-E.5).

    Three variants: one table fitted on both sides (`shared_calibration`, the
    default and Kisako et al. §3.5); one table per side, dequantized
    (`per_side_dequantized`); one table per side scored on the raw codes
    (`per_side_quantized`), which is what MTEB's `CompressionWrapper` returns.
    Each is scored by `_rank_fidelity` against fp32 cosine scores of the same
    query x document pairs, so the numbers are comparable across variants and
    the sides need not have the same number of rows.
    """
    queries = np.asarray(queries, dtype=np.float32)
    documents = np.asarray(documents, dtype=np.float32)
    reference = _cosine_matrix(queries, documents)

    shared = fit_quantizer(config, np.vstack([queries, documents]))
    query_quantizer = fit_quantizer(config, queries)
    document_quantizer = fit_quantizer(config, documents)
    side_query = query_quantizer.quantize(queries)
    side_document = document_quantizer.quantize(documents)

    variants = {
        "shared_calibration": (
            fake_quantize(queries, shared),
            fake_quantize(documents, shared),
        ),
        "per_side_dequantized": (
            query_quantizer.dequantize(side_query),
            document_quantizer.dequantize(side_document),
        ),
        "per_side_quantized": (side_query, side_document),
    }
    return {
        name: _rank_fidelity(reference, _cosine_matrix(query, document), k)
        for name, (query, document) in variants.items()
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
