"""Dimensionality reducers for cached embedding arrays."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA as SklearnPCA

from geopres_grid.identity import PostProcConfig, weights_id


def _unit_norm(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, np.finfo(np.float32).eps)


class Reducer(ABC):
    """Fit/transform contract for post-cache dimensionality reduction."""

    overhead_class: str

    @abstractmethod
    def fit(self, values: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def transform(self, values: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str | Path) -> None:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "Reducer":
        raise NotImplementedError


class Identity(Reducer):
    overhead_class = "slice"

    def fit(self, values: np.ndarray) -> None:
        del values

    def transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float32).copy()

    def save(self, path: str | Path) -> None:
        np.save(path, np.array([], dtype=np.float32))

    @classmethod
    def load(cls, path: str | Path) -> "Identity":
        np.load(path)
        return cls()


class Truncate(Reducer):
    overhead_class = "slice"

    def __init__(self, target_dim: int) -> None:
        if target_dim <= 0:
            raise ValueError("target_dim must be positive")
        self.target_dim = target_dim

    def fit(self, values: np.ndarray) -> None:
        if values.shape[1] < self.target_dim:
            raise ValueError("target_dim exceeds input dimension")

    def transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float32)[:, : self.target_dim]

    def save(self, path: str | Path) -> None:
        np.save(path, np.array([self.target_dim], dtype=np.int64))

    @classmethod
    def load(cls, path: str | Path) -> "Truncate":
        return cls(int(np.load(path)[0]))


class PCA(Reducer):
    overhead_class = "matmul+mean-shift"

    def __init__(self, target_dim: int, seed: int = 42) -> None:
        self.target_dim = target_dim
        self.seed = seed
        self.components_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> None:
        model = SklearnPCA(
            n_components=self.target_dim,
            svd_solver="auto",
            random_state=self.seed,
        )
        model.fit(np.asarray(values, dtype=np.float32))
        self.components_ = model.components_.astype(np.float32)
        self.mean_ = model.mean_.astype(np.float32)

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.components_ is None or self.mean_ is None:
            raise RuntimeError("PCA must be fitted before transform")
        return (np.asarray(values, dtype=np.float32) - self.mean_) @ self.components_.T

    def save(self, path: str | Path) -> None:
        if self.components_ is None or self.mean_ is None:
            raise RuntimeError("PCA must be fitted before save")
        np.savez(path, components=self.components_, mean=self.mean_, seed=self.seed)

    @classmethod
    def load(cls, path: str | Path) -> "PCA":
        state = np.load(path)
        reducer = cls(int(state["components"].shape[0]), int(state["seed"]))
        reducer.components_ = state["components"]
        reducer.mean_ = state["mean"]
        return reducer


class PCARotation(PCA):
    """PCA followed by a seeded orthogonal rotation."""

    def __init__(self, target_dim: int, seed: int = 42) -> None:
        super().__init__(target_dim, seed)
        self.rotation_: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> None:
        super().fit(values)
        generator = np.random.default_rng(self.seed)
        raw = generator.normal(size=(self.target_dim, self.target_dim))
        self.rotation_ = np.linalg.qr(raw)[0].astype(np.float32)

    def transform(self, values: np.ndarray) -> np.ndarray:
        projected = super().transform(values)
        if self.rotation_ is None:
            raise RuntimeError("PCARotation must be fitted before transform")
        return projected @ self.rotation_.T

    def save(self, path: str | Path) -> None:
        if self.components_ is None or self.mean_ is None or self.rotation_ is None:
            raise RuntimeError("PCARotation must be fitted before save")
        np.savez(
            path,
            components=self.components_,
            mean=self.mean_,
            rotation=self.rotation_,
            seed=self.seed,
        )

    @classmethod
    def load(cls, path: str | Path) -> "PCARotation":
        state = np.load(path)
        reducer = cls(int(state["components"].shape[0]), int(state["seed"]))
        reducer.components_ = state["components"]
        reducer.mean_ = state["mean"]
        reducer.rotation_ = state["rotation"]
        return reducer


class GeoPres(Reducer):
    overhead_class = "matmul"

    def __init__(self, weights: np.ndarray | None = None, path: str | Path | None = None) -> None:
        if weights is None and path is None:
            raise ValueError("weights or path is required")
        self.weights = np.asarray(weights, dtype=np.float32) if weights is not None else np.asarray(np.load(path), dtype=np.float32)

    @property
    def target_dim(self) -> int:
        return int(self.weights.shape[0])

    def fit(self, values: np.ndarray) -> None:
        if values.shape[1] != self.weights.shape[1]:
            raise ValueError("weight input dimension does not match values")

    def transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float32) @ self.weights.T

    def save(self, path: str | Path) -> None:
        np.save(path, self.weights)

    @classmethod
    def load(cls, path: str | Path) -> "GeoPres":
        return cls(path=path)


class RandomProjection(Reducer):
    overhead_class = "matmul"

    def __init__(self, target_dim: int, seed: int = 42) -> None:
        self.target_dim = target_dim
        self.seed = seed
        self.weights: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> None:
        generator = np.random.default_rng(self.seed)
        self.weights = generator.normal(
            0.0, 1.0 / np.sqrt(self.target_dim), (self.target_dim, values.shape[1])
        ).astype(np.float32)

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("RandomProjection must be fitted before transform")
        return np.asarray(values, dtype=np.float32) @ self.weights.T

    def save(self, path: str | Path) -> None:
        if self.weights is None:
            raise RuntimeError("RandomProjection must be fitted before save")
        np.savez(path, weights=self.weights, seed=self.seed)

    @classmethod
    def load(cls, path: str | Path) -> "RandomProjection":
        state = np.load(path)
        reducer = cls(int(state["weights"].shape[0]), int(state["seed"]))
        reducer.weights = state["weights"]
        return reducer


class RandomSelection(Reducer):
    overhead_class = "slice"

    def __init__(self, target_dim: int, seed: int = 42) -> None:
        self.target_dim = target_dim
        self.seed = seed
        self.indices: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> None:
        if self.target_dim > values.shape[1]:
            raise ValueError("target_dim exceeds input dimension")
        self.indices = np.sort(
            np.random.default_rng(self.seed).choice(values.shape[1], self.target_dim, replace=False)
        )

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.indices is None:
            raise RuntimeError("RandomSelection must be fitted before transform")
        return np.asarray(values, dtype=np.float32)[:, self.indices]

    def save(self, path: str | Path) -> None:
        if self.indices is None:
            raise RuntimeError("RandomSelection must be fitted before save")
        np.savez(path, indices=self.indices, seed=self.seed)

    @classmethod
    def load(cls, path: str | Path) -> "RandomSelection":
        state = np.load(path)
        reducer = cls(int(state["indices"].shape[0]), int(state["seed"]))
        reducer.indices = state["indices"]
        return reducer


def apply_reduction(
    values: np.ndarray,
    reducer: Reducer,
    *,
    normalize_before: bool = True,
    normalize_after: bool = True,
) -> np.ndarray:
    """Apply the canonical normalize -> reduce -> normalize pipeline."""
    prepared = _unit_norm(values) if normalize_before else np.asarray(values, dtype=np.float32)
    reduced = reducer.transform(prepared)
    return _unit_norm(reduced) if normalize_after else reduced


class PostProcessingPipeline:
    """Fit and apply one configured reducer to cached embedding arrays."""

    def __init__(
        self,
        config: PostProcConfig,
        source_dim: int,
        *,
        geopres_weights: np.ndarray | None = None,
    ) -> None:
        self.config = config
        self.reducer = reducer_from_config(
            config, source_dim, geopres_weights=geopres_weights
        )
        self._fitted = False

    def fit(self, values: np.ndarray) -> None:
        prepared = (
            _unit_norm(values)
            if self.config.normalize_before
            else np.asarray(values, dtype=np.float32)
        )
        self.reducer.fit(prepared)
        self._fitted = True

    def transform(self, values: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("PostProcessingPipeline must be fitted before transform")
        return apply_reduction(
            values,
            self.reducer,
            normalize_before=self.config.normalize_before,
            normalize_after=self.config.normalize_after,
        )


def reducer_from_config(
    config: PostProcConfig,
    source_dim: int,
    *,
    geopres_weights: np.ndarray | None = None,
) -> Reducer:
    """Construct the configured reducer for one cached embedding dimension."""
    target_dim = config.target_dim
    seed = config.resolved_dr_seed
    if config.dr_method == "none":
        return Identity()
    if target_dim is None:
        raise ValueError(f"{config.dr_method} requires target_dim")
    if target_dim > source_dim:
        raise ValueError("target_dim exceeds source_dim")
    if config.dr_method == "truncate":
        return Truncate(target_dim)
    if config.dr_method == "pca":
        return PCA(target_dim, seed)
    if config.dr_method == "pca_ror":
        return PCARotation(target_dim, seed)
    if config.dr_method == "geopres":
        if geopres_weights is None:
            raise ValueError("geopres_weights is required for dr_method='geopres'")
        if weights_id(geopres_weights) != config.dr_weights_id:
            raise ValueError(
                f"geopres_weights hash to {weights_id(geopres_weights)}, but the config "
                f"names dr_weights_id={config.dr_weights_id}; the run id would not "
                "identify the projection that was applied"
            )
        return GeoPres(weights=geopres_weights)
    if config.dr_method == "random_projection":
        return RandomProjection(target_dim, seed)
    if config.dr_method == "random_selection":
        return RandomSelection(target_dim, seed)
    raise ValueError(f"Unsupported dr_method: {config.dr_method}")
