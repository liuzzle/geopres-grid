"""Run identity: what a cached embedding is, and what was done to it afterwards.

Two configs, hashed separately, because they invalidate at different rates.

`EncodeConfig` describes everything that determines the vector the backbone
produced. It is the cache key: change any of it and the stored embeddings are no
longer the same object. Recomputing it costs GPU hours, so its hashed surface is
kept deliberately small and semantic.

`PostProcConfig` describes what happens to a cached vector on the way to a score --
normalisation, reduction, quantization. All of it is cheap and CPU-bound, so a run
is identified by the pair and the same cache serves every cell of the grid.

Canonical hashing: `sha256` over JSON with sorted keys. Python's built-in `hash()`
is salted per process and must never be used for anything persisted.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

HASH_LENGTH = 16
"""Characters of the hex digest kept. 16 hex chars = 64 bits; at the scale of a
few thousand configurations, a collision is not a practical concern and the short
form stays readable in a filesystem path."""


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no incidental whitespace, stable across runs."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def config_hash(payload: dict[str, Any]) -> str:
    """Stable short hash of a config payload."""
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return digest[:HASH_LENGTH]


def minor_version(version: str) -> str:
    """Reduce a version string to major.minor.

    Hashing full patch versions would invalidate the whole cache on a patch release.
    Tokenisation and pooling semantics change at minor granularity; patch releases
    that move low-order floating-point bits are recorded but not hashed.
    """
    match = re.match(r"^(\d+)\.(\d+)", version)
    return f"{match.group(1)}.{match.group(2)}" if match else version


# --- remote code ------------------------------------------------------------


def external_code_repos(auto_map: dict[str, str] | None) -> list[str]:
    """Repos, other than the model's own, that supply code via `auto_map`.

    An `auto_map` value of the form `"owner/repo--module.Class"` loads code from a
    *different* repository, which the model's own `revision` does not pin. A value
    without `--` is in-repo and therefore already pinned.

    mGTE is the case that matters here: it pulls `configuration.py` and `modeling.py`
    from `Alibaba-NLP/new-impl`, so pinning the backbone sha alone does not pin the
    code that actually runs.
    """
    if not auto_map:
        return []
    repos = {
        value.split("--", 1)[0]
        for value in auto_map.values()
        if isinstance(value, str) and "--" in value
    }
    return sorted(repos)


def resolve_repo_revision(repo_id: str, token: str | None = None) -> str | None:
    """Current commit sha of a Hub repo, or None if it cannot be resolved.

    Resolution failures are not fatal -- the caller records `None` and the gap shows
    up in `meta.json` rather than silently looking pinned.
    """
    try:
        from huggingface_hub import HfApi

        return HfApi().model_info(repo_id, token=token).sha
    except Exception:
        return None


# --- encode side ------------------------------------------------------------


@dataclass(frozen=True)
class EncodeConfig:
    """Identity of a set of cached backbone embeddings."""

    model_id: str
    revision: str
    max_seq_length: int
    dtype: str
    prompts: tuple[tuple[str, str], ...]
    """Resolved prompt name -> prefix, sorted. Empty when the model has none.

    Part of the hash because a prompt prefix changes the text that is encoded. The
    path scopes by prompt *type*; this pins what that type actually means.
    """
    sentence_transformers_version: str
    transformers_version: str
    external_code_revisions: tuple[tuple[str, str], ...] = ()
    """repo id -> commit sha for code loaded from outside the model's own repo."""

    # Recorded, not hashed. See `to_meta`.
    torch_version: str = ""
    device: str = ""
    batch_size: int | None = None
    notes: str = ""

    HASHED: tuple[str, ...] = field(
        default=(
            "model_id",
            "revision",
            "max_seq_length",
            "dtype",
            "prompts",
            "sentence_transformers_version",
            "transformers_version",
            "external_code_revisions",
        ),
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        # Canonical ordering is structural, not something callers must remember.
        object.__setattr__(self, "prompts", tuple(sorted(self.prompts)))
        object.__setattr__(
            self, "external_code_revisions", tuple(sorted(self.external_code_revisions))
        )

    def to_hashable(self) -> dict[str, Any]:
        """The fields that define the embedding, as a JSON-ready dict."""
        payload: dict[str, Any] = {}
        for name in self.HASHED:
            value = getattr(self, name)
            payload[name] = (
                {k: v for k, v in value} if isinstance(value, tuple) else value
            )
        return payload

    @property
    def hash(self) -> str:
        """Short, stable hash. This is the directory name in the cache layout."""
        return config_hash(self.to_hashable())

    @property
    def model_slug(self) -> str:
        return self.model_id.replace("/", "__")

    @property
    def has_unpinned_code(self) -> bool:
        """True when code is loaded from a repo whose revision could not be pinned."""
        return any(sha is None or sha == "" for _, sha in self.external_code_revisions)

    def to_meta(self) -> dict[str, Any]:
        """Everything written to `meta.json`, hashed and unhashed alike.

        `torch_version`, `device` and `batch_size` sit outside the hash: they move
        low-order floating-point bits rather than the meaning of the embedding, and
        hashing them would fragment the cache across machines for no benefit. They
        are recorded so the provenance of any stored block is still recoverable.
        """
        return {
            "encode_config_hash": self.hash,
            "hashed": self.to_hashable(),
            "recorded_only": {
                "torch_version": self.torch_version,
                "device": self.device,
                "batch_size": self.batch_size,
                "notes": self.notes,
            },
        }

    @classmethod
    def from_model(
        cls,
        model: Any,
        model_id: str,
        revision: str,
        *,
        dtype: str = "float32",
        device: str = "",
        batch_size: int | None = None,
        resolve_external_code: bool = True,
        hf_token: str | None = None,
    ) -> EncodeConfig:
        """Build from a loaded SentenceTransformer, reading the live values.

        Reads `max_seq_length` and `prompts` off the model rather than the registry,
        so the config records what the encoder will actually do -- including a
        `max_seq_length` that failed to take effect.
        """
        import sentence_transformers
        import torch
        import transformers

        prompts = {k: v for k, v in (getattr(model, "prompts", None) or {}).items() if v}

        external: list[tuple[str, str]] = []
        if resolve_external_code:
            auto_map = getattr(getattr(model[0], "auto_model", None), "config", None)
            auto_map = getattr(auto_map, "auto_map", None)
            for repo in external_code_repos(auto_map):
                external.append((repo, resolve_repo_revision(repo, hf_token) or ""))

        return cls(
            model_id=model_id,
            revision=revision,
            max_seq_length=int(model.max_seq_length),
            dtype=dtype,
            prompts=tuple(sorted(prompts.items())),
            sentence_transformers_version=minor_version(
                sentence_transformers.__version__
            ),
            transformers_version=minor_version(transformers.__version__),
            external_code_revisions=tuple(external),
            torch_version=torch.__version__,
            device=device,
            batch_size=batch_size,
        )


# --- post-processing side ---------------------------------------------------

BITS_PER_DIM: dict[str, int] = {
    "none": 32,
    "fp16": 16,
    "int8": 8,
    "uint8": 8,
    "equal_count_8": 8,
    "int4": 4,
    "uint4": 4,
    "equal_count_4": 4,
    "binary": 1,
}

DR_METHODS = (
    "none",
    "truncate",
    "pca",
    "pca_ror",
    "geopres",
    "random_projection",
    "random_selection",
)


@dataclass(frozen=True)
class PostProcConfig:
    """What happens to a cached vector on the way to a score.

    Default normalisation is `normalize -> reduce -> normalize`. The first puts all
    backbones on the unit sphere so one calibration can serve them; the second is a
    no-op for cosine scoring and exists only because scalar quantization is not
    scale-invariant and reduction does not preserve norms.
    """

    normalize_before: bool = True
    dr_method: str = "none"
    target_dim: int | None = None
    dr_seed: int | None = None
    """Seed for methods with a random component: pca_ror, random_projection,
    random_selection. Part of the hash, because the same method with a different
    seed is a different transform."""
    normalize_after: bool = True
    quant_method: str = "none"
    quant_symmetric: bool = True
    """Whether one calibration is shared by the query and document sides."""
    quant_calibration_id: str | None = None
    """Hash of the fitted calibration artefact, when the quantizer needs one."""

    def __post_init__(self) -> None:
        if self.dr_method not in DR_METHODS:
            raise ValueError(
                f"Unknown dr_method {self.dr_method!r}. Known: {', '.join(DR_METHODS)}"
            )
        if self.quant_method not in BITS_PER_DIM:
            raise ValueError(
                f"Unknown quant_method {self.quant_method!r}. "
                f"Known: {', '.join(sorted(BITS_PER_DIM))}"
            )
        if self.dr_method != "none" and self.target_dim is None:
            raise ValueError(f"dr_method={self.dr_method!r} requires a target_dim")
        if self.dr_method == "none" and self.target_dim is not None:
            raise ValueError("target_dim is meaningless with dr_method='none'")

    @property
    def hash(self) -> str:
        return config_hash(
            {
                "normalize_before": self.normalize_before,
                "dr_method": self.dr_method,
                "target_dim": self.target_dim,
                "dr_seed": self.dr_seed,
                "normalize_after": self.normalize_after,
                "quant_method": self.quant_method,
                "quant_symmetric": self.quant_symmetric,
                "quant_calibration_id": self.quant_calibration_id,
            }
        )

    @property
    def bits_per_dim(self) -> int:
        return BITS_PER_DIM[self.quant_method]

    def output_dim(self, source_dim: int) -> int:
        return self.target_dim if self.target_dim is not None else source_dim

    def bytes_per_vector(self, source_dim: int) -> int:
        """Storage per stored vector, packed. Binary packs 8 dimensions per byte."""
        bits = self.output_dim(source_dim) * self.bits_per_dim
        return math.ceil(bits / 8)

    def compression_factor(self, source_dim: int) -> float:
        """Against the uncompressed fp32 vector at the backbone's native dimension."""
        return (source_dim * 32) / (self.output_dim(source_dim) * self.bits_per_dim)

    def index_bytes(self, source_dim: int, n_vectors: int) -> int:
        """Total index size, for the configuration table."""
        return self.bytes_per_vector(source_dim) * n_vectors


@dataclass(frozen=True)
class RunId:
    """One cell of the grid: a cached encode plus one post-processing pipeline."""

    encode: EncodeConfig
    postproc: PostProcConfig

    @property
    def value(self) -> str:
        return f"{self.encode.hash}-{self.postproc.hash}"

    def __str__(self) -> str:
        return self.value
