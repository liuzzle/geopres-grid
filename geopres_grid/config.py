"""Environment-backed paths and small runtime helpers.

Carried over from the GeoPres repo, with three changes:
  - `EMBEDDING_CACHE_PATH` added for the WP-C embedding cache.
  - `TRAINED_AUTOENCODERS_PATH` dropped; the autoencoder baseline is out of scope.
  - `resolve_device` added, so evaluation can run without a GPU (meeting 26.08 §5.3).
"""

import logging
import os

import torch
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv(override=True)


DTYPE_ALIASES = {
    "fp16": "float16",
    "float16": "float16",
    "f16": "float16",
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "fp32": "float32",
    "float32": "float32",
    "f32": "float32",
    "float": "float32",
    "fp64": "float64",
    "float64": "float64",
    "f64": "float64",
    "double": "float64",
}


def parse_dtype(dtype_str: str) -> torch.dtype:
    """Map a dtype alias onto a torch dtype."""
    mapped = DTYPE_ALIASES.get(dtype_str, dtype_str)
    dtype = getattr(torch, mapped, None)
    if dtype is None:
        raise ValueError(
            f"Unknown dtype: '{dtype_str}'. "
            f"Supported: {', '.join(sorted(set(DTYPE_ALIASES.values())))}"
        )
    return dtype


def resolve_device(requested: str | None = None) -> str:
    """Pick a torch device, preferring an accelerator but never requiring one.

    Precomputation wants a GPU; evaluation does not (meeting 26.08 §5.3). The old
    repo raised `RuntimeError` when CUDA was missing, which made every CPU-only
    step unrunnable. Pass `requested` to force a device and get a hard failure if
    it is unavailable.
    """
    if requested is not None:
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("device='cuda' was requested but CUDA is not available.")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("device='mps' was requested but MPS is not available.")
        return requested

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    logger.info("No accelerator found, falling back to CPU.")
    return "cpu"


PROJECT_ROOT = os.getenv("PROJECT_ROOT")
if not PROJECT_ROOT:
    raise ValueError(
        "PROJECT_ROOT environment variable not set. "
        "Copy .env.example to .env and set it to an absolute path."
    )

STORAGE_PATH = os.getenv("STORAGE_PATH", os.path.join(PROJECT_ROOT, "storage"))

EMBEDDING_CACHE_PATH = os.getenv(
    "EMBEDDING_CACHE_PATH", os.path.join(STORAGE_PATH, "cache")
)

EVALUATION_RESULTS_PATH = os.getenv(
    "EVALUATION_RESULTS_PATH", os.path.join(STORAGE_PATH, "evaluation_results")
)

TRAINED_MODELS_PATH = os.getenv(
    "TRAINED_MODELS_PATH", os.path.join(STORAGE_PATH, "trained_models")
)
