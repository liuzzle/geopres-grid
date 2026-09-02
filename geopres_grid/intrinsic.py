"""Intrinsic evaluation of a projection against its held-out embeddings.

Carried over from `eval_utils.py`. Changes:
  - device is resolved rather than hard-coded to "cuda";
  - the Spearman metric degrades to None when `torchsort` is absent instead of
    taking the whole evaluation down with it;
  - the embedding load path is marked as the seam WP-C replaces.
"""

import json
import logging
import os

import torch

from geopres_grid.config import STORAGE_PATH, resolve_device
from geopres_grid.losses import (
    compute_angular_loss,
    compute_positional_loss,
    compute_spearman_loss,
)

logger = logging.getLogger(__name__)


def find_checkpoint_lowest_val_loss(trained_path: str) -> tuple[float, int | None]:
    """Return (lowest eval_loss, step) across the trainer state's log history."""
    checkpoint_dirs = [
        d for d in os.listdir(trained_path) if d.startswith("checkpoint-")
    ]
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoints found in {trained_path}")

    last_checkpoint = max(
        int(d.split("checkpoint-")[-1]) for d in checkpoint_dirs
    )

    state_path = os.path.join(
        trained_path, f"checkpoint-{last_checkpoint}", "trainer_state.json"
    )
    with open(state_path) as f:
        trainer_state = json.load(f)

    lowest_val = float("inf")
    best_checkpoint = None
    for info_dict in trainer_state["log_history"]:
        if "eval_loss" not in info_dict:
            continue
        if info_dict["eval_loss"] < lowest_val:
            lowest_val = info_dict["eval_loss"]
            best_checkpoint = info_dict["step"]

    return lowest_val, best_checkpoint


def _test_embeddings_path(backbone_model_path: str, dataset_name: str) -> str:
    """Where the held-out embeddings for a backbone live.

    WP-C seam: this is the old `precalculated_embeddings/` layout, keyed by
    dataset and backbone slug only. The embedding cache replaces it with a layout
    scoped by revision, encode-config hash, task, split, subset and prompt type.
    Everything above this function is layout-independent and survives that change.
    """
    return os.path.join(
        STORAGE_PATH,
        "precalculated_embeddings",
        dataset_name.split("/")[-1],
        backbone_model_path.replace("/", "__"),
        "test_embeddings.pt",
    )


@torch.no_grad()
def eval_intrinsic(
    projection: torch.nn.Module,
    backbone_model_path: str,
    dataset_name: str = "sentence-paraphrases",
    checkpoint: int | None = None,
    cache_path: str | None = None,
    model_name: str | None = None,
    spearman_test_batch_size: int | None = 20000,
    spearman_local_or_global: str = "local",
    device: str | None = None,
) -> dict:
    """Score a projection on angular, positional and Spearman preservation."""
    results = {
        "task_name": "IntrinsicEvaluation",
        "checkpoint": checkpoint,
        "spearman_loss": None,
        "angular_loss": None,
        "positional_loss": None,
    }

    test_embeddings_path = _test_embeddings_path(backbone_model_path, dataset_name)
    if not os.path.exists(test_embeddings_path):
        raise FileNotFoundError(
            f"Precalculated embeddings not found at {test_embeddings_path}"
        )

    device = resolve_device(device)
    high_dim_embeddings: torch.Tensor = torch.load(
        test_embeddings_path, weights_only=True
    ).to(device)
    projection = projection.to(device)
    low_dim_embeddings = projection(high_dim_embeddings)

    try:
        spearman_loss = compute_spearman_loss(
            low_dim_embeddings[:spearman_test_batch_size],
            high_dim_embeddings[:spearman_test_batch_size],
            training=False,
            weighted=False,
            local_or_global=spearman_local_or_global,
        )
        results["spearman_loss"] = spearman_loss.item()
    except ImportError as exc:
        logger.warning("Skipping the Spearman metric: %s", exc)

    results["angular_loss"] = compute_angular_loss(
        low_dim_embeddings=low_dim_embeddings,
        high_dim_embeddings=high_dim_embeddings,
        weighted=False,
    ).item()

    results["positional_loss"] = compute_positional_loss(
        low_dim_embeddings=low_dim_embeddings,
        high_dim_embeddings=high_dim_embeddings,
        weighted=False,
    ).item()

    logger.info(
        "Intrinsic results for %s, checkpoint %s: %s",
        model_name,
        checkpoint,
        json.dumps(results, indent=2),
    )

    if not cache_path:
        return results
    if not model_name:
        raise ValueError("model_name must be provided to save intrinsic test results")

    intrinsic_results_path = os.path.join(
        cache_path,
        "results",
        model_name.replace("/", "__"),
        "no_revision_available",
    )
    os.makedirs(intrinsic_results_path, exist_ok=True)
    save_path = os.path.join(intrinsic_results_path, "intrinsic.json")

    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Results saved at: %s", save_path)
    return results
