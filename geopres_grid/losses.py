"""Distance-preservation losses for the GeoPres projection.

Carried over unchanged in behaviour from `eval_utils.py` in the GeoPres repo. The
one structural change: `torchsort` is imported lazily inside the Spearman path
instead of at module import. It builds from source against the CUDA toolchain and
was the single worst install dependency in the old repo; nothing else in this
package needs it, so it must not be able to break an import.
"""

import torch

__all__ = [
    "compute_positional_loss",
    "compute_angular_loss",
    "compute_spearman_loss",
]

_TORCHSORT_HINT = (
    "The Spearman loss needs `torchsort`, which is not installed. "
    "Install it with `uv sync --extra spearman`, or see the README if the build "
    "fails -- it needs the CUDA toolchain and only ships prebuilt wheels for Linux."
)


def _torchsort():
    """Import torchsort on first use, with an actionable error if it is missing."""
    try:
        import torchsort
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(_TORCHSORT_HINT) from exc
    return torchsort


def compute_positional_loss(
    low_dim_embeddings: torch.Tensor,
    high_dim_embeddings: torch.Tensor,
    weighted: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Pairwise Euclidean distance preservation loss.

    L = (1 / n) * sum_{i<j} w_ij * (d_low(x_i, x_j) - d_high(x_i, x_j))^2
    with w_ij = 1 / (d_high(x_i, x_j) + eps) when `weighted`.
    """
    low_dim_dist = torch.cdist(low_dim_embeddings, low_dim_embeddings, p=2)
    high_dim_dist = torch.cdist(high_dim_embeddings, high_dim_embeddings, p=2)

    n = low_dim_dist.size(0)
    triu_indices = torch.triu_indices(
        n, n, offset=1, device=low_dim_embeddings.device
    )

    low_d = low_dim_dist[triu_indices[0], triu_indices[1]]
    high_d = high_dim_dist[triu_indices[0], triu_indices[1]]

    if not weighted:
        return (low_d - high_d).pow(2).mean()

    weights = 1.0 / (high_d + eps)
    return (weights * (low_d - high_d).pow(2)).mean()


def compute_angular_loss(
    low_dim_embeddings: torch.Tensor,
    high_dim_embeddings: torch.Tensor,
    weighted: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Pairwise cosine similarity preservation loss."""
    low_dim_embeddings = torch.nn.functional.normalize(low_dim_embeddings, p=2, dim=1)
    high_dim_embeddings = torch.nn.functional.normalize(high_dim_embeddings, p=2, dim=1)

    low_dim_sim = torch.mm(low_dim_embeddings, low_dim_embeddings.t())
    high_dim_sim = torch.mm(high_dim_embeddings, high_dim_embeddings.t())

    n = low_dim_sim.size(0)
    triu_indices = torch.triu_indices(
        n, n, offset=1, device=low_dim_embeddings.device
    )

    low_dim_sim_upper = low_dim_sim[triu_indices[0], triu_indices[1]]
    high_dim_sim_upper = high_dim_sim[triu_indices[0], triu_indices[1]]

    if not weighted:
        return (low_dim_sim_upper - high_dim_sim_upper).pow(2).mean()

    weights = 1.0 / (1.0 - high_dim_sim_upper + eps)
    return (weights * (low_dim_sim_upper - high_dim_sim_upper).pow(2)).mean()


def _spearman_global(
    pred: torch.Tensor,
    target: torch.Tensor,
    weighted: bool = False,
    **kw,
) -> torch.Tensor:
    """Differentiable Spearman correlation over all pairs in the batch."""
    torchsort = _torchsort()

    n = target.size(0)
    triu_indices = torch.triu_indices(n, n, offset=1, device=target.device)
    pred = pred[triu_indices[0], triu_indices[1]].unsqueeze(0)
    target = target[triu_indices[0], triu_indices[1]].unsqueeze(0)

    pred_ranks = torchsort.soft_rank(pred, **kw)
    target_ranks = torchsort.soft_rank(target, **kw)

    if weighted:
        weights = target_ranks / target_ranks.sum(dim=1, keepdim=True)
        return _weighted_correlation(pred_ranks, target_ranks, weights).mean()

    pred_ranks = pred_ranks - pred_ranks.mean()
    pred_ranks = pred_ranks / pred_ranks.norm()
    target_ranks = target_ranks - target_ranks.mean()
    target_ranks = target_ranks / target_ranks.norm()
    return (pred_ranks * target_ranks).sum()


def _spearman_local(
    pred: torch.Tensor,
    target: torch.Tensor,
    weighted: bool = False,
    **kw,
) -> torch.Tensor:
    """Differentiable Spearman correlation computed row-wise, then averaged."""
    torchsort = _torchsort()

    n = pred.shape[0]
    m = n - 1

    mask = ~torch.eye(n, dtype=torch.bool, device=pred.device)
    pred_offdiag = pred[mask].reshape(n, m)
    target_offdiag = target[mask].reshape(n, m)

    target_soft_ranks = torchsort.soft_rank(target_offdiag, **kw)
    pred_ranks = torchsort.soft_rank(pred_offdiag, **kw)

    if weighted:
        weights = target_soft_ranks / target_soft_ranks.sum(dim=1, keepdim=True)
    else:
        weights = torch.ones_like(target_soft_ranks)

    return _weighted_correlation(pred_ranks, target_soft_ranks, weights).mean()


def _weighted_correlation(
    pred_ranks: torch.Tensor,
    target_ranks: torch.Tensor,
    weights: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Weighted Pearson correlation between two rank vectors, row-wise."""
    w_sum = weights.sum(dim=1, keepdim=True)
    pred_mean = (weights * pred_ranks).sum(dim=1, keepdim=True) / w_sum
    target_mean = (weights * target_ranks).sum(dim=1, keepdim=True) / w_sum

    pred_centered = pred_ranks - pred_mean
    target_centered = target_ranks - target_mean

    cov = (weights * pred_centered * target_centered).sum(dim=1)
    pred_var = (weights * pred_centered**2).sum(dim=1)
    target_var = (weights * target_centered**2).sum(dim=1)

    return cov / (torch.sqrt(pred_var * target_var) + eps)


def compute_spearman_loss(
    low_dim_embeddings: torch.Tensor,
    high_dim_embeddings: torch.Tensor,
    local_or_global: str = "local",
    training: bool = False,
    weighted: bool = False,
) -> torch.Tensor:
    """1 - Spearman correlation between the low- and high-dimensional similarities."""
    low_dim_embeddings = torch.nn.functional.normalize(low_dim_embeddings, p=2, dim=1)
    high_dim_embeddings = torch.nn.functional.normalize(high_dim_embeddings, p=2, dim=1)

    low_dim_sim = torch.mm(low_dim_embeddings, low_dim_embeddings.t())
    high_dim_sim = torch.mm(high_dim_embeddings, high_dim_embeddings.t())

    with torch.set_grad_enabled(training):
        if local_or_global == "local":
            return 1.0 - _spearman_local(low_dim_sim, high_dim_sim, weighted=weighted)
        return 1.0 - _spearman_global(low_dim_sim, high_dim_sim, weighted=weighted)
