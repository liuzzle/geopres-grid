"""Does the probe's statistic actually respond to the property it claims to measure?

`scripts/probe_mrl.py` reports `mrl_gain`, which is supposed to read ~1 when a
model's first k coordinates are privileged and ~0 when they are not. That is a
claim about the statistic, and it can be checked without touching a model: build
embeddings that have the property by construction, build embeddings that provably
do not, and see whether the number separates them.

The construction of the negative is the point. Take the front-loaded embeddings and
apply a random orthogonal rotation. The rotation preserves every pairwise cosine, so
retrieval at full width is *identical* and PCA -- which is rotation-invariant, it
finds the top-variance subspace in whatever basis -- is identical too. The only
thing it destroys is the privilege of the prefix. Any statistic that separates the
two is therefore measuring exactly prefix privilege and nothing else.
"""

from __future__ import annotations

import numpy as np
import pytest

import probe_mrl as probe  # via tests/conftest.py


DIM = 128
N_DOCS = 400
N_QUERIES = 60
K = 16


def build_task(Xd: np.ndarray, Xq: np.ndarray) -> probe.TaskData:
    doc_ids = [f"d{i}" for i in range(Xd.shape[0])]
    query_ids = [f"q{i}" for i in range(Xq.shape[0])]
    return probe.TaskData(
        name="synthetic",
        split="test",
        doc_ids=doc_ids,
        doc_texts=[""] * len(doc_ids),
        query_ids=query_ids,
        query_texts=[""] * len(query_ids),
        # Query i is a noisy copy of document i.
        qrels={f"q{i}": {f"d{i}": 1} for i in range(Xq.shape[0])},
        ignore_identical_ids=False,
    )


@pytest.fixture(scope="module")
def front_loaded():
    """Embeddings whose information decays with coordinate index.

    A stand-in for what an MRL objective produces: the loss is a sum of contrastive
    terms over nested prefixes, so early coordinates have to carry the signal on
    their own (Kusupati et al., arXiv:2205.13147 §3).
    """
    rng = np.random.default_rng(0)
    scale = 1.0 / (np.arange(DIM) + 1.0) ** 1.5
    docs = rng.normal(size=(N_DOCS, DIM)) * scale
    # Query noise is FLAT across coordinates while the signal decays, so the
    # signal-to-noise ratio decays with index too. That is the property that
    # matters. A first attempt scaled the noise with the signal, which left SNR
    # uniform: the prefix then had more variance but no more usable information,
    # random columns did just as well, and the probe -- correctly -- reported no
    # prefix privilege. Variance front-loading alone is not MRL.
    queries = docs[:N_QUERIES] + rng.normal(size=(N_QUERIES, DIM)) * 0.05
    return probe.l2_normalize(docs), probe.l2_normalize(queries)


def rotate(Xd, Xq, seed=1):
    """A random orthogonal map: same geometry, no privileged basis."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(Xd.shape[1], Xd.shape[1])))
    return Xd @ Q, Xq @ Q


def gain_for(Xd, Xq, k=K):
    task = build_task(Xd, Xq)
    t = probe.mean(probe.ndcg_at_10(*reversed(probe.reduce_truncate(Xd, Xq, k)), task))
    p = probe.mean(probe.ndcg_at_10(*reversed(probe.reduce_pca(Xd, Xq, k)), task))
    r = float(np.mean([
        probe.mean(probe.ndcg_at_10(
            *reversed(probe.reduce_randsel(Xd, Xq, k, seed=s)), task))
        for s in range(probe.N_RAND_SEEDS)
    ]))
    return probe.mrl_gain(t, p, r), t, p, r


def test_gain_is_high_when_the_prefix_is_privileged(front_loaded):
    gain, t, p, r = gain_for(*front_loaded)
    assert gain is not None, "no resolving power in the synthetic setup; k is too large"
    assert t > r + 0.1, f"truncation {t:.3f} barely beat random columns {r:.3f}"
    assert gain > 0.8, f"mrl_gain={gain:.3f} (trunc={t:.3f} pca={p:.3f} rand={r:.3f})"


def test_gain_collapses_under_a_rotation_that_preserves_everything_else(front_loaded):
    Xd, Xq = front_loaded
    Rd, Rq = rotate(Xd, Xq)

    task = build_task(Xd, Xq)
    rot_task = build_task(Rd, Rq)
    full = probe.mean(probe.ndcg_at_10(Xq, Xd, task))
    rot_full = probe.mean(probe.ndcg_at_10(Rq, Rd, rot_task))
    assert full == pytest.approx(rot_full, abs=1e-6), "the rotation changed retrieval"

    gain_before, *_ = gain_for(Xd, Xq)
    gain_after, t, p, r = gain_for(Rd, Rq)
    assert gain_after is not None
    # PCA is rotation-invariant, so the reference arm is unmoved and the whole
    # difference is the prefix losing its privilege.
    assert gain_after < 0.3, f"mrl_gain={gain_after:.3f} after rotation (was {gain_before:.3f})"
    assert abs(t - r) < 0.15, f"rotated prefix still beat random columns: {t:.3f} vs {r:.3f}"


def test_statistic_refuses_to_answer_when_pca_and_random_agree():
    """The guard that stopped k=256 on a 640-d model producing a number from 0/0."""
    assert probe.mrl_gain(0.74, 0.725, 0.747) is None
    assert probe.mrl_gain(0.50, 0.70, 0.40) == pytest.approx(1 / 3)


def test_paired_bootstrap_brackets_a_known_difference():
    a = {f"q{i}": 0.5 for i in range(200)}
    b = {f"q{i}": 0.3 for i in range(200)}
    ci = probe.paired_bootstrap(a, b, n=500, seed=0)
    assert ci["mean"] == pytest.approx(0.2)
    assert ci["lo"] <= 0.2 <= ci["hi"]
    assert ci["n"] == 200
