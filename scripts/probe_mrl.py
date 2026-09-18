#!/usr/bin/env python
"""Empirical Matryoshka probe: is a model's *prefix* privileged, whatever its card says?

Why this exists
---------------
`backbones.mrl_dims` records what a model *documents*. For three of the five
backbones it is `None` with `mrl_checked=True`, which means "we looked and found no
source" -- not "the property is absent". A vendor can train with a Matryoshka loss
and never say so. mGTE and mDenseOn are the cautionary case in the other direction:
both cards are silent and both turned out to be MRL-trained per their papers.

So this script stops reading and measures. It is the direct test of the property
that `mrl_dims` only ever recorded second-hand.

The test
--------
MRL's defining behaviour is that the first k coordinates are a usable standalone
embedding: the loss is a sum of contrastive terms over nested prefixes
(Kusupati et al., "Matryoshka Representation Learning", arXiv:2205.13147, §3), so
information is deliberately front-loaded into low indices. At a fixed budget k,
compare three ways of getting k dimensions out of d:

    truncate   X[:, :k]                   the MRL operation
    randsel    X[:, k random columns]     the same operation, different columns
    pca        PCA(k) fitted on the docs  the strong linear reference

and read the ordering:

    truncate ~ randsel   the prefix is worth no more than any other k columns.
                         No MRL structure.
    truncate ~ pca       the prefix is about as good as the best linear k-dim
                         subspace of the data. That is MRL behaviour whether or
                         not anybody documented it.

`randsel` is the control that a truncation-vs-PCA comparison alone cannot provide.
PCA is a different class of object -- fitted, centred, rotating -- so "truncation
loses to PCA" has a boring explanation available: PCA saw the corpus. `randsel` is
the *identical* operation, differing only in which k columns are kept, so it
isolates exactly the claim under test and gives a per-model null instead of a
hand-picked threshold.

Reported statistic:

    mrl_gain = (ndcg_trunc - ndcg_rand) / (ndcg_pca - ndcg_rand)

0 means the prefix is no better than random columns; 1 means it is as good as PCA.
It is a ratio of differences, so it is comparable across models with very different
absolute nDCG -- which matters here, because the five backbones are not equally
strong to begin with.

Calibration instead of a magic number
-------------------------------------
mGTE and mDenseOn are MRL-trained per their papers (`backbones.mrl_source`). They
are the positive controls: whatever this statistic reads on them is what
"MRL-trained" looks like under this measurement, on these tasks, at these k. A
number for harrier-0.6b is only interpretable next to them. That is why the default
backbone set is all five rather than the three undocumented ones -- and why mGTE
being in the 4.x environment and mDenseOn in the 5.x one means the probe has to be
run twice and the two JSONs merged (see --out and `merge` below).

What a negative result does and does not establish
--------------------------------------------------
A low `mrl_gain` is evidence that truncation is not a supported reduction method
*for these dimensions on these tasks*, which is exactly the claim the grid needs.
It is not proof that no Matryoshka term appeared anywhere in training. A positive
result is the stronger direction: prefix >> random columns has no explanation other
than the prefix having been trained to stand alone.

Prompts
-------
Query and document sides are encoded with the prompts the registry records, which
for three of the five backbones differ. Getting this wrong depresses retrieval on
both sides and would contaminate the comparison. Both harriers take an instruct
prefix on the query and *nothing* on the document; LFM2.5 and mDenseOn take
`query: ` / `document: `. The per-row keys written next to the cached embeddings
are `sha256(prompt + text)`, not `sha256(text)`, which is the same key WP-C's cache
will use and the reason a query and a document with identical text cannot collide.

WP-C seam: the little embedding cache below is a stand-in. When WP-C lands, this
script should read through `GeoPresCache` instead of its own .npy files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from geopres_grid import backbones as B

# `geopres_grid.config` is imported inside main(), not here: it raises at import
# time when PROJECT_ROOT is unset, and the reduction and scoring functions below
# have no business requiring a configured storage path. tests/test_probe_mrl.py
# imports this module to check the statistic and must not need an .env.

DEFAULT_TASKS: tuple[str, ...] = (
    "NanoSciFactRetrieval",
    "NanoNFCorpusRetrieval",
    "NanoFiQA2018Retrieval",
    "NanoMSMARCORetrieval",
    "NanoNQRetrieval",
    "NanoQuoraRetrieval",
)
"""Six NanoBEIR tasks spanning scientific claims, medical, financial QA, web search,
open-domain QA and duplicate-question. 50 queries each, so no single task carries the
conclusion; the spread is there so a result cannot be an artefact of one domain."""

DEFAULT_DIMS: tuple[int, ...] = (32, 64, 128, 256, 512)
N_RAND_SEEDS = 8
TOP_K = 100


# --------------------------------------------------------------------------- data


@dataclass
class TaskData:
    name: str
    split: str
    doc_ids: list[str]
    doc_texts: list[str]
    query_ids: list[str]
    query_texts: list[str]
    qrels: dict[str, dict[str, int]]
    ignore_identical_ids: bool


def load_task(name: str) -> TaskData:
    import mteb

    task = mteb.get_tasks(tasks=[name])[0]
    task.load_data()
    split = task.metadata.eval_splits[0]

    corpus = task.corpus[split]
    queries = task.queries[split]
    qrels = task.relevant_docs[split]

    doc_ids = list(corpus)
    doc_texts = []
    for did in doc_ids:
        entry = corpus[did]
        if isinstance(entry, str):
            doc_texts.append(entry)
        else:
            title = entry.get("title") or ""
            doc_texts.append(f"{title} {entry['text']}".strip())

    query_ids = list(queries)
    query_texts = []
    for qid in query_ids:
        q = queries[qid]
        if not isinstance(q, str):
            raise TypeError(
                f"{name}: query {qid} is {type(q).__name__}, not str. This probe only "
                "handles single-turn text queries."
            )
        query_texts.append(q)

    return TaskData(
        name=name,
        split=split,
        doc_ids=doc_ids,
        doc_texts=doc_texts,
        query_ids=query_ids,
        query_texts=query_texts,
        qrels={q: {d: int(v) for d, v in rel.items()} for q, rel in qrels.items()},
        # MTEB drops results whose doc id equals the query id on these tasks; not
        # doing the same would inflate every number here relative to the leaderboard.
        ignore_identical_ids=bool(getattr(task, "ignore_identical_ids", False)),
    )


# ---------------------------------------------------------------------- encoding


def row_keys(prompt: str, texts: list[str]) -> list[str]:
    """`sha256(prompted_text)` per row -- the WP-C cache key, used here as the
    check that a cached array belongs to the side it is filed under."""
    return [hashlib.sha256((prompt + t).encode("utf-8")).hexdigest() for t in texts]


def digest(keys: list[str]) -> str:
    h = hashlib.sha256()
    for k in keys:
        h.update(k.encode("ascii"))
    return h.hexdigest()


def embed_side(
    model,
    backbone: B.Backbone,
    task: TaskData,
    side: str,
    texts: list[str],
    cache_dir: Path,
    batch_size: int,
    max_seq_length: int,
    dtype: str,
) -> np.ndarray:
    """Encode one side of one task, caching to .npy keyed by the *prompted* text."""
    prompt = backbone.prompt_for(side)
    prompt_name = backbone.prompt_name_for(side)

    out_dir = (
        cache_dir
        / f"{backbone.slug}@{backbone.revision[:12]}"
        / f"msl{max_seq_length}_{dtype}"
        / task.name
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    arr_path = out_dir / f"{side}.npy"
    meta_path = out_dir / f"{side}.meta.json"

    keys = row_keys(prompt, texts)
    want = {
        "model_id": backbone.model_id,
        "revision": backbone.revision,
        "task": task.name,
        "split": task.split,
        "side": side,
        "prompt_name": prompt_name,
        "prompt": prompt,
        "n": len(texts),
        "keys_sha256": digest(keys),
    }

    if arr_path.exists() and meta_path.exists():
        have = json.loads(meta_path.read_text())
        if have == want:
            return np.load(arr_path)
        print(f"    cache miss ({side}): stored meta differs, re-encoding")

    kwargs = {"batch_size": batch_size, "show_progress_bar": False,
              "convert_to_numpy": True}
    if prompt_name is not None:
        kwargs["prompt_name"] = prompt_name

    # monotonic, not time(): the first long run was interrupted by the machine
    # sleeping, and wall clock reported a 2919-document encode as 10422s. The
    # embeddings were fine -- the computation is deterministic and the cache is
    # keyed on content -- but the throughput line was nonsense.
    t0 = time.monotonic()
    emb = np.asarray(model.encode(texts, **kwargs), dtype=np.float32)
    dt = time.monotonic() - t0
    print(
        f"    {side:8s} n={len(texts):5d} prompt={prompt_name!r:20s} "
        f"{dt:6.1f}s ({len(texts) / max(dt, 1e-9):5.0f}/s)"
    )

    np.save(arr_path, emb)
    meta_path.write_text(json.dumps(want, indent=2))
    (out_dir / f"{side}.keys.json").write_text(json.dumps(keys))
    return emb


# ------------------------------------------------------------------- reductions


def l2_normalize(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, 1e-12)


def reduce_truncate(Xd, Xq, k, **_):
    return Xd[:, :k], Xq[:, :k]


def reduce_randsel(Xd, Xq, k, seed=0, **_):
    rng = np.random.default_rng(seed)
    cols = rng.choice(Xd.shape[1], size=k, replace=False)
    return Xd[:, cols], Xq[:, cols]


def reduce_pca(Xd, Xq, k, seed=42, **_):
    """PCA fitted on the documents only.

    Documents only, because that is the deployable version: at index time the
    corpus exists and the queries do not. It also keeps the fit identical to
    `geopres_grid.fit_pca`, which fits on a corpus sample with
    `svd_solver="auto", random_state=seed`.

    Fitting on the evaluation corpus gives PCA an in-domain advantage this probe
    does not correct for. That is deliberate: it makes PCA a *harder* reference, so
    a truncation score close to PCA is conservative evidence of MRL rather than
    generous.
    """
    from sklearn.decomposition import PCA

    pca = PCA(n_components=k, svd_solver="auto", random_state=seed)
    pca.fit(Xd)
    return pca.transform(Xd), pca.transform(Xq)


# ---------------------------------------------------------------------- scoring


def ndcg_at_10(Xq: np.ndarray, Xd: np.ndarray, task: TaskData) -> dict[str, float]:
    """Per-query nDCG@10 through MTEB's own scorer (pytrec_eval), cosine ranking."""
    import pytrec_eval

    Xq = l2_normalize(np.ascontiguousarray(Xq, dtype=np.float32))
    Xd = l2_normalize(np.ascontiguousarray(Xd, dtype=np.float32))
    sims = Xq @ Xd.T

    top_k = min(TOP_K, sims.shape[1])
    part = np.argpartition(-sims, kth=top_k - 1, axis=1)[:, :top_k]

    run: dict[str, dict[str, float]] = {}
    for i, qid in enumerate(task.query_ids):
        run[qid] = {task.doc_ids[j]: float(sims[i, j]) for j in part[i]}

    if task.ignore_identical_ids:
        for qid, rels in run.items():
            rels.pop(qid, None)

    evaluator = pytrec_eval.RelevanceEvaluator(task.qrels, {"ndcg_cut.10"})
    return {qid: s["ndcg_cut_10"] for qid, s in evaluator.evaluate(run).items()}


def mean(d: dict[str, float]) -> float:
    return float(np.mean(list(d.values()))) if d else float("nan")


MIN_RESOLVING_POWER = 0.02
"""How far PCA must beat random columns before `mrl_gain` means anything.

`mrl_gain` divides by `pca - rand`. As k approaches d that gap closes -- keeping 256
of 640 coordinates of a unit vector already preserves most of the cosine geometry,
so every method scores the same and the ratio is 0/0. Below this gap (in nDCG@10
points) the probe cannot distinguish the hypotheses at that k and reports `None`
rather than a large number manufactured by a small denominator. The informative
regime is small k/d.
"""


def mrl_gain(trunc: float, pca: float, rand: float) -> float | None:
    """Where truncation sits between "any k columns" (0) and "the best k" (1)."""
    if pca - rand < MIN_RESOLVING_POWER:
        return None
    return (trunc - rand) / (pca - rand)


def paired_bootstrap(
    a: dict[str, float], b: dict[str, float], n: int = 10_000, seed: int = 0
) -> dict[str, float]:
    """95% CI on the per-query mean of (a - b), resampling queries.

    Paired because both arms are scored on the same queries, and 50 queries per
    NanoBEIR task is small enough that the between-query variance would otherwise
    swamp the effect being measured.
    """
    shared = sorted(set(a) & set(b))
    diff = np.array([a[q] - b[q] for q in shared], dtype=np.float64)
    if diff.size == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n, diff.size))
    means = diff[idx].mean(axis=1)
    return {
        "mean": float(diff.mean()),
        "lo": float(np.percentile(means, 2.5)),
        "hi": float(np.percentile(means, 97.5)),
        "n": int(diff.size),
    }


# ------------------------------------------------------------------------- main


def probe_backbone(
    backbone: B.Backbone,
    tasks: list[str],
    dims: tuple[int, ...],
    cache_dir: Path,
    device: str,
    batch_size: int,
    max_seq_length: int,
    dtype: str,
    pca_seed: int,
) -> dict:
    from sentence_transformers import SentenceTransformer

    B.require_runnable(backbone)

    print(f"\n=== {backbone.key}  ({backbone.model_id})")
    print(
        f"    prompts: query={backbone.prompt_name_for('query')!r} "
        f"document={backbone.prompt_name_for('document')!r} "
        f"asymmetric={backbone.has_asymmetric_prompts}"
    )

    load_kwargs = {"device": device, "model_kwargs": {"dtype": dtype}}
    if backbone.trust_remote_code:
        load_kwargs["trust_remote_code"] = True
    model = SentenceTransformer(backbone.model_id, revision=backbone.revision,
                                **load_kwargs)
    model.max_seq_length = max_seq_length

    usable = [k for k in dims if k < backbone.native_dim]
    result: dict = {
        "backbone": backbone.key,
        "model_id": backbone.model_id,
        "revision": backbone.revision,
        "native_dim": backbone.native_dim,
        "documented_mrl_dims": list(backbone.mrl_dims or ()),
        "query_prompt_name": backbone.prompt_name_for("query"),
        "document_prompt_name": backbone.prompt_name_for("document"),
        "max_seq_length": max_seq_length,
        "dtype": dtype,
        "dims": usable,
        "tasks": {},
    }

    for task_name in tasks:
        print(f"  {task_name}")
        task = load_task(task_name)
        Xd = embed_side(model, backbone, task, "document", task.doc_texts,
                        cache_dir, batch_size, max_seq_length, dtype)
        Xq = embed_side(model, backbone, task, "query", task.query_texts,
                        cache_dir, batch_size, max_seq_length, dtype)

        # Uniform geometry: the stack ends in Normalize for three of five models
        # and not for the other two, so normalise here and reduce unit vectors in
        # every case. Matches WP-D's planned normalize -> reduce -> normalize.
        Xd = l2_normalize(Xd)
        Xq = l2_normalize(Xq)

        entry: dict = {
            "n_docs": len(task.doc_ids),
            "n_queries": len(task.query_ids),
            "ignore_identical_ids": task.ignore_identical_ids,
            "full": {},
            "dims": {},
        }
        full_pq = ndcg_at_10(Xq, Xd, task)
        entry["full"] = {"ndcg@10": mean(full_pq), "per_query": full_pq}

        # Variance front-loading: cheap second signal, no retrieval involved.
        coord_var = Xd.var(axis=0)
        total_var = float(coord_var.sum())

        for k in usable:
            dk: dict = {}

            Dt, Qt = reduce_truncate(Xd, Xq, k)
            trunc_pq = ndcg_at_10(Qt, Dt, task)
            dk["truncate"] = {"ndcg@10": mean(trunc_pq), "per_query": trunc_pq}

            Dp, Qp = reduce_pca(Xd, Xq, k, seed=pca_seed)
            pca_pq = ndcg_at_10(Qp, Dp, task)
            dk["pca"] = {"ndcg@10": mean(pca_pq), "per_query": pca_pq}

            rand_scores, rand_pq = [], {}
            for seed in range(N_RAND_SEEDS):
                Dr, Qr = reduce_randsel(Xd, Xq, k, seed=seed)
                pq = ndcg_at_10(Qr, Dr, task)
                rand_scores.append(mean(pq))
                for qid, v in pq.items():
                    rand_pq.setdefault(qid, []).append(v)
            dk["randsel"] = {
                "ndcg@10": float(np.mean(rand_scores)),
                "std": float(np.std(rand_scores)),
                "seeds": rand_scores,
                # Per query, averaged over the seeds: the null this query's
                # truncation score is compared against in the paired bootstrap.
                "per_query": {q: float(np.mean(v)) for q, v in rand_pq.items()},
            }

            dk["variance_in_prefix"] = float(coord_var[:k].sum() / total_var)
            dk["variance_if_uniform"] = k / backbone.native_dim

            t, p, r = (dk["truncate"]["ndcg@10"], dk["pca"]["ndcg@10"],
                       dk["randsel"]["ndcg@10"])
            full = entry["full"]["ndcg@10"]
            dk["retention_vs_full"] = t / full if full else float("nan")
            dk["truncate_over_pca"] = t / p if p else float("nan")
            dk["resolving_power"] = p - r
            dk["mrl_gain"] = mrl_gain(t, p, r)

            gain = dk["mrl_gain"]
            print(
                f"    k={k:4d}  trunc={t:.4f}  pca={p:.4f}  rand={r:.4f}  "
                + (f"gain={gain:+.3f}" if gain is not None
                   else f"gain=n/a (pca-rand={p - r:+.3f})")
            )
            entry["dims"][str(k)] = dk

        result["tasks"][task_name] = entry

    del model
    return result


def aggregate(results: list[dict]) -> list[dict]:
    """Pool per-query nDCG across tasks, then recompute the statistic.

    Pooled rather than a mean of task means: 50 queries per task is small, and
    `mrl_gain` is a ratio of differences, so averaging ratios each computed on 50
    queries is noisier than computing one ratio on the pooled ~250.
    """
    rows = []
    for res in results:
        for k in res["dims"]:
            def pq(method: str) -> dict[str, float]:
                """Per-query scores, task-qualified so ids cannot collide."""
                out = {}
                for tname, t in res["tasks"].items():
                    src = (t["full"] if method == "full" else t["dims"][str(k)][method])
                    for qid, v in src["per_query"].items():
                        out[f"{tname}:{qid}"] = v
                return out

            full_pq, trunc_pq = pq("full"), pq("truncate")
            pca_pq, rand_pq = pq("pca"), pq("randsel")
            full = float(np.mean(list(full_pq.values())))
            trunc = float(np.mean(list(trunc_pq.values())))
            pca = float(np.mean(list(pca_pq.values())))
            rand = float(np.mean(list(rand_pq.values())))

            rows.append({
                "backbone": res["backbone"],
                "k": k,
                "k_over_d": k / res["native_dim"],
                "documented_mrl": k in res["documented_mrl_dims"],
                "ndcg_full": full,
                "ndcg_truncate": trunc,
                "ndcg_pca": pca,
                "ndcg_randsel": rand,
                "retention_vs_full": trunc / full if full else float("nan"),
                "truncate_over_pca": trunc / pca if pca else float("nan"),
                "resolving_power": pca - rand,
                "mrl_gain": mrl_gain(trunc, pca, rand),
                # The claim under test, stated as an effect with an interval:
                # does the prefix beat an arbitrary set of k columns at all?
                "truncate_minus_randsel": paired_bootstrap(trunc_pq, rand_pq),
                "truncate_minus_pca": paired_bootstrap(trunc_pq, pca_pq),
                "variance_in_prefix": float(np.mean([
                    t["dims"][str(k)]["variance_in_prefix"] for t in res["tasks"].values()
                ])),
                "variance_if_uniform": float(np.mean([
                    t["dims"][str(k)]["variance_if_uniform"] for t in res["tasks"].values()
                ])),
            })
    return rows


def verdict(row: dict) -> str:
    """One label per (model, k), from the bootstrap rather than from eyeballing.

    Deliberately reports only the *direction* -- whether the prefix beats an
    arbitrary set of k columns at all -- and leaves the magnitude to `mrl_gain`,
    read against the two known-MRL controls.

    An earlier version had absolute bands ("gain >= 0.8 means MRL-like"). The first
    real numbers killed it: mGTE, which is MRL-trained at every multiple of 32 per
    arXiv:2407.19669 eq. 3, reads roughly 0.35-0.49 at k=32. A 0.8 threshold would
    have declared a documented MRL model non-MRL. That is the whole argument for
    running the controls through the same measurement instead of picking a cutoff:
    "as good as PCA" is not what MRL training buys you at aggressive truncation, so
    it is the wrong yardstick. What MRL buys is "much better than an arbitrary
    subspace", which is what the interval below tests.

    Deliberately per-k, not per-model: whether a prefix is usable is a question
    about a particular k, and the answer changes with k/d.
    """
    if row["resolving_power"] < MIN_RESOLVING_POWER:
        return "no resolution"
    ci = row["truncate_minus_randsel"]
    if ci["hi"] < 0:
        return "worse than random"
    if ci["lo"] <= 0:
        return "not distinguishable"
    return "above random"


def markdown_table(rows: list[dict]) -> str:
    header = (
        "| backbone | k | k/d | doc. MRL | full | trunc | PCA | rand "
        "| trunc-rand (95% CI) | mrl_gain | var in prefix (uniform) | verdict |"
    )
    out = [header, "|---" * 12 + "|"]
    for r in rows:
        ci = r["truncate_minus_randsel"]
        gain = r["mrl_gain"]
        out.append(
            f"| `{r['backbone']}` "
            f"| {r['k']} "
            f"| {r['k_over_d']:.2f} "
            f"| {'yes' if r['documented_mrl'] else '-'} "
            f"| {r['ndcg_full']:.4f} "
            f"| {r['ndcg_truncate']:.4f} "
            f"| {r['ndcg_pca']:.4f} "
            f"| {r['ndcg_randsel']:.4f} "
            f"| {ci['mean']:+.4f} [{ci['lo']:+.4f}, {ci['hi']:+.4f}] "
            f"| {'n/a' if gain is None else format(gain, '+.3f')} "
            f"| {r['variance_in_prefix']:.3f} ({r['variance_if_uniform']:.3f}) "
            f"| {verdict(r)} |"
        )
    return "\n".join(out)


def per_task_table(payloads: list[dict], k: int) -> str:
    """Per-task detail at one k, so a saturated task is visible, not averaged away.

    NanoQuoraRetrieval is the case that forced this: duplicate-question retrieval
    over 9-word documents is near-saturated, so at every k the three arms land
    within noise of each other and PCA does not beat random columns at all. Pooling
    it in with the rest dilutes the contrast without changing the ordering. It is
    kept in the run and flagged here rather than dropped, because "this task cannot
    answer the question" is a result about the measurement worth reporting.
    """
    lines = [
        f"Per task at k={k} (trunc / PCA / rand, nDCG@10):",
        "",
        "| backbone | task | trunc | PCA | rand | PCA-rand | mrl_gain |",
        "|---|---|---|---|---|---|---|",
    ]
    for payload in payloads:
        for res in payload["per_backbone"]:
            if k not in res["dims"]:
                continue
            for tname, t in res["tasks"].items():
                d = t["dims"][str(k)]
                trunc = d["truncate"]["ndcg@10"]
                pca = d["pca"]["ndcg@10"]
                rand = d["randsel"]["ndcg@10"]
                gain = mrl_gain(trunc, pca, rand)
                flag = "" if pca - rand >= MIN_RESOLVING_POWER else "  (saturated)"
                lines.append(
                    f"| `{res['backbone']}` | {tname.replace('Retrieval', '')} "
                    f"| {trunc:.4f} | {pca:.4f} | {rand:.4f} | {pca - rand:+.4f} "
                    f"| {'n/a' if gain is None else format(gain, '+.3f')}{flag} |"
                )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--backbones", nargs="*", default=None,
                    help="registry keys; default = every backbone runnable here")
    ap.add_argument("--tasks", nargs="*", default=list(DEFAULT_TASKS))
    ap.add_argument("--dims", nargs="*", type=int, default=list(DEFAULT_DIMS))
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-seq-length", type=int, default=B.PRIMARY_MAX_SEQ_LENGTH)
    ap.add_argument("--dtype", default="float32",
                    help="fp32 by default: the comparison is between reductions, and "
                         "bf16 noise is the same order as the effects being read")
    ap.add_argument("--pca-seed", type=int, default=42)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--out", default=None, help="JSON output path")
    ap.add_argument(
        "--report", nargs="*", default=None, metavar="JSON",
        help="re-render the table from finished runs instead of encoding anything. "
             "Takes several, which is how the transformers 4.x and 5.x halves of the "
             "model set end up in one table: mGTE cannot run in the same environment "
             "as mDenseOn, and those two are the controls.",
    )
    ap.add_argument("--per-task", nargs="*", type=int, default=[32, 64],
                    metavar="K", help="with --report: also break these k down by task")
    args = ap.parse_args(argv)

    if args.report is not None:
        rows: list[dict] = []
        for path in args.report:
            payload = json.loads(Path(path).read_text())
            rows.extend(payload["summary"])
        # Registry order, then k, so the controls read next to the models in question.
        order = {b.key: i for i, b in enumerate(B.all_backbones())}
        rows.sort(key=lambda r: (order.get(r["backbone"], 99), r["k"]))
        print(markdown_table(rows))
        payloads = [json.loads(Path(f).read_text()) for f in args.report]
        for k in args.per_task:
            print()
            print(per_task_table(payloads, k))
        return 0

    storage = os.getenv("STORAGE_PATH")
    if storage is None:
        from geopres_grid.config import STORAGE_PATH as storage  # noqa: N813
    base = Path(storage) / "mrl_probe"
    cache_dir = Path(args.cache_dir) if args.cache_dir else base / "embeddings"
    cache_dir.mkdir(parents=True, exist_ok=True)

    from geopres_grid.config import resolve_device

    device = resolve_device(args.device)
    major = B.installed_transformers_major()

    if args.backbones:
        selected = [B.get(k) for k in args.backbones]
        for b in selected:
            B.require_runnable(b, major)
    else:
        selected = B.runnable_backbones(major)
        keys = {b.key for b in selected}
        skipped = [b.key for b in B.all_backbones() if b.key not in keys]
        if skipped:
            print(f"skipping (wrong environment): {', '.join(skipped)}")

    print(f"device={device}  transformers major={major}  "
          f"max_seq_length={args.max_seq_length}  dtype={args.dtype}")
    print(f"cache: {cache_dir}")

    results = [
        probe_backbone(b, args.tasks, tuple(args.dims), cache_dir, device,
                       args.batch_size, args.max_seq_length, args.dtype,
                       args.pca_seed)
        for b in selected
    ]

    rows = aggregate(results)
    payload = {
        "meta": {
            "device": device,
            "transformers_major": major,
            "max_seq_length": args.max_seq_length,
            "dtype": args.dtype,
            "tasks": args.tasks,
            "dims": args.dims,
            "n_rand_seeds": N_RAND_SEEDS,
            "pca_seed": args.pca_seed,
            "pca_fitted_on": "documents of the same task",
        },
        "per_backbone": results,
        "summary": rows,
    }

    out = Path(args.out) if args.out else base / f"probe_tf{major}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}\n")
    print(markdown_table(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
