#!/usr/bin/env python
"""What the query/document prompt split actually costs, and why it lands in the cache key.

Two questions, both about the same registry fact.

1. Role sensitivity -- does the *same text* get a different vector depending on
   which side it is encoded as? This is the premise of the WP-C cache key. If the
   answer were "barely", `sha256(text)` would be a merely inelegant key; if the
   answer is "substantially", MTEB's `CachedEmbeddingWrapper` returns the wrong
   vector for one of the two sides on four of the five backbones. Measured as the
   cosine between the two encodings of the same string, which costs nothing to run:
   no retrieval, no qrels, just 2N encodes of N strings.

2. Retrieval cost of getting it wrong -- nDCG@10 on NanoBEIR with the documented
   query prompt, with no prompt, and with a plausible wrong prompt. The maintainers
   say of harrier's asymmetric usage: "Yes, this is how the model is trained,
   otherwise you will see a performance degradation", and LFM2.5's card says
   omitting the prefixes "silently degrades retrieval quality". Neither quantifies
   it. This does, on our own task set, so the number in the write-up is ours.

Document embeddings are read from the cache `probe_mrl.py` already filled, so the
only new encoding here is 50 queries per task per variant. Run `probe_mrl.py` first.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_mrl import (  # noqa: E402
    TaskData,
    embed_side,
    l2_normalize,
    load_task,
    mean,
    ndcg_at_10,
)

from geopres_grid import backbones as B  # noqa: E402

DEFAULT_TASKS = ("NanoSciFactRetrieval", "NanoMSMARCORetrieval")


def query_variants(backbone: B.Backbone) -> dict[str, str | None]:
    """Prompt names to try on the query side: documented, bare, and a wrong one.

    The "wrong" variant is a prompt the model really declares but for another task,
    which is the realistic mistake -- not a made-up string. For the harriers that is
    `sts_query`; for the `query: `/`document: ` models it is using the document
    prefix on the query, which is what a symmetric cache wrapper effectively does.
    """
    documented = backbone.prompt_name_for("query")
    variants: dict[str, str | None] = {"documented": documented, "bare": None}
    if backbone.key.startswith("harrier"):
        variants["wrong_task_prompt"] = "sts_query"
    elif documented == "query":
        variants["document_side_prompt"] = "document"
    return variants


def role_sensitivity(
    model, backbone: B.Backbone, texts: list[str], batch_size: int
) -> dict:
    """Cosine between the query-side and document-side encoding of the same text."""
    q_prompt = backbone.prompt_name_for("query")
    d_prompt = backbone.prompt_name_for("document")

    def enc(prompt_name):
        kwargs = {"batch_size": batch_size, "show_progress_bar": False,
                  "convert_to_numpy": True}
        if prompt_name is not None:
            kwargs["prompt_name"] = prompt_name
        return l2_normalize(np.asarray(model.encode(texts, **kwargs), dtype=np.float32))

    as_query, as_document = enc(q_prompt), enc(d_prompt)
    cos = np.sum(as_query * as_document, axis=1)
    return {
        "n_texts": len(texts),
        "query_prompt_name": q_prompt,
        "document_prompt_name": d_prompt,
        "cosine_mean": float(cos.mean()),
        "cosine_min": float(cos.min()),
        "cosine_max": float(cos.max()),
        "cosine_p05": float(np.percentile(cos, 5)),
    }


def probe(backbone: B.Backbone, tasks, cache_dir, device, batch_size,
          max_seq_length, dtype) -> dict:
    from sentence_transformers import SentenceTransformer

    B.require_runnable(backbone)
    print(f"\n=== {backbone.key}")

    load_kwargs = {"device": device, "model_kwargs": {"dtype": dtype}}
    if backbone.trust_remote_code:
        load_kwargs["trust_remote_code"] = True
    model = SentenceTransformer(backbone.model_id, revision=backbone.revision,
                                **load_kwargs)
    model.max_seq_length = max_seq_length

    out: dict = {
        "backbone": backbone.key,
        "model_id": backbone.model_id,
        "asymmetric": backbone.has_asymmetric_prompts,
        "query_prompt_name": backbone.prompt_name_for("query"),
        "document_prompt_name": backbone.prompt_name_for("document"),
        "tasks": {},
    }

    for task_name in tasks:
        task = load_task(task_name)
        Xd = l2_normalize(
            embed_side(model, backbone, task, "document", task.doc_texts,
                       cache_dir, batch_size, max_seq_length, dtype)
        )

        entry: dict = {"variants": {}}
        for label, prompt_name in query_variants(backbone).items():
            kwargs = {"batch_size": batch_size, "show_progress_bar": False,
                      "convert_to_numpy": True}
            if prompt_name is not None:
                kwargs["prompt_name"] = prompt_name
            Xq = l2_normalize(
                np.asarray(model.encode(task.query_texts, **kwargs), dtype=np.float32)
            )
            score = mean(ndcg_at_10(Xq, Xd, task))
            entry["variants"][label] = {"prompt_name": prompt_name, "ndcg@10": score}
            print(f"  {task_name:26s} query={label:22s} "
                  f"prompt={prompt_name!r:20s} nDCG@10={score:.4f}")

        base = entry["variants"]["documented"]["ndcg@10"]
        for label, v in entry["variants"].items():
            v["delta_vs_documented"] = v["ndcg@10"] - base
            v["relative"] = v["ndcg@10"] / base if base else float("nan")

        # Role sensitivity on this task's documents: the cache-key premise.
        sample = task.doc_texts[:200]
        entry["role_sensitivity"] = role_sensitivity(model, backbone, sample, batch_size)
        rs = entry["role_sensitivity"]
        print(f"  {task_name:26s} same text as query vs document: "
              f"cos mean={rs['cosine_mean']:.4f} min={rs['cosine_min']:.4f}")

        out["tasks"][task_name] = entry

    del model
    return out


def markdown(results: list[dict]) -> str:
    lines = [
        "| backbone | task | query prompt | nDCG@10 | Δ vs documented | % of documented |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        for tname, t in r["tasks"].items():
            for label, v in t["variants"].items():
                lines.append(
                    f"| `{r['backbone']}` | {tname} | {label} (`{v['prompt_name']}`) "
                    f"| {v['ndcg@10']:.4f} | {v['delta_vs_documented']:+.4f} "
                    f"| {100 * v['relative']:.1f}% |"
                )
    lines += ["", "| backbone | task | cos(same text as query, as document) | min | p05 |",
              "|---|---|---|---|---|"]
    for r in results:
        for tname, t in r["tasks"].items():
            rs = t["role_sensitivity"]
            lines.append(
                f"| `{r['backbone']}` | {tname} | {rs['cosine_mean']:.4f} "
                f"| {rs['cosine_min']:.4f} | {rs['cosine_p05']:.4f} |"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--backbones", nargs="*", default=None)
    ap.add_argument("--tasks", nargs="*", default=list(DEFAULT_TASKS))
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-seq-length", type=int, default=B.PRIMARY_MAX_SEQ_LENGTH)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    storage = os.getenv("STORAGE_PATH")
    if storage is None:
        from geopres_grid.config import STORAGE_PATH as storage  # noqa: N813
    base = Path(storage) / "mrl_probe"
    cache_dir = Path(args.cache_dir) if args.cache_dir else base / "embeddings"

    from geopres_grid.config import resolve_device

    device = resolve_device(args.device)
    major = B.installed_transformers_major()

    if args.backbones:
        selected = [B.get(k) for k in args.backbones]
    else:
        # Only the asymmetric ones have anything to say here; mGTE is the control
        # that should come out at cosine 1.0 and zero delta.
        selected = B.runnable_backbones(major)

    results = [probe(b, args.tasks, cache_dir, device, args.batch_size,
                     args.max_seq_length, args.dtype) for b in selected]

    out = Path(args.out) if args.out else base / f"prompts_tf{major}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": {"device": device, "dtype": args.dtype,
                                        "max_seq_length": args.max_seq_length,
                                        "tasks": args.tasks},
                               "per_backbone": results}, indent=2))
    print(f"\nwrote {out}\n")
    print(markdown(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
