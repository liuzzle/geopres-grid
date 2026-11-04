#!/usr/bin/env python
"""Probe one backbone at MLDR's length on the GPU before the MLDR precompute.

The MLDR timing run (logging.md §104.5) died in its first batch: SDPA asked for
128 GiB, one fp32 [32, 16, 8192, 8192] score tensor, so attention ran in PyTorch's
math kernel, which materialises the scores. This probe answers, in minutes per
backbone, what precompute would only show by failing:

1. whether SDPA's memory-efficient kernel accepts this backbone's inputs (forced
   with `sdpa_kernel`; when it refuses, PyTorch's warnings name the reason);
2. peak GPU memory and seconds per document at each batch size, on documents cut
   at the cap, with one shorter document per batch so a padding mask exists, as
   in real MLDR batches. The largest batch that fits is MLDR's `--batch-size`
   (recorded, not hashed: the encode hash stays the same).

Nothing is cached or written; output goes to stdout only.
"""

from __future__ import annotations

import argparse
import time
import warnings

from sentence_transformers import SentenceTransformer

from geopres_grid.backbones import PRIMARY_MAX_SEQ_LENGTH, get, require_runnable

SENTENCE = (
    "The committee met again on Tuesday to discuss the new railway line, the budget of the "
    "canton and the letters that readers had sent to the newspaper during the winter. "
)


def probe_texts(batch_size: int, repeats: int = 1200) -> list[str]:
    """`batch_size` documents far over the cap, the last one half as long, so the
    batch is padded and the model builds an attention mask (MLDR batches always
    mix lengths). 1200 repeats are ~30k tokens in every backbone's tokenizer."""
    long = SENTENCE * repeats
    if batch_size == 1:
        return [long]
    return [long] * (batch_size - 1) + [SENTENCE * (repeats // 2)]


def efficient_kernel_check(model: SentenceTransformer) -> str:
    """Encode a batch of two with only SDPA's memory-efficient kernel allowed."""
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                model.encode(probe_texts(2), batch_size=2)
            verdict = "ok (or the model does not attend through SDPA)"
        except (RuntimeError, torch.OutOfMemoryError) as error:
            verdict = f"refused: {type(error).__name__}: {str(error).splitlines()[0]}"
    reasons = sorted({str(w.message) for w in caught if "kernel" in str(w.message).lower()})
    torch.cuda.empty_cache()
    return "\n".join([verdict, *(f"  warning: {reason}" for reason in reasons)])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--max-seq-length", type=int, default=PRIMARY_MAX_SEQ_LENGTH)
    parser.add_argument("--dtype", default="float32")
    args = parser.parse_args(argv)

    import torch

    backbone = get(args.backbone)
    require_runnable(backbone)
    model = SentenceTransformer(
        backbone.model_id,
        revision=backbone.revision,
        device="cuda",
        trust_remote_code=backbone.trust_remote_code,
        model_kwargs={"dtype": args.dtype},
    )
    model.max_seq_length = args.max_seq_length
    try:  # sentence-transformers 6 (embeddinggemma-2's environment) may not expose it
        tokens = len(model.tokenizer(probe_texts(1)[0], truncation=True, max_length=args.max_seq_length)["input_ids"])
    except Exception as error:  # noqa: BLE001 -- the count is informative only
        tokens = f"? ({type(error).__name__})"
    print(f"{args.backbone}: torch {torch.__version__}, {torch.cuda.get_device_name()}, "
          f"{args.dtype}, {tokens} tokens per long document (cap {args.max_seq_length})")
    print(f"memory-efficient SDPA kernel: {efficient_kernel_check(model)}")

    print("batch  peak GiB  s/batch  s/doc")
    for batch_size in args.batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        texts = probe_texts(batch_size)
        try:
            torch.cuda.synchronize()
            start = time.perf_counter()
            model.encode(texts, batch_size=batch_size)
            torch.cuda.synchronize()
        except torch.OutOfMemoryError as error:
            print(f"{batch_size:>5}  OOM: {str(error).splitlines()[0][:120]}")
            break
        seconds = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(f"{batch_size:>5}  {peak:8.1f}  {seconds:7.2f}  {seconds / batch_size:5.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
