#!/usr/bin/env python
"""List cached rows whose text sentence-transformers 6 would have read as media.

Before `cache.as_text_inputs`, sentence-transformers 6 guessed each passage's
modality from the raw string, and for a model that accepts images, video or audio
(EmbeddingGemma-2) it fetched a bare image or YouTube URL and embedded the media,
not the text. Where the fetch failed the job died; where it succeeded, the cache
holds a media vector under a text key. This finds such rows: every row whose
unprompted text `infer_modality` classifies as anything but text, with every media
modality allowed. Reads `ids.parquet` only -- no model, no GPU, no vectors.

    uv run --project envs/sentence_transformers6 python scripts/scan_media_rows.py --backbone embeddinggemma-2

Exit status 1 if any row is found. Run it in an environment with
sentence-transformers 6 (the heuristic it reproduces is that version's).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyarrow.parquet as pq

from geopres_grid.backbones import get
from geopres_grid.cache import backbone_cache_root
from geopres_grid.config import EMBEDDING_CACHE_PATH

MEDIA_PREFIXES = ("http://", "https://", "data:image/")
MEDIA_SUFFIXES = (
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp",
    ".mp4", ".avi", ".mov", ".wmv", ".flv", ".mkv",
    ".mp3", ".wav", ".ogg", ".flac", ".aac",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--cache-root", default=str(EMBEDDING_CACHE_PATH))
    args = parser.parse_args(argv)

    from sentence_transformers.base.modality import infer_modality

    backbone = get(args.backbone)
    # Longest first, so a prompt that extends another is stripped whole.
    prefixes = sorted({prompt for prompt in backbone.prompts.values() if prompt}, key=len, reverse=True)
    root = backbone_cache_root(args.cache_root, backbone)
    found = blocks = 0
    for ids_file in sorted(Path(root).glob("*/*/*/*/*/ids.parquet")):
        blocks += 1
        hits = []
        for batch in pq.ParquetFile(ids_file).iter_batches(batch_size=200_000, columns=["row_idx", "prompted_text"]):
            for row, prompted in zip(batch.column("row_idx").to_pylist(), batch.column("prompted_text").to_pylist()):
                raw = next((prompted[len(p):] for p in prefixes if prompted.startswith(p)), prompted)
                # Cheap filter first; the heuristic only ever fires on these.
                if not (raw.startswith(MEDIA_PREFIXES) or raw.lower().endswith(MEDIA_SUFFIXES)):
                    continue
                modality = infer_modality(raw)
                if modality != "text":
                    hits.append((row, modality, raw))
        block = ids_file.parent.relative_to(root)
        print(f"{block}: {len(hits)}", flush=True)
        for row, modality, raw in hits:
            print(f"    row {row} {modality}: {raw[:120]}")
        found += len(hits)
    print(f"{found} media-like rows in {blocks} blocks under {root}")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
