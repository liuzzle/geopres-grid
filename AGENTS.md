# AGENTS.md — geopres-grid

A configuration grid over `(backbone × dimensionality reduction × quantization)`
evaluated on MTEB. Single-author thesis research code. Read `README.md` first for the
model set, the version rationale, and the work-package sequence.

## Quick start

```bash
uv sync
cp .env.example .env          # PROJECT_ROOT is required
uv run python scripts/smoke_backbones.py
```

This is a real installed package. Imports are absolute (`from geopres_grid.config
import ...`) and there is no `PYTHONPATH` to set — unlike the upstream GeoPres repo,
which used bare imports plus `PYTHONPATH=geopres`.

## Pinned dependencies — do not bump without reading this

- `sentence-transformers>=5.7.0,<6.0.0` — the floor is not a preference:
  `lightonai/mDenseOn` declares its modules under
  `sentence_transformers.base.modules.*`, which does not exist before 5.4. The ceiling
  is supervisor guidance — 6.x introduces new bugs and compatibility issues. Do not
  raise it without retesting the whole model set.
- `transformers==4.56.0`, `tokenizers>=0.22.0,<=0.23.0`,
  `huggingface-hub>=0.34.0,<1.0` — the combination the supervisor tested, verified
  here on 4 of 5 backbones. Staying on the 4.x line also keeps
  `Alibaba-NLP/gte-multilingual-base` working at encode time. **But the 4.x line
  blocks `lightonai/mDenseOn`**, whose tokenizer needs transformers 5.x — see README,
  "mDenseOn is blocked by the transformers major version". Unresolved; do not assume
  all five backbones load.
- `mteb==2.15.1` — work package G depends on `abstasks/retrieval.py` dispatching to
  a model that implements `SearchProtocol`. That is the hook for bit-exact scoring
  and it is version-sensitive. Re-verify it before bumping.
- `torchsort` — optional extra, imported lazily. Never make it a hard dependency;
  it builds from source against the CUDA toolchain.

After any dependency change, `scripts/smoke_backbones.py` is the regression test.

## Conventions

- **The registry is the source of truth.** Every fact the pipeline branches on —
  dimension, pinned sha, prompt handling, whether the stack normalises — lives in
  `geopres_grid/backbones.py`. Do not re-derive these inline; add a field.
- **Never hard-code a device.** Use `config.resolve_device()`. Evaluation must run
  on CPU; only precomputation needs a GPU.
- **Large artefacts never enter the repo.** Everything goes under `$STORAGE_PATH`.
- **`WP-x seam` comments mark deliberate temporary code.** They point at the layout
  or convention a later work package replaces. Do not "clean them up" — update the
  marker when the work package lands.
- Tests live in `tests/` and run with `uv run pytest`. The upstream repo had no test
  suite and its AGENTS.md said not to add one; that does not apply here. The cache
  correctness test is a required deliverable of work package C.

## Things that will bite

- **A pinned sha does not pin a `trust_remote_code` model.** mGTE pulls its
  modelling code from `Alibaba-NLP/new-impl`, a separate unpinned repo. Anything
  claiming reproducibility from the registry sha alone is overclaiming.
- **MTEB's `CachedEmbeddingWrapper` keys on `sha256(text)` scoped by task name
  only** — no split, no subset, no prompt type. Three of the five backbones use
  asymmetric prompts, so a query and a document with the same text collide. Do not
  use it unmodified.
- **MTEB's `CompressionWrapper` refits min/max inside every `encode` call**, so
  documents get a different affine map per 50k corpus chunk than queries do, and it
  returns un-dequantized integer levels. Quantization calibration must be fitted
  once and shared across both sides.
- **`get_sentence_embedding_dimension` is renamed in sentence-transformers 5.7** to
  `get_embedding_dimension`; the old name works but emits a `FutureWarning`. Prefer
  the new name with a `getattr` fallback. Relevant to work package D, which
  overrides it on the reduced model.
- **Three of five stacks end in `Normalize`, two do not.** Never append a projection
  to the module stack and assume a consistent input geometry.
