# AGENTS.md — geopres-grid

A configuration grid over `(backbone × dimensionality reduction × quantization)`
evaluated on MTEB. Single-author thesis research code. Read `README.md` first for the
model set, the version rationale, and the work-package sequence.

## Quick start

```bash
uv sync                                         # root: transformers 4.x
uv sync --project envs/transformers5            # transformers 5.x
uv sync --project envs/sentence_transformers6   # sentence-transformers 6.x
cp .env.example .env                            # PROJECT_ROOT is required
uv run python scripts/smoke_backbones.py
uv run --project envs/transformers5 python scripts/smoke_backbones.py
uv run --project envs/sentence_transformers6 python scripts/smoke_backbones.py
```

**There are three environments and there is no way around it.** mGTE runs only on
`transformers` 4.x, mDenseOn only on 5.x, EmbeddingGemma-2 only on
`sentence-transformers` 6.x; all established by loading the models. Ask
`backbones.runnable_backbones()` rather than assuming, and call
`backbones.require_runnable(backbone)` before any load — the native failures are an
`Unrecognized processing class`, an `IndexError` with a nonsense index inside
`forward`, and a missing `sentence_transformers.base.modules.normalize`, none of which
names the real problem. Only precomputation is affected: the cache holds plain fp32
arrays and its `meta.json`, so evaluation runs in the root environment with no model
loaded (`cache.resolve_encode_config`). See `envs/README.md`.

This is a real installed package. Imports are absolute (`from geopres_grid.config
import ...`) and there is no `PYTHONPATH` to set — unlike the upstream GeoPres repo,
which used bare imports plus `PYTHONPATH=geopres`.

## Pinned dependencies — do not bump without reading this

- `sentence-transformers>=5.7.0,<6.0.0` — the floor is not a preference:
  `lightonai/mDenseOn` declares its modules under
  `sentence_transformers.base.modules.*`, which does not exist before 5.4. The ceiling
  is supervisor guidance — 6.x introduces new bugs and compatibility issues. Do not
  raise it without retesting the whole model set. The one exception is
  `envs/sentence_transformers6`, which overrides it for `google/embeddinggemma-2`
  alone (`>=6.1.0`, with `transformers>=5.19.0`, the first release that has the
  model type).
- `transformers==4.56.0`, `tokenizers>=0.22.0,<=0.23.0`,
  `huggingface-hub>=0.34.0,<1.0` — the combination the supervisor tested, verified
  here on 4 of 5 backbones. Staying on the 4.x line also keeps
  `Alibaba-NLP/gte-multilingual-base` working at encode time. The 4.x line cannot run
  `lightonai/mDenseOn`, which is why `envs/transformers5` exists. Resolved, but do not
  assume all five backbones load **in one environment** — they never will.
- `huggingface-hub>=0.34.0,<1.0` in the root; **`>=1.3.0` in `envs/transformers5`**,
  because transformers 5.x requires it. The two environments therefore differ in the
  hub-client major as well as in transformers. Overridden explicitly in
  `envs/transformers5/pyproject.toml`, not left to the resolver.
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
- **Never use Python's `hash()` for anything persisted.** It is salted per process.
  Use `identity.config_hash`, which is `sha256` over key-sorted JSON. Anything that
  names a directory, a file or a result row goes through it.
- **Think before adding a field to `EncodeConfig`.** Its hash names a directory
  holding GPU-hours of embeddings; a new hashed field invalidates every cache that
  exists. Provenance that does not change the meaning of an embedding belongs in the
  recorded-only block, not in the hash.
- **Never hard-code a device.** Use `config.resolve_device()`. Evaluation must run
  on CPU with no model loaded; only precomputation needs a GPU. Non-retrieval tasks
  are precomputed by a baseline pass through MTEB on the GPU node
  (`evaluation.baseline_pass`), because what they encode depends on MTEB's sampling.
- **One `max_seq_length` for everything: `PRIMARY_MAX_SEQ_LENGTH = 8192`**, the
  smallest documented `context_window`. Never leave it unset: harrier and
  EmbeddingGemma-2 declare none and fall back to their tokenizer's limit.
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
  only** — no split, no subset, no prompt type. Four of the five backbones use
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
- **Four of five stacks end in `Normalize`, mDenseOn's does not.** Never append a
  projection to the module stack and assume a consistent input geometry.
- **`transformers` 5.x loads in `dtype="auto"`.** A bf16 checkpoint (EmbeddingGemma-2)
  then runs in bf16 unless `model_kwargs={"dtype": "float32"}` is passed — which every
  load in this repo does. EmbeddingGemma-2 must never run in fp16.
- **Symmetric tasks pick their prompt by MTEB task type** (`Backbone.task_prompt_names`):
  a dedicated prompt for STS / classification / clustering if the model declares
  one, else the query prompt. One prompt for every symmetric task gave harrier's
  STS instruction to classifiers.
- **Silence in a model card is not evidence.** mGTE and mDenseOn were called non-MRL
  on the basis of silent cards; both are MRL-trained per their papers. Every backbone
  now carries `mrl_checked` plus a source for the claim in either direction, and
  `tests/test_backbones.py` fails if any model is left in the unchecked state.
- **MTEB's result cache is keyed by `ModelMeta` name and revision only**, and its
  default `overwrite_strategy="only-missing"` skips any task whose result file
  exists. A wrapper with an empty `ModelMeta` shares one slot with every other such
  wrapper, so a second backbone or a post-processed run silently returns the first
  run's scores. Every wrapper returns `cache.run_model_meta(...)`: the encode hash
  for `CachedBackbone`, the run id for `PostProcessedBackbone`.
- **MTEB rewrites the text before `encode`.** Documents become
  `(title + " " + text).strip()`; queries get any instruction appended. Cache keys
  hash that prepared text, so precompute goes through MTEB's own `create_dataloader`,
  and misses are encoded from the prepared text -- never re-prepared, which doubles
  the title. `SentenceTransformer.encode` does not accept MTEB's `DataLoader`; misses
  go in as strings with the registry prefix as `prompt=` (`""`, not `None`, on a bare
  side, or a `default_prompt_name` would apply).
- **FEVER and ClimateFEVER share one document block** (`cache.SHARED_CORPORA`,
  `corpus-fever-wikipedia/`). Read a task's blocks through
  `CachedBackbone.block_roots`, not `task_directory`, and never precompute both
  for one backbone in parallel jobs: two writers corrupt the block.
- **Whether a quantizer shares one grid across coordinates decides how much PCA+ROR
  matters.** `EqualCount` (Kisako et al. §3.4: one global table) and `Binary` share
  one; `UniformAffine` fits a range per dimension and absorbs most of PCA's variance
  imbalance itself. Do not make `EqualCount` per-dimension "for consistency" -- the
  global table is the published method.
