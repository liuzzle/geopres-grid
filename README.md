# geopres-grid

Combining Dimensionality Reduction and Compression for Efficient Indexing.

A configuration grid over `(backbone × dimensionality reduction × quantization)`,
evaluated on MTEB. Built on top of
[GeoPres](https://openreview.net/forum?id=Xc8ulFlMrl); the projection training code
is carried over from that repo, everything about caching, quantization and the
evaluation grid is new here.


### Three environments

No single set of pins runs all five backbones: mGTE only works on `transformers` 4.x,
mDenseOn only on 5.x, and EmbeddingGemma-2 only on `sentence-transformers` 6.x, above
the project's `<6.0` ceiling. All three were established by loading the models.

```bash
uv sync                                        # root: mgte, harrier-270m, harrier-06b
uv sync --project envs/transformers5           # mdenseon, harrier-270m, harrier-06b
uv sync --project envs/sentence_transformers6  # embeddinggemma-2
```

Only precomputation is affected. The cache stores plain fp32 arrays, so evaluation
runs in the root environment for every backbone, with no model loaded. See
`envs/README.md`.


### Backbone table, read from the HF configs

| Model | Architecture | d | `max_seq_length` | Last ST module | Prompts | Notes |
|---|---|---|---|---|---|---|
| `Alibaba-NLP/gte-multilingual-base` | `NewModel` (remote code) | 768 | 8192 | **Normalize** | none | needs `trust_remote_code`; the reason transformers stays on the 4.x line |
| `lightonai/mDenseOn` | `ModernBertModel` | 768 | 8192 | Pooling | `query: ` / `document: ` | new-style `modules.json` ⇒ **requires ST ≥ 5.4**; tokenizer needs transformers ≥ 5.0 ⇒ runs in `envs/transformers5` |
| `microsoft/harrier-oss-v1-270m` | `Gemma3TextModel` | 640 | **no `sentence_bert_config.json`** | **Normalize** | `web_search_query` on queries, documents bare | falls back to tokenizer `model_max_length` (32768); fp16-overflow risk on T4 |
| `microsoft/harrier-oss-v1-0.6b` | `Qwen3Model` | 1024 | **no `sentence_bert_config.json`** | **Normalize** | `web_search_query` on queries, documents bare | |
| `google/embeddinggemma-2` | `EmbeddingGemma2Model` | 768 | **none declared** (card: 8192) | **Normalize** | `task: search result \| query: ` / `title: none \| text: `; own STS, classification and clustering prompts | needs **ST ≥ 6.1** and transformers ≥ 5.19 ⇒ `envs/sentence_transformers6`; multimodal checkpoint, text path only; **never fp16** (card: NaN or silently degraded) |

LFM2.5 (`LiquidAI/LFM2.5-Embedding-350M`) was dropped on 07.10.2026: its 512-token
ceiling capped every model's context and kept MLDR out. Its findings stay in
`logging.md`.

### Sequence length

`PRIMARY_MAX_SEQ_LENGTH = 8192` for every backbone and every task set, the smallest
documented context window of the five. It has to be set explicitly: harrier and
EmbeddingGemma-2 declare no `max_seq_length`, and sentence-transformers then falls
back to the tokenizer's limit (32768, and about 10^30 for EmbeddingGemma-2).
### Matryoshka support

| Model | MRL dimensions | Source |
|---|---|---|
| `Alibaba-NLP/gte-multilingual-base` | every multiple of 32 up to 768 | arXiv:2407.19669 §2.2, eq. 3 |
| `lightonai/mDenseOn` | 128, 256, 512, 768 | arXiv:2607.27178 appendix C.3 |
| `microsoft/harrier-oss-v1-270m` | none documented | model card + Foundry/Bing posts, checked 15.09.2026; probed 18.09.2026 |
| `microsoft/harrier-oss-v1-0.6b` | none documented | model card + Foundry/Bing posts, checked 15.09.2026; probed 18.09.2026 |
| `google/embeddinggemma-2` | 128, 256, 512, 768 | model card (no paper yet), checked 07.10.2026; not probed yet |

So an MRL comparison across the three MRL backbones is legitimate at **128, 256 and
512**. For the two harriers, truncation is not a zero-overhead DR method.
`scripts/probe_mrl.py` tested this directly rather than inferring it from silent model
cards: at the same k, truncation is compared against PCA and against k *randomly
chosen* columns, with the documented-MRL models run through the identical measurement
as controls. On **both harriers truncation is worse than random columns**, so it is
not a naive baseline there but below one (on the dropped LFM2.5 it was
indistinguishable from random). Truncation stays in the grid for every model (meeting
06.10 §1), as an MRL method where documented and as a baseline elsewhere. See
`docs/README-draft.md`.


### Evaluating a grid cell

```bash
uv run python scripts/precompute.py --backbone harrier-270m --tier tier1   # GPU: fill the cache
uv run python scripts/evaluate.py --backbone harrier-270m --task NanoArguAnaRetrieval \
    --dr-method pca_ror --target-dim 128 --quant-method equal_count_4      # CPU: one cell
uv run python scripts/symmetry_ablation.py --backbone harrier-270m --task NanoArguAnaRetrieval
```

Precompute is the only step that needs a GPU or the backbone's own environment.
Retrieval tasks have corpus and queries encoded directly; STS, classification and
clustering get a baseline pass through MTEB on the GPU node, which caches exactly the
inputs MTEB samples and stores the fp32 scores as a by-product. `evaluate.py` then
reads the encode config back from the cache's `meta.json` and runs on CPU, in the
root environment, without loading the model; a cache miss is an error.
`--load-model` restores encoding for a cold cache on a small task set.

### Running the grid

The cells are fixed in `geopres_grid/grid.py`: no DR, plus `truncate`, `pca`,
`pca_ror`, `random_projection` and `random_selection` at 64/128/256/512 (below the
model's native dimension), each with `none`, `fp16`, `int8`, `int4`, `int2`,
`binary` and `equal_count_{8,4,2}`. That is 189 cells per backbone; GeoPres runs
separately once its projections are trained.

```bash
scripts/slurm/submit_grid.sh harrier-270m tier1     # cluster: baselines, then the cells (CPU)
uv run python scripts/run_grid.py --backbone harrier-270m --tier tier0   # locally, one process
```

`submit_grid.sh` scores the fp32 baselines first and starts the shards only once
they succeeded, so no two shards write the same result. MTEB skips any (cell, task)
already scored, so a shard that hit its time limit is resumed by resubmitting it.

### Task sets

| `--tier` | Tasks | Split |
|---|---|---|
| `tier0` | NanoArguAna, STSBenchmark (smoke) | declared |
| `tier1` | NanoBEIR, 13 tasks (primary) | `train`, the only one |
| `tier2` | full BEIR, 15 tasks (supplementary) | `test`; MSMARCO `dev` |
| `nonretrieval` | upstream GeoPres' set: STS12–16, STSBenchmark, SICK-R; AmazonCounterfactual, AmazonReviews, Imdb, ToxicConversations, AmazonPolarity; ArxivClusteringS2S, Reddit, StackExchange clustering | `test` |
| `mldr` | MultiLongDocRetrieval, 13 languages | `test` |

Every language subset is kept, as upstream did. Symmetric tasks (STS, classification,
clustering) get the model's dedicated prompt for the task type if it declares one,
else its query prompt (Andrianos, 06.10, extended per task type 07.10).

`uv run python scripts/collect_results.py` then writes `results.csv` (tidy: one row per
run x task x split, configuration in columns) and `comparison_results.csv` (upstream
GeoPres's wide layout: a row per run, a column per task, per-type averages) to the
results root. Result folders are named by hash; each run's configuration is in
`runs/<run id>.json` beside them.

Per task, `evaluate.py` runs a warm-up pass (the backbone's fp32 baseline), fits PCA
and the quantization table once on up to 10,000 of that task's cached embeddings, and
then scores the cell -- the calibration protocol of Kisako et al. (arXiv:2606.01074,
§3.5), including its stated limitation that it is transductive. Results are stored
per run id under `$EVALUATION_RESULTS_PATH`.

Quantizers follow Kisako et al.'s bit grid (b ∈ {1, 2, 4, 8, 16, 32}): `binary` is
sign quantization, `equal_count_{2,4,8}` their global equal-count lookup table,
`fp16` a float16 round trip, `none` the fp32 original. `int*`/`uint*` are uniform
affine with one range per dimension, the comparison point. The distinction matters:
the variance imbalance that PCA+ROR removes costs most under a shared grid
(`equal_count_*`, `binary`).
