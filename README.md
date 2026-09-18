# geopres-grid

Combining Dimensionality Reduction and Compression for Efficient Indexing.

A configuration grid over `(backbone × dimensionality reduction × quantization)`,
evaluated on MTEB. Built on top of
[GeoPres](https://openreview.net/forum?id=Xc8ulFlMrl); the projection training code
is carried over from that repo, everything about caching, quantization and the
evaluation grid is new here.


### Two environments

No single set of pins runs all five backbones: mGTE only works on `transformers` 4.x,
mDenseOn only on 5.x. Both were established by loading the models, and neither has an
upstream fix pending.

```bash
uv sync                                  # root: mgte, lfm25, harrier-270m, harrier-06b
uv sync --project envs/transformers5     # mdenseon, lfm25, harrier-270m, harrier-06b
```

Only precomputation is affected — the cache stores plain fp32 arrays, so everything
downstream runs in one environment. See `envs/README.md`.


### Backbone table, read from the HF configs

| Model | Architecture | d | `max_seq_length` | Last ST module | Prompts | Notes |
|---|---|---|---|---|---|---|
| `Alibaba-NLP/gte-multilingual-base` | `NewModel` (remote code) | 768 | 8192 | **Normalize** | none | needs `trust_remote_code`; the reason transformers stays on the 4.x line |
| `lightonai/mDenseOn` | `ModernBertModel` | 768 | 8192 | Pooling | `query: ` / `document: ` | new-style `modules.json` ⇒ **requires ST ≥ 5.4**; tokenizer needs transformers ≥ 5.0 ⇒ runs in `envs/transformers5` |
| `LiquidAI/LFM2.5-Embedding-350M` | `Lfm2BidirectionalModel` (remote code) | 1024 | **512** | Pooling | `query: ` / `document: ` | shortest context of the five |
| `microsoft/harrier-oss-v1-270m` | `Gemma3TextModel` | 640 | **no `sentence_bert_config.json`** | **Normalize** | `web_search_query` on queries, documents bare | falls back to tokenizer `model_max_length` (32768); fp16-overflow risk on T4 |
| `microsoft/harrier-oss-v1-0.6b` | `Qwen3Model` | 1024 | **no `sentence_bert_config.json`** | **Normalize** | `web_search_query` on queries, documents bare | |

### Matryoshka support

| Model | MRL dimensions | Source |
|---|---|---|
| `Alibaba-NLP/gte-multilingual-base` | every multiple of 32 up to 768 | arXiv:2407.19669 §2.2, eq. 3 |
| `lightonai/mDenseOn` | 128, 256, 512, 768 | arXiv:2607.27178 appendix C.3 |
| `LiquidAI/LFM2.5-Embedding-350M` | none documented | model card + release blog, checked 15.09.2026; probed 18.09.2026 |
| `microsoft/harrier-oss-v1-270m` | none documented | model card + Foundry/Bing posts, checked 15.09.2026; probed 18.09.2026 |
| `microsoft/harrier-oss-v1-0.6b` | none documented | model card + Foundry/Bing posts, checked 15.09.2026; probed 18.09.2026 |

So an MRL comparison across both MRL backbones is legitimate at **128, 256 and 512**.
For the other three, truncation is not a zero-overhead DR method. `scripts/probe_mrl.py`
tested this directly rather than inferring it from silent model cards: at the same k,
truncation is compared against PCA and against k *randomly chosen* columns, with the two
documented-MRL models run through the identical measurement as controls. On LFM2.5
truncation is indistinguishable from random columns; on **both harriers it is worse than
random columns**, so it is not a naive baseline there but below one. See
`docs/README-draft.md`.
