# geopres-grid

Combining Dimensionality Reduction and Compression for Efficient Indexing.

A configuration grid over `(backbone × dimensionality reduction × quantization)`,
evaluated on MTEB. Built on top of
[GeoPres](https://openreview.net/forum?id=Xc8ulFlMrl); the projection training code
is carried over from that repo, everything about caching, quantization and the
evaluation grid is new here.


### Backbone table, read from the HF configs

| Model | Architecture | d | `max_seq_length` | Last ST module | Prompts | Notes |
|---|---|---|---|---|---|---|
| `Alibaba-NLP/gte-multilingual-base` | `NewModel` (remote code) | 768 | 8192 | **Normalize** | none | needs `trust_remote_code`; the reason transformers stays on the 4.x line |
| `lightonai/mDenseOn` | `ModernBertModel` | 768 | 8192 | Pooling | `query: ` / `document: ` | new-style `modules.json` ⇒ **requires ST ≥ 5.4**; **does not load** — tokenizer needs transformers ≥ 5.0 |
| `LiquidAI/LFM2.5-Embedding-350M` | `Lfm2BidirectionalModel` (remote code) | 1024 | **512** | Pooling | `query: ` / `document: ` | shortest context of the five |
| `microsoft/harrier-oss-v1-270m` | `Gemma3TextModel` | 640 | **no `sentence_bert_config.json`** | **Normalize** | instruct-style query prompts | falls back to tokenizer `model_max_length` (32768); fp16-overflow risk on T4 |
| `microsoft/harrier-oss-v1-0.6b` | `Qwen3Model` | 1024 | **no `sentence_bert_config.json`** | **Normalize** | instruct-style query prompts | |