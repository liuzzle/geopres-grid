# Environments

Three, because no single set of pins runs all five backbones.

| | `transformers` | `sentence-transformers` | Backbones | Command |
|---|---|---|---|---|
| root (`.venv`) | 4.56.0 | 5.x | mgte, harrier-270m, harrier-06b | `uv sync` |
| `envs/transformers5` | 5.x | 5.x | **mdenseon**, harrier-270m, harrier-06b | `uv sync --project envs/transformers5` |
| `envs/sentence_transformers6` | ≥ 5.19 | **6.x** | **embeddinggemma-2** | `uv sync --project envs/sentence_transformers6` |

The two harriers run in either of the first two. mGTE (transformers 4.x only),
mDenseOn (5.x only) and EmbeddingGemma-2 (sentence-transformers 6.x only) are
exclusive, and all three restrictions were established by loading the model, not
by reading its metadata. The tracebacks are in the comment block at the top of each
`pyproject.toml`.

The third environment exists so that the supervisor's `<6.0` ceiling on
sentence-transformers can hold for every other model: only EmbeddingGemma-2 needs
6.x, so only it gets it. It also carries `pillow` and `torchvision`, which the
checkpoint's processor imports even for text-only input.

`geopres_grid.backbones` knows about this:

```python
runnable_backbones()        # what this interpreter can actually load
require_runnable(backbone)  # fails with the fix instead of a confusing traceback
```

Only precomputation is environment-sensitive. The embedding cache stores plain
fp32 arrays and records its encode config in `meta.json`, so `scripts/evaluate.py`
runs in the root environment for every backbone without loading a model. DR,
quantization, scoring and every reported number come out of the root environment,
whichever environment wrote the embeddings.
