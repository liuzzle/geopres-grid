# Environments

Two, because no single set of pins runs all five backbones.

| | `transformers` | Backbones | Command |
|---|---|---|---|
| root (`.venv`) | 4.56.0 | mgte, lfm25, harrier-270m, harrier-06b | `uv sync` |
| `envs/transformers5` | 5.x | **mdenseon**, lfm25, harrier-270m, harrier-06b | `uv sync --project envs/transformers5` |

Three of the five run in either. Only mGTE (4.x only) and mDenseOn (5.x only) are
exclusive, and both restrictions were established by loading the model, not by
reading its metadata — see the comment block at the top of
`transformers5/pyproject.toml` for the two tracebacks.

`geopres_grid.backbones` knows about this:

```python
runnable_backbones()        # what this interpreter can actually load
require_runnable(backbone)  # fails with the fix instead of a confusing traceback
```

Only precomputation is environment-sensitive. The embedding cache stores plain
fp32 arrays, so DR, quantization, scoring and every reported number come out of
the root environment regardless of which environment wrote the embeddings.
