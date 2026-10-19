# Evaluating mDenseOn with semantic-search-eval

This guide assumes `docs.csv` and `queries.csv` already exist (see `prepare_eval_data.py`).

## 1. Install the repo

```bash
git clone https://github.com/machinelearningZH/semantic-search-eval.git
cd semantic-search-eval

pip3 install uv
uv venv
source .venv/bin/activate
uv sync
```

## 2. Put the data in place

```text
semantic-search-eval/
  data/
    docs.csv
    queries.csv
```

## 3. Create the config

mDenseOn uses a `query` prompt and a `document` prompt (from the model card). The repo's config has options for built-in prompts, so no code change is needed. The jina-v5 entry in the example config uses the same setup.

Save as `configs/mdenseon.yaml`:

```yaml
folder: results/mdenseon

name: mdenseon_test
docs: data/docs.csv
queries: data/queries.csv

metrics:
  - accuracy@1
  - accuracy@5
  - accuracy@10
  - ndcg@10
  - latency

# mDenseOn supports up to 8192 tokens
max-len: 8192

models:
  huggingface:
    mdenseon:
      model: lightonai/mDenseOn
      set_builtin_query_prompt: query
      set_builtin_passage_prompt: document
```

## 4. Run

```bash
python semsearcheval/evaluate.py -c configs/mdenseon.yaml
```

Results (a CSV with the metrics and plots) are saved in `results/mdenseon`.

## 5. Optional check: prompts in plain Python

To confirm the prompt names work on their own, outside the repo:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("lightonai/mDenseOn")

query_emb = model.encode(["Wie hoch ist die Steuer?"], prompt_name="query")
doc_emb = model.encode(["Die Steuer beträgt ..."], prompt_name="document")

print(model.similarity(query_emb, doc_emb))
```

## Note

If you want to examine the evaluation on a specific subset (e.g. OCR quality >= 0.95), you can filter the dataset before running it and run it as a separate evaluation.
