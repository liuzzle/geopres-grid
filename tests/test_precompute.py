import numpy as np

from geopres_grid.backbones import get
from geopres_grid.identity import EncodeConfig
from geopres_grid.precompute import precompute_task_cache


class FakeDataset:
    def __init__(self, rows):
        self.rows = list(rows)

    def __iter__(self):
        return iter(self.rows)

    def select(self, indices):
        return FakeDataset([self.rows[index] for index in indices])


class FakeInputs:
    def __init__(self, rows):
        self.dataset = FakeDataset(rows)


class FakeTask:
    def __init__(self):
        self.name = "FakeRetrieval"
        self.corpus = {"test": {"doc-1": {"text": "alpha"}, "doc-2": {"text": "beta"}}}
        self.queries = {"test": {"q-1": "alpha", "q-2": "beta"}}
        self.metadata = type("Meta", (), {"eval_splits": ["test"]})()


class FakeModel:
    def __init__(self):
        self.calls = 0

    @property
    def mteb_model_meta(self):
        return {"name": "fake"}

    def encode(self, inputs, **kwargs):
        self.calls += 1
        rows = list(inputs.dataset)
        return np.asarray([[len(row["text"]), sum(row["text"].encode())] for row in rows], dtype=np.float32)


def make_encode_config():
    return EncodeConfig(
        model_id="fake/model",
        revision="a" * 40,
        max_seq_length=512,
        dtype="float32",
        prompts=(("query", "query: "), ("document", "document: ")),
        sentence_transformers_version="5.7",
        transformers_version="4.56",
    )


def test_precompute_task_cache_populates_query_and_document_blocks(tmp_path):
    model = FakeModel()
    backbone = get("mdenseon")
    config = make_encode_config()
    task = FakeTask()

    result = precompute_task_cache(
        model=model,
        backbone=backbone,
        encode_config=config,
        task=task,
        cache_root=tmp_path,
        hf_split="test",
        hf_subset="default",
        batch_size=8,
    )

    assert result["documents"] == 2
    assert result["queries"] == 2
    query_path = tmp_path / f"{backbone.slug}@{backbone.revision}" / config.hash / task.name / "test" / "default" / "query"
    doc_path = tmp_path / f"{backbone.slug}@{backbone.revision}" / config.hash / task.name / "test" / "default" / "document"
    assert (query_path / "ids.parquet").exists()
    assert (doc_path / "ids.parquet").exists()
