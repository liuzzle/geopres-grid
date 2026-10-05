import mteb
import numpy as np
import pytest
from datasets import Dataset
from mteb._create_dataloaders import create_dataloader
from mteb.types import PromptType

from geopres_grid.backbones import get
from geopres_grid.cache import CachedBackbone
from geopres_grid.identity import EncodeConfig
from geopres_grid.precompute import leaf_tasks, precompute_task_cache

TASK_METADATA = mteb.get_tasks(tasks=["NanoArguAnaRetrieval"])[0].metadata


class FakeTask:
    """A loaded retrieval task in MTEB's v2 layout, with real task metadata."""

    def __init__(self, corpus, queries):
        self.metadata = TASK_METADATA
        self.eval_splits = ["train"]
        self.hf_subsets = ["default"]
        self.converted = False
        self.dataset = {
            "default": {
                "train": {
                    "corpus": Dataset.from_list(corpus),
                    "queries": Dataset.from_list(queries),
                }
            }
        }

    def convert_v1_dataset_format_to_v2(self, num_proc=None):
        self.converted = True


class FakeModel:
    def __init__(self):
        self.seen = []

    def encode(self, texts, *, prompt=None, batch_size=32, **kwargs):
        texts = [(prompt or "") + text for text in texts]
        self.seen.extend(texts)
        return np.asarray([[len(text), sum(text.encode())] for text in texts], dtype=np.float32)


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


CORPUS = [
    {"id": "d1", "title": "Cats", "text": "Cats purr. "},
    {"id": "d2", "title": "", "text": "Dogs bark."},
]
QUERIES = [{"id": "q1", "text": "why do cats purr"}]


def test_precompute_task_cache_populates_query_and_document_blocks(tmp_path):
    model = FakeModel()
    backbone = get("mdenseon")
    config = make_encode_config()
    task = FakeTask(CORPUS, QUERIES)

    result = precompute_task_cache(
        model=model,
        backbone=backbone,
        encode_config=config,
        task=task,
        cache_root=tmp_path,
        batch_size=8,
    )

    assert task.converted
    assert result["documents"] == 2
    assert result["queries"] == 1
    assert result["splits"] == ["train"]
    root = tmp_path / f"{backbone.slug}@{backbone.revision}" / config.hash / TASK_METADATA.name / "train" / "default"
    assert (root / "query" / "ids.parquet").exists()
    assert (root / "document" / "ids.parquet").exists()


def test_precompute_encodes_the_text_mteb_sends(tmp_path):
    """Title joined, whitespace stripped: the keys must match MTEB's evaluation path."""
    model = FakeModel()
    precompute_task_cache(
        model=model,
        backbone=get("mdenseon"),
        encode_config=make_encode_config(),
        task=FakeTask(CORPUS, QUERIES),
        cache_root=tmp_path,
    )

    assert model.seen == [
        "document: Cats Cats purr.",
        "document: Dogs bark.",
        "query: why do cats purr",
    ]


def test_evaluation_after_precompute_encodes_nothing(tmp_path):
    backbone = get("mdenseon")
    config = make_encode_config()
    precompute_task_cache(
        model=FakeModel(),
        backbone=backbone,
        encode_config=config,
        task=FakeTask(CORPUS, QUERIES),
        cache_root=tmp_path,
    )

    # What MTEB's retrieval evaluator hands the encoder for the same corpus.
    wrapper = CachedBackbone(FakeModel(), backbone, config, tmp_path)
    for prompt_type, rows in ((PromptType.document, CORPUS), (PromptType.query, QUERIES)):
        loader = create_dataloader(
            Dataset.from_list(rows), task_metadata=TASK_METADATA, prompt_type=prompt_type
        )
        wrapper.encode(
            loader,
            task_metadata=TASK_METADATA,
            hf_split="train",
            hf_subset="default",
            prompt_type=prompt_type,
        )
    assert wrapper.newly_encoded == 0
    wrapper.close()


def test_precompute_refuses_to_cache_nothing(tmp_path):
    with pytest.raises(ValueError, match="nothing was cached"):
        precompute_task_cache(
            model=FakeModel(),
            backbone=get("mdenseon"),
            encode_config=make_encode_config(),
            task=FakeTask([], QUERIES),
            cache_root=tmp_path,
        )


def test_leaf_tasks_expands_aggregates():
    cqadupstack = mteb.get_tasks(tasks=["CQADupstackRetrieval"])[0]
    nano = mteb.get_tasks(tasks=["NanoArguAnaRetrieval"])[0]

    leaves = leaf_tasks(cqadupstack)

    assert len(leaves) == 12
    assert all(leaf.metadata.name.startswith("CQADupstack") for leaf in leaves)
    assert leaf_tasks(nano) == [nano]
