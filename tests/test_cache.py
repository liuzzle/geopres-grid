import json

import numpy as np
import pytest

from geopres_grid.cache import GeoPresCache
from geopres_grid.cache import CachedBackbone
from geopres_grid.backbones import get
from geopres_grid.identity import EncodeConfig


class FakeDataset:
    def __init__(self, rows):
        self.rows = rows

    def __iter__(self):
        return iter(self.rows)

    def select(self, indices):
        return FakeDataset([self.rows[index] for index in indices])


class FakeInputs:
    def __init__(self, rows):
        self.dataset = FakeDataset(rows)


class FakeMetadata:
    name = "FakeRetrieval"


class FakeModel:
    def __init__(self):
        self.calls = 0

    @property
    def mteb_model_meta(self):
        return {"name": "fake"}

    def encode(self, inputs, **kwargs):
        self.calls += 1
        return np.array(
            [[len(row["text"]), sum(row["text"].encode())] for row in inputs.dataset],
            dtype=np.float32,
        )

    def similarity(self, embeddings1, embeddings2):
        return np.asarray(embeddings1) @ np.asarray(embeddings2).T

    def similarity_pairwise(self, embeddings1, embeddings2):
        return np.sum(np.asarray(embeddings1) * np.asarray(embeddings2), axis=1)


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


def test_round_trip_preserves_order_and_metadata(tmp_path):
    cache = GeoPresCache(tmp_path, metadata={"encode_config_hash": "abc"})
    items = [{"id": "a", "text": "same"}, {"id": "b", "text": "other"}]
    vectors = np.array([[1, 2], [3, 4]], dtype=np.float32)
    cache.add(items, vectors, prompted_texts=["query: same", "query: other"])
    cache.save()
    cache.close()

    loaded = GeoPresCache(tmp_path)
    loaded.load()
    result, missing = loaded.get_vectors(
        [items[1], items[0]], prompted_texts=["query: other", "query: same"]
    )
    assert not missing.any()
    np.testing.assert_array_equal(result, vectors[[1, 0]])
    assert json.loads((tmp_path / "meta.json").read_text())["encode_config_hash"] == "abc"


def test_prompted_texts_are_distinct_keys(tmp_path):
    cache = GeoPresCache(tmp_path)
    item = {"id": "same", "text": "same"}
    cache.add(
        [item, item],
        np.array([[1, 0], [0, 1]], dtype=np.float32),
        prompted_texts=["query: same", "document: same"],
    )
    result, missing = cache.get_vectors(
        [item, item], prompted_texts=["query: same", "document: same"]
    )
    assert not missing.any()
    np.testing.assert_array_equal(result, [[1, 0], [0, 1]])


def test_missing_items_are_reported_and_added(tmp_path):
    cache = GeoPresCache(tmp_path)
    first = {"id": "a", "text": "a"}
    second = {"id": "b", "text": "b"}
    cache.add([first], np.array([[1, 2]], dtype=np.float32), prompted_texts=["a"])
    _, missing = cache.get_vectors([first, second], prompted_texts=["a", "b"])
    assert missing.tolist() == [False, True]
    cache.add([second], np.array([[3, 4]], dtype=np.float32), prompted_texts=["b"])
    result, missing = cache.get_vectors([first, second], prompted_texts=["a", "b"])
    assert not missing.any()
    np.testing.assert_array_equal(result, [[1, 2], [3, 4]])


def test_conflicting_duplicate_is_rejected(tmp_path):
    cache = GeoPresCache(tmp_path)
    item = {"text": "same"}
    cache.add([item], np.array([[1, 2]], dtype=np.float32))
    with pytest.raises(ValueError, match="conflicting vector"):
        cache.add([item], np.array([[9, 9]], dtype=np.float32))


def test_cached_backbone_encodes_misses_then_warm_starts(tmp_path, monkeypatch):
    model = FakeModel()
    backbone = get("mdenseon")
    wrapper = CachedBackbone(model, backbone, make_encode_config(), tmp_path)
    rows = [{"id": "a", "text": "alpha"}, {"id": "b", "text": "beta"}]

    monkeypatch.setattr(
        "geopres_grid.cache.create_dataloader",
        lambda dataset, **kwargs: FakeInputs(dataset),
    )
    first = wrapper.encode(
        FakeInputs(rows),
        task_metadata=FakeMetadata(),
        hf_split="test",
        hf_subset="default",
        prompt_type="query",
    )
    assert model.calls == 1
    assert wrapper.newly_encoded == 2

    second = wrapper.encode(
        FakeInputs([rows[1], rows[0]]),
        task_metadata=FakeMetadata(),
        hf_split="test",
        hf_subset="default",
        prompt_type="query",
    )
    assert model.calls == 1
    np.testing.assert_array_equal(second, first[[1, 0]])


def test_cached_backbone_load_model_is_noop(tmp_path):
    wrapper = CachedBackbone(FakeModel(), get("mdenseon"), make_encode_config(), tmp_path)

    assert wrapper.load_model() is wrapper


def test_cached_backbone_provides_metadata_when_model_has_none(tmp_path):
    class ModelWithoutMetadata(FakeModel):
        @property
        def mteb_model_meta(self):
            raise AttributeError("metadata is not defined")

    wrapper = CachedBackbone(
        ModelWithoutMetadata(), get("mdenseon"), make_encode_config(), tmp_path
    )

    assert wrapper.mteb_model_meta is not None


def test_cached_backbone_does_not_forward_mteb_only_encode_keywords(tmp_path, monkeypatch):
    class StrictModel(FakeModel):
        def encode(self, inputs, *, batch_size=32):
            return super().encode(inputs, batch_size=batch_size)

    monkeypatch.setattr(
        "geopres_grid.cache.create_dataloader",
        lambda dataset, **kwargs: FakeInputs(dataset),
    )
    wrapper = CachedBackbone(
        StrictModel(), get("mdenseon"), make_encode_config(), tmp_path
    )

    result = wrapper.encode(
        FakeInputs([{"id": "a", "text": "alpha"}]),
        task_metadata=FakeMetadata(),
        hf_split="test",
        hf_subset="default",
        prompt_type="query",
    )

    assert result.shape == (1, 2)


def test_cached_backbone_warm_starts_after_wrapper_reload(tmp_path, monkeypatch):
    backbone = get("mdenseon")
    config = make_encode_config()
    rows = [{"id": "a", "text": "alpha"}, {"id": "b", "text": "beta"}]
    monkeypatch.setattr(
        "geopres_grid.cache.create_dataloader",
        lambda dataset, **kwargs: FakeInputs(dataset),
    )

    first_model = FakeModel()
    first_wrapper = CachedBackbone(first_model, backbone, config, tmp_path)
    first = first_wrapper.encode(
        FakeInputs(rows),
        task_metadata=FakeMetadata(),
        hf_split="test",
        hf_subset="default",
        prompt_type="query",
    )
    first_wrapper.close()

    second_model = FakeModel()
    second_wrapper = CachedBackbone(second_model, backbone, config, tmp_path)
    second = second_wrapper.encode(
        FakeInputs([rows[1], rows[0]]),
        task_metadata=FakeMetadata(),
        hf_split="test",
        hf_subset="default",
        prompt_type="query",
    )

    assert second_model.calls == 0
    assert second_wrapper.newly_encoded == 0
    np.testing.assert_array_equal(second, first[[1, 0]])
    second_wrapper.close()


def test_cached_backbone_separates_query_and_document_blocks(tmp_path, monkeypatch):
    model = FakeModel()
    backbone = get("mdenseon")
    wrapper = CachedBackbone(model, backbone, make_encode_config(), tmp_path)
    row = {"id": "same", "text": "same"}
    monkeypatch.setattr(
        "geopres_grid.cache.create_dataloader",
        lambda dataset, **kwargs: FakeInputs(dataset),
    )

    wrapper.encode(
        FakeInputs([row]),
        task_metadata=FakeMetadata(),
        hf_split="test",
        hf_subset="default",
        prompt_type="query",
    )
    wrapper.encode(
        FakeInputs([row]),
        task_metadata=FakeMetadata(),
        hf_split="test",
        hf_subset="default",
        prompt_type="passage",
    )

    assert model.calls == 2
    query_path = wrapper._cache_path("FakeRetrieval", "test", "default", "query")
    document_path = wrapper._cache_path(
        "FakeRetrieval", "test", "default", "document"
    )
    assert (query_path / "ids.parquet").exists()
    assert (document_path / "ids.parquet").exists()