import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from geopres_grid.cache import GeoPresCache
from geopres_grid.cache import CachedBackbone
from geopres_grid.cache import encode_configs_on_disk
from geopres_grid.cache import resolve_encode_config
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
    type = "Retrieval"


class FakeSTSMetadata:
    name = "FakeSTS"
    type = "STS"


class FakeClassificationMetadata:
    name = "FakeClassification"
    type = "Classification"


class FakeModel:
    """Stands in for `SentenceTransformer.encode`: a list of texts plus `prompt`."""

    def __init__(self):
        self.calls = 0
        self.seen = []
        self.prompts = []

    def encode(self, texts, *, prompt=None, batch_size=32, **kwargs):
        self.calls += 1
        self.prompts.append(prompt)
        texts = [(prompt or "") + text for text in texts]
        self.seen.extend(texts)
        return np.array(
            [[len(text), sum(text.encode())] for text in texts],
            dtype=np.float32,
        )

    def similarity(self, embeddings1, embeddings2):
        return np.asarray(embeddings1) @ np.asarray(embeddings2).T

    def similarity_pairwise(self, embeddings1, embeddings2):
        return np.sum(np.asarray(embeddings1) * np.asarray(embeddings2), axis=1)


def make_encode_config(**overrides):
    return EncodeConfig(
        model_id="fake/model",
        revision="a" * 40,
        max_seq_length=512,
        dtype="float32",
        prompts=(("query", "query: "), ("document", "document: ")),
        sentence_transformers_version="5.7",
        transformers_version="4.56",
        **overrides,
    )


def harrier_encode_config():
    return make_encode_config(task_prompt_names=(("STS", "sts_query"),))


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
        def encode(self, texts, *, prompt, batch_size, convert_to_numpy, show_progress_bar):
            return super().encode(texts, prompt=prompt, batch_size=batch_size)

    wrapper = CachedBackbone(
        StrictModel(), get("mdenseon"), make_encode_config(), tmp_path
    )

    result = wrapper.encode(
        FakeInputs([{"id": "a", "text": "alpha"}]),
        task_metadata=FakeMetadata(),
        hf_split="test",
        hf_subset="default",
        prompt_type="query",
        num_proc=1,
    )

    assert result.shape == (1, 2)


def test_cached_backbone_warm_starts_after_wrapper_reload(tmp_path, monkeypatch):
    backbone = get("mdenseon")
    config = make_encode_config()
    rows = [{"id": "a", "text": "alpha"}, {"id": "b", "text": "beta"}]

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

def test_cached_backbone_encodes_the_prepared_text_under_its_own_key(tmp_path):
    model = FakeModel()
    backbone = get("mdenseon")
    wrapper = CachedBackbone(model, backbone, make_encode_config(), tmp_path)
    # MTEB has already prepared the row: title and body joined into `text`.
    prepared = {"id": "d1", "text": "Cats Cats purr.", "body": "Cats purr.", "title": "Cats"}

    wrapper.encode(
        FakeInputs([prepared]),
        task_metadata=FakeMetadata(),
        hf_split="test",
        hf_subset="default",
        prompt_type="document",
    )

    assert model.seen == ["document: Cats Cats purr."]
    cache = GeoPresCache(wrapper._cache_path("FakeRetrieval", "test", "default", "document"))
    cache.load()
    _, missing = cache.get_vectors([prepared], prompted_texts=model.seen)
    assert not missing.any()


def test_cached_backbone_passes_an_empty_prompt_on_a_bare_side(tmp_path):
    model = FakeModel()
    wrapper = CachedBackbone(model, get("harrier-270m"), harrier_encode_config(), tmp_path)

    wrapper.encode(
        FakeInputs([{"id": "d1", "text": "body"}]),
        task_metadata=FakeMetadata(),
        hf_split="test",
        hf_subset="default",
        prompt_type="document",
    )

    # `prompt=""`, not None: None would let a default_prompt_name apply.
    assert model.prompts == [""]


def test_run_model_meta_gives_each_encode_config_its_own_result_slot(tmp_path):
    other = dataclasses.replace(make_encode_config(), max_seq_length=256)
    first = CachedBackbone(FakeModel(), get("mdenseon"), make_encode_config(), tmp_path)
    second = CachedBackbone(FakeModel(), get("mdenseon"), other, tmp_path)

    assert first.mteb_model_meta.name == "geopres-grid/mdenseon"
    assert first.mteb_model_meta.revision == make_encode_config().hash
    assert first.mteb_model_meta.revision != second.mteb_model_meta.revision


def encode_symmetric(wrapper, metadata, text="a sentence"):
    return wrapper.encode(
        FakeInputs([{"id": "s1", "text": text}]),
        task_metadata=metadata,
        hf_split="test",
        hf_subset="default",
        prompt_type=None,
    )


def test_a_task_without_prompt_type_gets_the_task_types_prompt(tmp_path):
    """harrier's STS pairs take `sts_query`, not the retrieval instruction."""
    harrier = get("harrier-270m")
    model = FakeModel()
    wrapper = CachedBackbone(model, harrier, harrier_encode_config(), tmp_path)

    encode_symmetric(wrapper, FakeSTSMetadata())

    assert model.prompts == [harrier.prompts["sts_query"]]
    # The cache side is still `query`: one side per symmetric task, as before.
    assert (wrapper.task_directory(FakeSTSMetadata.name) / "test" / "default" / "query").exists()


def test_a_task_type_without_a_dedicated_prompt_gets_the_query_prompt(tmp_path):
    """harrier declares no classification prompt, so its STS one must not leak in."""
    harrier = get("harrier-270m")
    model = FakeModel()
    wrapper = CachedBackbone(model, harrier, harrier_encode_config(), tmp_path)

    encode_symmetric(wrapper, FakeClassificationMetadata())

    assert model.prompts == [harrier.prompt_for("query")]


def test_a_backbone_without_task_prompts_uses_its_query_prompt(tmp_path):
    model = FakeModel()
    wrapper = CachedBackbone(model, get("mdenseon"), make_encode_config(), tmp_path)

    encode_symmetric(wrapper, FakeSTSMetadata())

    assert model.prompts == [get("mdenseon").prompt_for("query")]


def test_encode_config_must_name_the_backbones_task_prompts(tmp_path):
    """Otherwise the STS prompt changes while the cache directory and the result
    slot -- both named by the encode hash -- stay where the old prompt's are."""
    with pytest.raises(ValueError, match="task prompts"):
        CachedBackbone(FakeModel(), get("harrier-270m"), make_encode_config(), tmp_path)


class NoisyModel(FakeModel):
    """Like a real model on padded batches: the same text encodes to vectors that
    differ in the last bits from one call position to the next."""

    def encode(self, texts, **kwargs):
        values = super().encode(texts, **kwargs)
        return values + np.arange(len(texts), dtype=np.float32)[:, None] * 1e-6


def test_a_repeated_text_is_encoded_once(tmp_path):
    """NanoArguAna repeats documents; encoding each copy failed in `GeoPresCache.add`."""
    model = NoisyModel()
    wrapper = CachedBackbone(model, get("mdenseon"), make_encode_config(), tmp_path)
    rows = [{"id": "d1", "text": "same"}, {"id": "d2", "text": "other"}, {"id": "d3", "text": "same"}]

    result = wrapper.encode(
        FakeInputs(rows), task_metadata=FakeMetadata(), hf_split="test", hf_subset="default",
        prompt_type="document",
    )

    assert model.seen == ["document: same", "document: other"]
    assert wrapper.newly_encoded == 2
    np.testing.assert_array_equal(result[0], result[2])


# --- a warm cache with no model -------------------------------------------------


def test_a_warm_cache_is_served_without_a_model(tmp_path):
    rows = [{"id": "d1", "text": "one"}, {"id": "d2", "text": "two"}]
    loaded = CachedBackbone(FakeModel(), get("mdenseon"), make_encode_config(), tmp_path)
    expected = loaded.encode(
        FakeInputs(rows), task_metadata=FakeMetadata(), hf_split="test", hf_subset="default",
        prompt_type="document",
    )
    loaded.close()

    modelless = CachedBackbone(None, get("mdenseon"), make_encode_config(), tmp_path)
    served = modelless.encode(
        FakeInputs(rows), task_metadata=FakeMetadata(), hf_split="test", hf_subset="default",
        prompt_type="document",
    )

    np.testing.assert_array_equal(served, expected)
    assert modelless.newly_encoded == 0


def test_a_miss_without_a_model_is_an_error_not_an_encode(tmp_path):
    modelless = CachedBackbone(None, get("mdenseon"), make_encode_config(), tmp_path)
    with pytest.raises(RuntimeError, match="no model is loaded"):
        modelless.encode(
            FakeInputs([{"id": "d1", "text": "never cached"}]),
            task_metadata=FakeMetadata(), hf_split="test", hf_subset="default",
            prompt_type="document",
        )


def test_similarity_is_cosine_whether_or_not_a_model_is_loaded(tmp_path):
    """FakeModel.similarity is a raw dot product; the wrapper must not use it."""
    a = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
    b = np.array([[6.0, 8.0], [0.0, 2.0]], dtype=np.float32)
    for model in (FakeModel(), None):
        wrapper = CachedBackbone(model, get("mdenseon"), make_encode_config(), tmp_path)
        np.testing.assert_allclose(
            np.asarray(wrapper.similarity(a, b)), [[1.0, 0.8], [0.6, 0.0]], atol=1e-6
        )
        np.testing.assert_allclose(np.asarray(wrapper.similarity_pairwise(a, b)), [1.0, 0.0], atol=1e-6)


def test_the_encode_config_is_read_back_from_the_cache(tmp_path):
    config = make_encode_config()
    wrapper = CachedBackbone(FakeModel(), get("mdenseon"), config, tmp_path)
    encode_symmetric(wrapper, FakeSTSMetadata())
    wrapper.close()

    on_disk = encode_configs_on_disk(tmp_path, get("mdenseon"))

    assert list(on_disk) == [config.hash]
    assert on_disk[config.hash] == config
    assert resolve_encode_config(tmp_path, get("mdenseon")) == config
    assert resolve_encode_config(tmp_path, get("mdenseon"), max_seq_length=512) == config
    with pytest.raises(LookupError, match="0 cached encode configs"):
        resolve_encode_config(tmp_path, get("mdenseon"), max_seq_length=8192)
    with pytest.raises(LookupError, match="no cached encode hash"):
        resolve_encode_config(tmp_path, get("mdenseon"), encode_hash="0" * 16)


def test_two_cached_configs_must_be_chosen_between(tmp_path):
    """A harrier precomputed in both transformers environments has two hashes."""
    first = harrier_encode_config()
    second = dataclasses.replace(first, transformers_version="5.17")
    for config in (first, second):
        wrapper = CachedBackbone(FakeModel(), get("harrier-270m"), config, tmp_path)
        encode_symmetric(wrapper, FakeSTSMetadata())
        wrapper.close()

    with pytest.raises(LookupError, match="--encode-hash"):
        resolve_encode_config(tmp_path, get("harrier-270m"))
    assert resolve_encode_config(tmp_path, get("harrier-270m"), encode_hash=second.hash) == second


class FakeFEVERMetadata:
    name = "FEVER"
    type = "Retrieval"


class FakeClimateFEVERMetadata:
    name = "ClimateFEVER"
    type = "Retrieval"


def encode_retrieval(wrapper, metadata, documents, queries):
    for prompt_type, rows in (("document", documents), ("query", queries)):
        if rows:
            wrapper.encode(
                FakeInputs(rows),
                task_metadata=metadata,
                hf_split="test",
                hf_subset="default",
                prompt_type=prompt_type,
            )


FEVER_PASSAGES = [{"id": f"d{i}", "text": f"passage {i}"} for i in range(3)]


def test_climate_fever_starts_from_a_verified_copy_of_fever(tmp_path):
    """ClimateFEVER reuses FEVER's 5.4M-passage corpus: copied, not re-encoded."""
    from geopres_grid.cache import copy_document_blocks
    from geopres_grid.evaluation import cached_blocks

    wrapper = CachedBackbone(FakeModel(), get("mdenseon"), make_encode_config(), tmp_path)
    encode_retrieval(wrapper, FakeFEVERMetadata(), FEVER_PASSAGES, [{"id": "q", "text": "FEVER"}])
    wrapper.close()
    root = wrapper.encode_directory

    copied = copy_document_blocks(root, "FEVER", "ClimateFEVER", splits=["test"], subsets=["default"])

    target = root / "ClimateFEVER" / "test" / "default" / "document"
    assert copied == [target]
    for name in ("embeddings.npy", "ids.parquet", "meta.json"):
        assert (target / name).read_bytes() == (root / "FEVER" / "test" / "default" / "document" / name).read_bytes()
    assert not list(target.parent.glob(".document-copy-*"))

    climate = CachedBackbone(FakeModel(), get("mdenseon"), make_encode_config(), tmp_path)
    extra = {"id": "d3", "text": "only in ClimateFEVER"}
    encode_retrieval(climate, FakeClimateFEVERMetadata(), FEVER_PASSAGES + [extra], [{"id": "q", "text": "ClimateFEVER"}])
    climate.close()

    assert climate.newly_encoded == 1 + 1  # the extra passage and the query
    assert [len(block) for block in cached_blocks(climate.task_directory("ClimateFEVER"))["document"]] == [4]
    assert [len(block) for block in cached_blocks(climate.task_directory("FEVER"))["document"]] == [3]


def test_corpus_copy_needs_a_finished_source_and_a_missing_target(tmp_path):
    from geopres_grid.cache import copy_document_blocks

    wrapper = CachedBackbone(FakeModel(), get("mdenseon"), make_encode_config(), tmp_path)
    root = wrapper.encode_directory
    copy = lambda: copy_document_blocks(root, "FEVER", "ClimateFEVER", splits=["test"], subsets=["default"])

    assert copy() == []  # nothing cached
    encode_retrieval(wrapper, FakeFEVERMetadata(), FEVER_PASSAGES, [])
    assert copy() == []  # documents written, queries not: FEVER may still be running
    encode_retrieval(wrapper, FakeClimateFEVERMetadata(), [{"id": "x", "text": "own"}], [])
    encode_retrieval(wrapper, FakeFEVERMetadata(), [], [{"id": "q", "text": "FEVER"}])
    wrapper.close()
    assert copy() == []  # ClimateFEVER already has a block: never overwritten
    assert len(pd.read_parquet(root / "ClimateFEVER" / "test" / "default" / "document" / "ids.parquet")) == 1


def test_corpus_copy_sources_are_precomputed_before_their_targets():
    """Within one tier-2 job, FEVER must be finished when ClimateFEVER starts."""
    from geopres_grid.cache import CORPUS_COPIES
    from geopres_grid.evaluation import BEIR_TASKS

    for target, source in CORPUS_COPIES.items():
        assert BEIR_TASKS.index(source) < BEIR_TASKS.index(target)


def test_a_block_with_fewer_rows_than_its_index_is_refused(tmp_path):
    cache = GeoPresCache(tmp_path)
    cache.add([{"id": "a", "text": "a"}, {"id": "b", "text": "b"}], np.ones((2, 3)))
    cache.save()
    cache.close()
    np.save(tmp_path / "embeddings.npy", np.ones((1, 3), dtype=np.float32))

    with pytest.raises(ValueError, match="two writers"):
        GeoPresCache(tmp_path).load()


def items(*texts):
    return [{"id": text, "text": text} for text in texts]


def rows(start, count, dim=3):
    return np.arange(start * dim, (start + count) * dim, dtype=np.float32).reshape(count, dim)


def test_add_appends_in_place_and_never_rewrites_existing_rows(tmp_path):
    cache = GeoPresCache(tmp_path)
    cache.add(items("a", "b"), rows(0, 2))
    cache.save()
    with open(tmp_path / "embeddings.npy", "r+b") as handle:  # mark row 0 on disk
        handle.seek(-2 * 3 * 4, 2)
        handle.write(np.full(3, -1, dtype=np.float32).tobytes())

    cache.add(items("c"), rows(2, 1))
    cache.add(items("d", "e"), rows(3, 2))
    cache.save()

    stored = np.load(tmp_path / "embeddings.npy")
    assert stored.shape == (5, 3)
    np.testing.assert_array_equal(stored[0], np.full(3, -1))  # survived: not rewritten
    np.testing.assert_array_equal(stored[1:], rows(1, 4))
    reloaded = GeoPresCache(tmp_path)
    vectors, missing = reloaded.get_vectors(items("e", "c"), prompted_texts=["e", "c"])
    assert not missing.any()
    np.testing.assert_array_equal(vectors, np.vstack([rows(4, 1), rows(2, 1)]))


def test_rows_appended_but_not_committed_are_ignored_then_overwritten(tmp_path):
    """A job killed after `add` and before `save`: the index is the commit."""
    cache = GeoPresCache(tmp_path)
    cache.add(items("a", "b"), rows(0, 2))
    cache.save()
    cache.add(items("c", "d"), rows(2, 2))  # killed here: no save
    assert np.load(tmp_path / "embeddings.npy", mmap_mode="r").shape == (4, 3)

    resumed = GeoPresCache(tmp_path)
    assert resumed.missing(["a", "b", "c", "d"]).tolist() == [False, False, True, True]
    resumed.add(items("d"), rows(10, 1))
    resumed.save()

    stored = np.load(tmp_path / "embeddings.npy")
    np.testing.assert_array_equal(stored, np.vstack([rows(0, 2), rows(10, 1)]))
    vectors, missing = GeoPresCache(tmp_path).get_vectors(items("d"), prompted_texts=["d"])
    assert not missing.any()
    np.testing.assert_array_equal(vectors, rows(10, 1))


def test_an_array_written_before_any_index_is_overwritten(tmp_path):
    """Killed after the first chunk's array, before its index existed."""
    cache = GeoPresCache(tmp_path)
    cache.add(items("a", "b"), rows(0, 2))  # no save

    resumed = GeoPresCache(tmp_path)
    assert resumed.missing(["a"]).tolist() == [True]
    resumed.add(items("x"), rows(7, 1))
    resumed.save()

    np.testing.assert_array_equal(np.load(tmp_path / "embeddings.npy"), rows(7, 1))


def test_a_second_writer_on_one_block_is_refused(tmp_path):
    first = GeoPresCache(tmp_path)
    first.add(items("a"), rows(0, 1))
    first.save()
    second = GeoPresCache(tmp_path)
    second.load()
    first.add(items("b"), rows(1, 1))
    first.save()

    with pytest.raises(ValueError, match="another process"):
        second.add(items("c"), rows(2, 1))
