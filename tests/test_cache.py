import json

import numpy as np
import pytest

from geopres_grid.cache import GeoPresCache


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