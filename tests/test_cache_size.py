from types import SimpleNamespace

from scripts.estimate_cache_size import estimate_task


def test_estimate_task_reports_side_counts_and_fp32_bytes():
    task = SimpleNamespace(
        name="FakeRetrieval",
        corpus={"test": {"d1": {}, "d2": {}}},
        queries={"test": {"q1": "one", "q2": "two", "q3": "three"}},
    )
    backbone = SimpleNamespace(key="fake", native_dim=8)

    result = estimate_task(task, backbone)

    assert result["documents"] == 2
    assert result["queries"] == 3
    assert result["document_bytes"] == 64
    assert result["query_bytes"] == 96
    assert result["total_bytes"] == 160


def test_estimate_task_handles_missing_split():
    task = SimpleNamespace(name="Fake", corpus={"train": {}}, queries={"train": {}})
    backbone = SimpleNamespace(key="fake", native_dim=3)

    result = estimate_task(task, backbone, split="test", itemsize=2)

    assert result["documents"] == 0
    assert result["queries"] == 0
    assert result["total_bytes"] == 0