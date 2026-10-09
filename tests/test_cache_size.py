from types import SimpleNamespace

from scripts.estimate_cache_size import estimate_task


def test_estimate_task_reports_side_counts_and_fp32_bytes():
    task = SimpleNamespace(
        name="FakeRetrieval",
        metadata=SimpleNamespace(name="FakeRetrieval", eval_splits=["test"]),
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
    task = SimpleNamespace(
        name="Fake",
        metadata=SimpleNamespace(name="Fake", eval_splits=["test"]),
        corpus={"train": {}},
        queries={"train": {}},
    )
    backbone = SimpleNamespace(key="fake", native_dim=3)

    result = estimate_task(task, backbone, split="test", itemsize=2)

    assert result["documents"] == 0
    assert result["queries"] == 0
    assert result["total_bytes"] == 0


def test_estimate_task_reads_loaded_rows():
    task = SimpleNamespace(
        name="Loaded",
        metadata=SimpleNamespace(name="Loaded", eval_splits=["test"]),
        corpus={"test": {"d1": {}, "d2": {}}},
        queries={"test": {"q1": "one"}},
    )
    backbone = SimpleNamespace(key="fake", native_dim=2)

    result = estimate_task(task, backbone, split="test", itemsize=4)

    assert result["total_bytes"] == 24

def test_estimate_task_reads_the_v2_layout_of_standard_retrieval_tasks():
    task = SimpleNamespace(
        name="Beir",
        metadata=SimpleNamespace(name="Beir", eval_splits=["test"]),
        dataset={"default": {"test": {"corpus": [{}] * 5, "queries": [{}] * 2}}},
    )
    backbone = SimpleNamespace(key="fake", native_dim=4)

    result = estimate_task(task, backbone)

    assert result["documents"] == 5
    assert result["queries"] == 2
    assert result["total_bytes"] == 7 * 4 * 4


class FakeRows(list):
    def __init__(self, rows):
        super().__init__(rows)
        self.column_names = list(rows[0]) if rows else []


def test_estimate_task_counts_both_sentences_of_an_sts_pair():
    task = SimpleNamespace(
        name="FakeSTS",
        metadata=SimpleNamespace(name="FakeSTS", eval_splits=["test"], type="STS"),
        dataset={"default": {"test": FakeRows([{"sentence1": "a", "sentence2": "b", "score": 1.0}] * 3)}},
    )
    backbone = SimpleNamespace(key="fake", native_dim=4)

    result = estimate_task(task, backbone)

    assert (result["documents"], result["queries"]) == (0, 6)
    assert result["total_bytes"] == 6 * 4 * 4


def test_estimate_task_counts_legacy_clustering_sentences_across_subsets():
    rows = FakeRows([{"sentences": ["a", "b", "c"], "labels": [0, 1, 0]}, {"sentences": ["d"], "labels": [1]}])
    task = SimpleNamespace(
        name="FakeClustering",
        metadata=SimpleNamespace(name="FakeClustering", eval_splits=["test"], type="Clustering"),
        dataset={"en": {"test": rows}, "de": {"test": rows}},
    )
    backbone = SimpleNamespace(key="fake", native_dim=1)

    assert estimate_task(task, backbone)["queries"] == 8


def test_totals_count_a_shared_corpus_once():
    from scripts.estimate_cache_size import totals

    backbone = SimpleNamespace(key="fake", native_dim=2)

    def task(name, documents):
        return SimpleNamespace(
            name=name,
            metadata=SimpleNamespace(name=name, eval_splits=["test"]),
            dataset={"default": {"test": {"corpus": [{}] * documents, "queries": [{}]}}},
        )

    rows = [estimate_task(task(name, n), backbone) for name, n in (("FEVER", 10), ("ClimateFEVER", 12), ("NQ", 5))]
    result = totals(rows)

    assert [row["corpus"] for row in rows] == ["corpus-fever-wikipedia", "corpus-fever-wikipedia", "NQ"]
    assert result["fake"]["document_bytes"] == (12 + 5) * 2 * 4
    assert result["fake"]["query_bytes"] == 3 * 2 * 4
    assert result["all"] == result["fake"]


def test_estimate_task_reads_a_monolingual_non_retrieval_task():
    """STSBenchmark loads as `dataset[split]`, with no subset level."""
    task = SimpleNamespace(
        name="FakeSTS",
        metadata=SimpleNamespace(name="FakeSTS", eval_splits=["test"], type="STS"),
        dataset={
            "train": FakeRows([{"sentence1": "a", "sentence2": "b", "score": 1.0}] * 7),
            "test": FakeRows([{"sentence1": "a", "sentence2": "b", "score": 1.0}] * 3),
        },
    )
    backbone = SimpleNamespace(key="fake", native_dim=4)

    assert estimate_task(task, backbone)["queries"] == 6
