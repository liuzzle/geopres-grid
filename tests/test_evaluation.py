from geopres_grid.evaluation import NANOBEIR_TASKS
from geopres_grid.evaluation import task_names


def test_tier_zero_is_small_and_mixed():
    assert task_names("tier0") == ("NanoArguAnaRetrieval", "STSBenchmark")


def test_tier_one_is_the_pinned_nanobeir_set():
    assert task_names("tier1") == NANOBEIR_TASKS
    assert len(NANOBEIR_TASKS) == 13
    assert len(set(NANOBEIR_TASKS)) == 13


def test_tier_two_is_supplementary_retrieval_only():
    assert task_names("tier2") == (
        "ArguAna",
        "QuoraRetrieval",
        "HotpotQA",
        "NQ",
        "MSMARCO",
        "DBPedia",
    )