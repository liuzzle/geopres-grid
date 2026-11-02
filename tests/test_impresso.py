import csv
from pathlib import Path

import mteb
import numpy as np
import pytest

from geopres_grid import config
from geopres_grid.backbones import get
from geopres_grid.cache import CachedBackbone
from geopres_grid.evaluation import (
    IMPRESSO_TASKS,
    TIER_TASKS,
    PostProcessedBackbone,
    evaluate_cell,
    load_tasks,
    task_names,
    with_query_expansions,
)
from geopres_grid.expansion import ExpansionConfig, expanded_task_name
from geopres_grid.identity import EncodeConfig, PostProcConfig
from geopres_grid.impresso import TASKS, load_split
from geopres_grid.precompute import precompute_task_cache

WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet"]
LETTERS = "abcdefghijklmnopqrstuvwxyz"


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def impresso_dir(tmp_path, monkeypatch):
    """The sent layout, shrunk: ten documents per language, query n asks for document n.

    Document ids are shuffled against row order so a loader that matched by row
    position instead of by `idx` would score zero.
    """
    root = tmp_path / "impresso"
    for form in ("crosslingual", "multilingual"):
        for language in ("DE", "FR"):
            order = list(reversed(range(10)))
            write_csv(
                root / form / language / "docs.csv",
                [{"id": str(i), "text": f"{WORDS[i]} {WORDS[i]} story text", "ocrQuality": "0.9"} for i in order],
            )
            write_csv(
                root / form / language / "queries.csv",
                [{"search_query": WORDS[i], "idx": str(i), "ocrQuality": "0.9"} for i in range(10)],
            )
    monkeypatch.setattr(config, "IMPRESSO_PATH", str(root))
    return root


class LetterModel:
    """Bag-of-letters vectors: text sharing letters with another ranks near it."""

    def encode(self, texts, *, prompt=None, batch_size=32, **kwargs):
        vectors = [[text.lower().count(c) for c in LETTERS] for text in texts]
        return np.asarray(vectors, dtype=np.float32)


def make_encode_config():
    return EncodeConfig(
        model_id="fake/model",
        revision="a" * 40,
        max_seq_length=512,
        dtype="float32",
        prompts=(("query", ""), ("document", "")),
        sentence_transformers_version="5.7",
        transformers_version="4.56",
    )


def test_the_tier_names_match_the_task_classes():
    assert tuple(TASKS) == IMPRESSO_TASKS == task_names("impresso") == TIER_TASKS["impresso"]


def test_load_tasks_builds_the_custom_tasks(impresso_dir):
    tasks = load_tasks(IMPRESSO_TASKS)

    assert [task.metadata.name for task in tasks] == list(IMPRESSO_TASKS)
    assert all(task.eval_splits == ["test"] and task.hf_subsets == ["default"] for task in tasks)
    assert all(not task.metadata.is_public for task in tasks)
    assert tasks[0].metadata.eval_langs == ["eng-Latn", "deu-Latn"]
    assert tasks[1].metadata.eval_langs == ["deu-Latn"]


def test_qrels_follow_idx_not_row_order(impresso_dir):
    split = load_split(impresso_dir / "multilingual" / "DE")

    assert len(split["corpus"]) == len(split["queries"]) == 10
    document_text = {row["id"]: row["text"] for row in split["corpus"]}
    for query in split["queries"]:
        (relevant,) = split["relevant_docs"][query["id"]]
        assert query["text"] in document_text[relevant]
    assert not set(split["relevant_docs"]) & set(document_text)


def test_a_query_without_a_document_is_an_error(impresso_dir):
    write_csv(impresso_dir / "multilingual" / "DE" / "queries.csv", [{"search_query": "x", "idx": "99"}])

    with pytest.raises(ValueError, match="match no document id"):
        load_split(impresso_dir / "multilingual" / "DE")


def test_missing_data_names_the_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "IMPRESSO_PATH", str(tmp_path / "nowhere"))

    with pytest.raises(FileNotFoundError, match="IMPRESSO_PATH"):
        load_tasks(["ImpressoMultilingualFR"])[0].load_data()


def test_a_task_runs_from_a_warm_cache_with_no_model(impresso_dir, tmp_path):
    cache_root = tmp_path / "cache"
    backbone, encode_config = get("mdenseon"), make_encode_config()
    (task,) = load_tasks(["ImpressoMultilingualDE"])
    task.load_data()
    precompute_task_cache(
        model=LetterModel(), backbone=backbone, encode_config=encode_config, task=task, cache_root=cache_root
    )

    cached = CachedBackbone(None, backbone, encode_config, cache_root)
    results = mteb.ResultCache(tmp_path / "results")
    baseline, scored = evaluate_cell(
        PostProcessedBackbone(cached, PostProcConfig(dr_method="pca", target_dim=8, quant_method="int8")),
        task,
        results_cache=results,
    )

    assert cached.newly_encoded == 0
    scores = baseline.task_results[0].scores["test"][0]
    assert scores["main_score"] == scores["ndcg_at_10"] == pytest.approx(1.0)
    assert scores["recall_at_5"] == pytest.approx(1.0)
    assert scored.task_results[0].get_score() > 0.5


def test_results_know_the_impresso_task_type():
    import pandas as pd

    from geopres_grid.results import add_task_metadata

    frame = add_task_metadata(pd.DataFrame({"task": ["ImpressoMultilingualDE", "NotATask"]}))

    assert frame["task_type"].tolist()[0] == "Retrieval"
    assert frame["main_score_name"].tolist()[0] == "ndcg_at_10"
    assert frame["task_type"].isna().tolist()[1]


# --- query-expanded variants ---------------------------------------------------------


@pytest.fixture
def expansion(impresso_dir, tmp_path, monkeypatch):
    """Keyword expansions of every query by the stub: the query word and "story"."""
    from geopres_grid.expansion import Budget, ExpansionConfig, StubExpander, read_queries, run_expansion

    root = tmp_path / "query_expansions"
    monkeypatch.setattr(config, "QUERY_EXPANSION_PATH", str(root))
    expansion = ExpansionConfig("gpt-test", "keywords")
    summary = run_expansion(
        expansion,
        read_queries(impresso_dir),
        StubExpander(lambda instructions, text: f"{text.split(': ', 1)[1]} story"),
        root,
        Budget(max_calls=100, max_tokens=10**6),
        requests_per_minute=0,
    )
    assert summary["usable"] == 40
    return expansion


def test_query_ids_match_the_task_ids(impresso_dir):
    from geopres_grid.expansion import read_queries

    queries = [q for q in read_queries(impresso_dir) if q.query_id.startswith("crosslingual/FR/")]
    split = load_split(impresso_dir / "crosslingual" / "FR")

    assert [(q.task_query_id, q.text) for q in queries] == [(row["id"], row["text"]) for row in split["queries"]]


def test_expanded_tasks_are_named_per_expansion_and_mode(expansion):
    names = with_query_expansions(["ImpressoMultilingualDE", "NFCorpus"], [expansion.hash])

    assert names == (
        f"ImpressoMultilingualDE__qe-keywords-{expansion.hash}-replace",
        f"ImpressoMultilingualDE__qe-keywords-{expansion.hash}-concat",
        "NFCorpus",
    )
    replace, concat = load_tasks(names[:2])
    replace.load_data()
    concat.load_data()
    texts = lambda task: {row["id"]: row["text"] for row in task.dataset["default"]["test"]["queries"]}
    assert texts(replace)["q3"] == "delta story"
    assert texts(concat)["q3"] == "delta\ndelta story"
    assert replace.dataset["default"]["test"]["relevant_docs"] == load_split(
        Path(config.IMPRESSO_PATH) / "multilingual" / "DE"
    )["relevant_docs"]
    assert replace.corpus_source == "ImpressoMultilingualDE"
    assert replace.metadata.eval_langs == TASKS["ImpressoMultilingualDE"].metadata.eval_langs


def test_an_expanded_task_needs_every_expansion_for_the_current_text(expansion, impresso_dir):
    (task,) = load_tasks([expanded_task_name("ImpressoCrosslingualDE", expansion, "concat")])
    write_csv(
        impresso_dir / "crosslingual" / "DE" / "queries.csv",
        [{"search_query": "changed", "idx": "0"}],
    )
    with pytest.raises(ValueError, match="another text"):
        task.load_data()

    other = ExpansionConfig("gpt-test", "pseudodoc")
    with pytest.raises(FileNotFoundError, match="expand_queries"):
        load_tasks([expanded_task_name("ImpressoCrosslingualDE", other, "concat")])


def test_an_expanded_task_copies_the_documents_and_encodes_only_queries(expansion, tmp_path):
    cache_root = tmp_path / "cache"
    backbone, encode_config = get("mdenseon"), make_encode_config()
    precompute = lambda task: precompute_task_cache(
        model=LetterModel(), backbone=backbone, encode_config=encode_config, task=task, cache_root=cache_root
    )
    (original,) = load_tasks(["ImpressoMultilingualDE"])
    original.load_data()
    precompute(original)

    names = with_query_expansions(["ImpressoMultilingualDE"], [expansion.hash])
    cached = CachedBackbone(None, backbone, encode_config, cache_root)
    results = mteb.ResultCache(tmp_path / "results")
    cell = PostProcConfig(dr_method="pca", target_dim=8, quant_method="int8")
    for task in load_tasks(names):
        task.load_data()
        summary = precompute(task)
        assert summary["corpus_source"] == "ImpressoMultilingualDE" and len(summary["copied_blocks"]) == 1
        assert summary["newly_encoded"] == 10  # the queries; all ten documents came with the copy
        baseline, scored = evaluate_cell(PostProcessedBackbone(cached, cell), task, results_cache=results)
        assert baseline.task_results[0].get_score() > 0.5

    assert cached.newly_encoded == 0
    stored = sorted(path.stem for path in (tmp_path / "results").rglob("Impresso*.json"))
    # A baseline and a cell slot for each of the two variants, none overwriting another.
    assert stored == sorted(names * 2)
    run_dirs = {path.parent for path in (tmp_path / "results").rglob("Impresso*.json")}
    assert len(run_dirs) == 2


def test_results_give_an_expanded_task_its_originals_type(expansion):
    import pandas as pd

    from geopres_grid.results import add_task_metadata

    name = expanded_task_name("ImpressoCrosslingualFR", expansion, "replace")
    frame = add_task_metadata(pd.DataFrame({"task": [name]}))

    assert frame["task_type"].tolist() == ["Retrieval"] and frame["main_score_name"].tolist() == ["ndcg_at_10"]
