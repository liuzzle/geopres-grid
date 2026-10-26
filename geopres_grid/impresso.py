"""The Impresso IR test set as four custom MTEB retrieval tasks.

Andrianos' set (logging.md §25, §36): historical Swiss newspaper articles with OCR
noise, one corpus per language (DE 738 documents, FR 725) and one relevant document
per query, matched by row index. Queries come in two forms: `crosslingual` (English
queries over German/French documents) and `multilingual` (queries in the document
language). Four tasks, not one with four subsets: `evaluation.cached_blocks` pools
every subset of a task into one calibration sample, which would fit one PCA and
one quantizer on German and French, English and native queries together. The two
tasks of a language share a corpus byte for byte but encode it twice, because the
cache is keyed by task name; ~1.5k documents, minutes on a GPU.

The data is read from `config.IMPRESSO_PATH`, in the layout it was sent in. The task
classes need `mteb`, so this module is imported lazily (`evaluation.load_tasks`);
`evaluation.IMPRESSO_TASKS` carries the names.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from datasets import Dataset
from mteb.abstasks import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

from geopres_grid import config

SPLIT = "test"
SUBSET = "default"

QUERY_FORMS = {"crosslingual": ("eng-Latn", "English queries"), "multilingual": (None, "queries in the document language")}
LANGUAGES = {"DE": "deu-Latn", "FR": "fra-Latn"}

DATA_REVISION = "andrianos-2026-10-19"
"""MTEB keys a stored result by task name and this revision, so a changed data set
must get a new one (the embedding cache is keyed by text and cannot go stale)."""


def read_rows(path: Path) -> list[dict[str, str]]:
    # Articles reach 3,500 words; the csv module's default field limit is 128 kB.
    csv.field_size_limit(2**31 - 1)
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_split(folder: Path) -> dict[str, Any]:
    """MTEB's `RetrievalSplitData` for one `<form>/<language>` folder.

    Ids are prefixed (`d` for documents, `q` for queries, the query's row number) so
    no query id equals a document id. The qrels are the identity map of the data:
    query row `r` with `idx` `i` is answered by the document whose `id` is `i`.
    """
    documents = read_rows(folder / "docs.csv")
    queries = read_rows(folder / "queries.csv")
    document_ids = [row["id"] for row in documents]
    if len(set(document_ids)) != len(document_ids):
        raise ValueError(f"{folder}: duplicate document ids")
    unknown = sorted({row["idx"] for row in queries} - set(document_ids))
    if unknown:
        raise ValueError(f"{folder}: {len(unknown)} query idx values match no document id, e.g. {unknown[:3]}")
    return {
        "corpus": Dataset.from_list([{"id": f"d{row['id']}", "title": "", "text": row["text"]} for row in documents]),
        "queries": Dataset.from_list([{"id": f"q{n}", "text": row["search_query"]} for n, row in enumerate(queries)]),
        "relevant_docs": {f"q{n}": {f"d{row['idx']}": 1} for n, row in enumerate(queries)},
        "top_ranked": None,
    }


class ImpressoRetrieval(AbsTaskRetrieval):
    """Base of the four tasks; a subclass sets `metadata`, `form` and `language`."""

    form: str
    language: str

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        if self.data_loaded:
            return
        folder = Path(config.IMPRESSO_PATH) / self.form / self.language
        if not folder.is_dir():
            raise FileNotFoundError(
                f"Impresso data not found at {folder}. Copy Impresso_IR_test_set_eval_eval_format "
                f"to {config.IMPRESSO_PATH} (or set IMPRESSO_PATH)."
            )
        self.dataset = {SUBSET: {SPLIT: load_split(folder)}}
        self.data_loaded = True


def _metadata(form: str, language: str) -> TaskMetadata:
    query_language, query_note = QUERY_FORMS[form]
    return TaskMetadata(
        name=f"Impresso{form.capitalize()}{language}",
        description=(
            f"Impresso historical newspaper retrieval, {language} documents with OCR noise, "
            f"{query_note}; one relevant document per query."
        ),
        reference=None,
        dataset={"path": str(Path("impresso") / form / language), "revision": DATA_REVISION},
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=[SPLIT],
        eval_langs=[LANGUAGES[language]] if query_language is None else [query_language, LANGUAGES[language]],
        main_score="ndcg_at_10",
        domains=["News"],
        license=None,
        is_public=False,
        contributed_by="Andrianos",
        bibtex_citation=None,
    )


def _task(form: str, language: str) -> type[ImpressoRetrieval]:
    metadata = _metadata(form, language)
    return type(metadata.name, (ImpressoRetrieval,), {"metadata": metadata, "form": form, "language": language})


TASKS: dict[str, type[ImpressoRetrieval]] = {
    cls.metadata.name: cls for cls in (_task(form, language) for language in LANGUAGES for form in QUERY_FORMS)
}
