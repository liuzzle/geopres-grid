"""`scripts/analyse_results.py` on the `nonretrieval` tier: three task types, none of them retrieval."""

import json

import pandas as pd
import pytest

from analyse_results import main

TASKS = {
    "STS12": ("STS", 0.8),
    "SICK-R": ("STS", 0.7),
    "ImdbClassification": ("Classification", 0.9),
    "AmazonReviewsClassification": ("Classification", 0.5),
    "RedditClustering": ("Clustering", 0.6),
}
CELLS = (
    ("none", None, "none", 1.0),
    ("none", None, "binary", 0.9),
    ("pca", 4, "none", 0.95),
    ("pca_ror", 4, "binary", 0.8),
    ("truncate", 4, "none", 0.85),
)


def results_csv(path):
    rows = []
    for backbone in ("mgte", "mdenseon"):
        for task, (task_type, baseline) in TASKS.items():
            for subset in ("de", "en") if task == "AmazonReviewsClassification" else ("default",):
                for dr_method, target_dim, quant_method, retention in CELLS:
                    rows.append({
                        "source": "geopres-grid", "backbone": backbone,
                        "kind": "baseline" if retention == 1.0 else "cell",
                        "run_id": f"{backbone}-{dr_method}-{target_dim}-{quant_method}",
                        "dr_method": dr_method, "target_dim": target_dim, "dr_seed": None,
                        "pca_mean": "restore" if dr_method.startswith("pca") else None,
                        "quant_method": quant_method, "bits_per_dim": 1 if quant_method == "binary" else 32,
                        "bytes_per_vector": 16, "compression_factor": 1.0,
                        "task": task, "split": "test", "subset": subset,
                        "main_score": baseline * retention, "task_type": task_type,
                    })
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.mark.parametrize("task_type", ["STS", "Classification", "Clustering"])
def test_each_non_retrieval_type_is_analysed_without_corpus_sizes(tmp_path, task_type):
    out = tmp_path / task_type
    # Figures are left on: they are skipped for non-retrieval types, not drawn in nDCG@10.
    main([str(results_csv(tmp_path / "results.csv")), "--tier", "nonretrieval", "--task-type", task_type, "--out", str(out)])

    summary = json.loads((out / "summary.json").read_text())
    assert summary["tasks"] == sum(kind == task_type for kind, _ in TASKS.values())
    assert summary["incomplete_cells"] == 0
    assert "corpus_documents" not in summary
    assert not (out / "corpus_sizes.csv").exists() and not list(out.glob("*.png"))
    cells = pd.read_csv(out / "cells.csv")
    binary = cells[(cells["dr_method"] == "none") & (cells["quant_method"] == "binary")]
    assert binary["retention"].round(6).eq(0.9).all()


def test_a_task_type_with_no_rows_names_the_ones_there_are(tmp_path):
    with pytest.raises(SystemExit, match="Classification"):
        main([str(results_csv(tmp_path / "results.csv")), "--tier", "nonretrieval", "--out", str(tmp_path / "out")])
