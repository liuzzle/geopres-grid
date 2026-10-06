import json

import pytest

from geopres_grid.backbones import get
from geopres_grid.cache import CachedBackbone
from geopres_grid.evaluation import run_record
from geopres_grid.identity import EncodeConfig
from geopres_grid.identity import PostProcConfig
from geopres_grid.results import comparison_table
from geopres_grid.results import load_results
from geopres_grid.results import load_upstream_results
from geopres_grid.results import parse_upstream_model_name
from geopres_grid.results import write_run_record


def make_cached(tmp_path):
    config = EncodeConfig(
        model_id="lightonai/mDenseOn",
        revision="a" * 40,
        max_seq_length=512,
        dtype="float32",
        prompts=(("query", "query: "), ("document", "document: ")),
        sentence_transformers_version="5.7",
        transformers_version="5.16",
    )
    return CachedBackbone(None, get("mdenseon"), config, tmp_path / "cache")


def write_result(directory, task, split, main_score, **metrics):
    directory.mkdir(parents=True, exist_ok=True)
    entry = {"main_score": main_score, "hf_subset": "default", "languages": ["eng-Latn"], **metrics}
    payload = {"task_name": task, "mteb_version": "2.15.1", "dataset_revision": "r",
               "scores": {split: [entry]}}
    (directory / f"{task}.json").write_text(json.dumps(payload))
    (directory / "model_meta.json").write_text(json.dumps({"name": "ignored"}))


def make_results(tmp_path):
    cached = make_cached(tmp_path)
    cell = PostProcConfig(dr_method="pca_ror", target_dim=128, quant_method="equal_count_4")
    baseline_record = run_record(cached, None)
    cell_record = run_record(cached, cell)
    root = tmp_path / "evaluation_results"
    for record, scores in ((baseline_record, (0.60, 0.85)), (cell_record, (0.54, 0.83))):
        write_run_record(root, record)
        slot = root / "results" / "geopres-grid__mdenseon" / record["revision"]
        write_result(slot, "NanoArguAnaRetrieval", "train", scores[0], recall_at_100=0.9)
        write_result(slot, "STSBenchmark", "test", scores[1])
    return root, baseline_record, cell_record


def test_load_results_joins_each_slot_to_its_run_record(tmp_path):
    root, baseline_record, cell_record = make_results(tmp_path)

    frame = load_results(root, metrics=("recall_at_100",))

    assert len(frame) == 4
    cell = frame[frame["run_id"] == cell_record["revision"]].set_index("task")
    assert cell.loc["NanoArguAnaRetrieval", "dr_method"] == "pca_ror"
    assert cell.loc["NanoArguAnaRetrieval", "target_dim"] == 128
    assert cell.loc["NanoArguAnaRetrieval", "bits_per_dim"] == 4
    assert cell.loc["NanoArguAnaRetrieval", "compression_factor"] == 768 * 32 / (128 * 4)
    assert cell.loc["NanoArguAnaRetrieval", "split"] == "train"
    assert cell.loc["NanoArguAnaRetrieval", "recall_at_100"] == 0.9
    assert cell.loc["NanoArguAnaRetrieval", "task_type"] == "Retrieval"
    assert cell.loc["STSBenchmark", "main_score_name"] == "cosine_spearman"
    baseline = frame[frame["run_id"] == baseline_record["revision"]]
    assert set(baseline["kind"]) == {"baseline"}
    assert set(baseline["compression_factor"]) == {1.0}


def test_a_result_slot_without_a_run_record_is_an_error(tmp_path):
    root, _, cell_record = make_results(tmp_path)
    (root / "runs" / f"{cell_record['revision']}.json").unlink()

    with pytest.raises(ValueError, match="without a run record"):
        load_results(root)


def test_comparison_table_has_upstreams_wide_layout(tmp_path):
    root, baseline_record, cell_record = make_results(tmp_path)

    wide = comparison_table(load_results(root), tasks=["NanoArguAnaRetrieval", "STSBenchmark"])

    assert len(wide) == 2
    assert list(wide.columns) == ["AVG_ALL", "AVG_RETRIEVAL", "AVG_STS", "NanoArguAnaRetrieval", "STSBenchmark"]
    row = wide.xs(cell_record["revision"], level="run_id").iloc[0]
    assert row["AVG_ALL"] == pytest.approx((0.54 + 0.83) / 2)


@pytest.mark.parametrize(
    "name, expected",
    [
        ("Alibaba-NLP__gte-multilingual-base", ("Alibaba-NLP/gte-multilingual-base", "none", None, "")),
        ("Alibaba-NLP__gte-multilingual-base_reduced_128_pca", ("Alibaba-NLP/gte-multilingual-base", "pca", 128, "")),
        ("jinaai__jina-embeddings-v2-small-en_reduced_32_truncation", ("jinaai/jina-embeddings-v2-small-en", "truncate", 32, "")),
        ("Qwen__Qwen3-Embedding-0.6B_reduced_2_batch_20000_poslossfactor_1.0",
         ("Qwen/Qwen3-Embedding-0.6B", "geopres", 2, "batch_20000_poslossfactor_1.0")),
    ],
)
def test_parse_upstream_model_name(name, expected):
    parsed = parse_upstream_model_name(name)

    assert (parsed["model_id"], parsed["dr_method"], parsed["target_dim"], parsed["variant"]) == expected


def test_load_upstream_results_reads_upstreams_tree(tmp_path):
    """Layout written by upstream's eval_backbone.py, baselines/eval_pca.py, eval_model.py."""
    slug = "Alibaba-NLP__gte-multilingual-base"
    write_result(tmp_path / "backbone" / slug / "results" / slug / "9bbca17d", "ArguAna", "test", 0.55)
    pca = tmp_path / "pca_projection" / slug / "results" / f"{slug}_reduced_128_pca" / "no_revision_available"
    write_result(pca, "ArguAna", "test", 0.50)
    (pca / "intrinsic.json").write_text(json.dumps({"spearman_loss": 0.1, "angular_loss": 0.2,
                                                    "positional_loss": 0.3, "checkpoint": None}))
    trained = (tmp_path / "trained_models" / slug / "results"
               / f"{slug}_reduced_128_batch_20000_poslossfactor_1.0" / "no_revision_available")
    write_result(trained, "MSMARCO", "test", 0.30)

    frame = load_upstream_results(tmp_path)

    assert set(frame["source"]) == {"geopres-upstream"}
    assert set(frame["backbone"]) == {"mgte"}
    pca_rows = frame[frame["dr_method"] == "pca"].set_index("task")
    assert pca_rows.loc["ArguAna", "compression_factor"] == 6.0
    assert pca_rows.loc["spearman_loss", "task_type"] == "Intrinsic"
    geopres = frame[frame["dr_method"] == "geopres"].iloc[0]
    assert (geopres["variant"], geopres["split"]) == ("batch_20000_poslossfactor_1.0", "test")
    assert frame[frame["kind"] == "baseline"]["main_score"].tolist() == [0.55]

    wide = comparison_table(frame)
    assert "AVG_INTRINSIC" not in wide.columns
    assert "spearman_loss" in wide.columns


def test_averages_are_missing_when_a_task_is_missing(tmp_path):
    root, _, cell_record = make_results(tmp_path)
    (root / "results" / "geopres-grid__mdenseon" / cell_record["revision"] / "STSBenchmark.json").unlink()

    wide = comparison_table(load_results(root))

    row = wide.xs(cell_record["revision"], level="run_id").iloc[0]
    assert row["AVG_RETRIEVAL"] == pytest.approx(0.54)
    assert row.isna()["AVG_STS"] and row.isna()["AVG_ALL"]
