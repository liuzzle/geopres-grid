import numpy as np
import pandas as pd
import pytest

from geopres_grid import analysis


def row(backbone, task, dr_method, target_dim, quant_method, score, kind="cell", bytes_per_vector=None):
    return {
        "backbone": backbone,
        "kind": kind,
        "run_id": f"{backbone}-{dr_method}-{target_dim}-{quant_method}",
        "dr_method": dr_method,
        "target_dim": target_dim,
        "quant_method": quant_method,
        "bits_per_dim": 32,
        "bytes_per_vector": bytes_per_vector or (target_dim or 8) * 4,
        "compression_factor": 1.0,
        "task": task,
        "split": "test",
        "subset": "default",
        "main_score": score,
        "task_type": "Retrieval",
    }


def make_frame():
    rows = []
    for backbone in ("mdenseon", "mgte"):
        for task, base in (("A", 0.8), ("B", 0.4)):
            rows.append(row(backbone, task, "none", None, "none", base, kind="baseline", bytes_per_vector=32))
            rows.append(row(backbone, task, "none", None, "binary", base * 0.9, bytes_per_vector=1))
            rows.append(row(backbone, task, "pca", 4, "none", base * 0.95))
            rows.append(row(backbone, task, "pca_ror", 4, "none", base * 0.95))
            rows.append(row(backbone, task, "pca", 4, "binary", base * 0.5, bytes_per_vector=1))
            rows.append(row(backbone, task, "pca_ror", 4, "binary", base * 0.7, bytes_per_vector=1))
            rows.append(row(backbone, task, "truncate", 4, "none", base * (0.9 if backbone == "mgte" else 0.6)))
            rows.append(row(backbone, task, "random_selection", 4, "none", base * 0.7))
    sts = row("mgte", "STSBenchmark", "none", None, "none", 0.86, kind="baseline")
    sts["task_type"] = "STS"
    rows.append(sts)
    return pd.DataFrame(rows)


def test_scored_rows_retention_and_other_task_types_dropped():
    rows = analysis.scored_rows(make_frame())
    assert set(rows["task"]) == {"A", "B"}
    binary = rows[(rows["dr_method"] == "none") & (rows["quant_method"] == "binary")]
    assert np.allclose(binary["retention"], 0.9)
    assert (rows.loc[rows["kind"] == "baseline", "retention"] == 1).all()
    assert rows["target_dim"].dtype.kind == "i"


def test_scored_rows_refuses_missing_or_duplicate_baselines():
    frame = make_frame()
    without = frame[~((frame["kind"] == "baseline") & (frame["task"] == "A") & (frame["backbone"] == "mgte"))]
    with pytest.raises(ValueError, match="no baseline"):
        analysis.scored_rows(without)
    doubled = pd.concat([frame, frame[frame["kind"] == "baseline"].head(1)])
    with pytest.raises(ValueError, match="more than one baseline"):
        analysis.scored_rows(doubled)


def test_retention_is_the_mean_of_ratios_not_the_ratio_of_means():
    frame = make_frame()
    frame.loc[(frame["task"] == "B") & (frame["quant_method"] == "binary") & (frame["dr_method"] == "none"), "main_score"] = 0.2
    cells = analysis.cell_summary(analysis.scored_rows(frame))
    cell = cells[(cells["backbone"] == "mgte") & (cells["dr_method"] == "none") & (cells["quant_method"] == "binary")]
    assert cell["retention"].item() == pytest.approx((0.9 + 0.5) / 2)
    assert cell["n_tasks"].item() == 2
    assert cell["label"].item() == "full/binary"


def test_ror_gain_is_zero_at_fp32_and_matched_per_cell():
    cells = analysis.cell_summary(analysis.scored_rows(make_frame()))
    gain = analysis.ror_gain(cells).set_index(["backbone", "target_dim", "quant_method"])["gain"]
    assert gain[("mgte", 4, "none")] == pytest.approx(0.0)
    assert gain[("mgte", 4, "binary")] == pytest.approx(20.0)


def test_prefix_gap_signs_follow_truncation():
    cells = analysis.cell_summary(analysis.scored_rows(make_frame()))
    gap = analysis.prefix_gap(cells).set_index("backbone")["gain"]
    assert gap["mgte"] == pytest.approx(20.0)
    assert gap["mdenseon"] == pytest.approx(-10.0)


def test_budget_frontier_keeps_only_cells_no_cheaper_cell_beats():
    cells = analysis.cell_summary(analysis.scored_rows(make_frame()))
    front = analysis.budget_frontier(cells)
    for _, group in front.groupby("backbone"):
        assert group["bytes_per_vector"].is_monotonic_increasing
        assert group["score"].is_monotonic_increasing
        assert group["bytes_per_vector"].is_unique
    mgte = front[front["backbone"] == "mgte"]
    # At 1 byte the best is full/binary (0.9), not the PCA binaries; at 32 the baseline.
    assert mgte.iloc[0]["label"] == "full/binary"
    assert mgte.iloc[-1]["score"] == pytest.approx(0.6)


def test_backbone_order_follows_the_registry():
    assert analysis.backbone_order(make_frame()) == ["mgte", "mdenseon"]


def test_centring_check_full_rank_uncentred_is_exact_and_centring_shifts_a_cone():
    generator = np.random.default_rng(0)
    # A narrow cone: a large shared direction plus noise, as real embeddings are.
    shared = np.zeros(16, dtype=np.float32)
    shared[0] = 3.0
    documents = shared + generator.normal(size=(300, 16)).astype(np.float32)
    queries = shared + generator.normal(size=(20, 16)).astype(np.float32)
    table = analysis.centring_check(queries, documents, [16, 8])
    assert table.loc[16, "pca_uncentred"] == pytest.approx(1.0)
    assert table.loc[16, "truncate"] == pytest.approx(1.0)
    assert table.loc[16, "pca_centred"] < 1.0
    assert table.attrs["mean_norm"] > 0.5
