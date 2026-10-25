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


def test_pca_cells_of_both_mean_modes_are_never_averaged_together():
    old = make_frame()  # written before the column existed: all PCA cells subtracted
    restored = old[old["dr_method"].isin(["pca", "pca_ror"])].assign(pca_mean="restore", main_score=0.1)
    both = pd.concat([old.assign(pca_mean=None), restored])

    with pytest.raises(ValueError, match="both pca_mean modes"):
        analysis.scored_rows(both)
    kept = analysis.scored_rows(both, pca_mean="restore")
    assert (kept.loc[kept["dr_method"] == "pca", "main_score"] == 0.1).all()
    assert len(kept[kept["dr_method"] == "truncate"]) == 4
    assert set(analysis.scored_rows(old)["pca_mean"].dropna()) == {"subtract"}


def cells_of(frame=None):
    return analysis.cell_summary(analysis.scored_rows(make_frame() if frame is None else frame))


def test_native_dim_comes_from_the_fp32_baseline():
    assert analysis.native_dims(cells_of()).to_dict() == {"mdenseon": 8, "mgte": 8}


def test_reduction_only_ends_at_the_native_dimension_with_the_baseline():
    retention = analysis.reduction_only(cells_of())
    assert list(retention.columns) == [4, 8]
    assert (retention[8] == 1.0).all()
    scores = analysis.reduction_only(cells_of(), "score")
    assert scores.loc[("mgte", "truncate"), 8] == pytest.approx(0.6)  # mean of the 0.8 / 0.4 baselines
    assert scores.loc[("mgte", "truncate"), 4] == pytest.approx(0.6 * 0.9)


def test_quantization_only_has_an_absolute_twin():
    scores = analysis.quantization_only(cells_of(), "score")
    assert list(scores.columns) == ["none", "binary"]
    assert scores.loc["mgte", "none"] == pytest.approx(0.6)
    assert scores.loc["mgte", "binary"] == pytest.approx(0.54)
    assert analysis.quantization_only(cells_of()).loc["mgte", "binary"] == pytest.approx(0.9)


def test_gains_carry_both_absolute_scores():
    gain = analysis.ror_gain(cells_of()).set_index(["backbone", "target_dim", "quant_method"])
    row = gain.loc[("mgte", 4, "binary")]
    assert row["better_score"] == pytest.approx(0.6 * 0.7)
    assert row["worse_score"] == pytest.approx(0.6 * 0.5)


def test_task_losses_deltas_and_scores_agree():
    rows = analysis.scored_rows(make_frame())
    selection = {"full/binary": ("none", 0, "binary"), "truncate-4": ("truncate", 4, "none")}
    losses = analysis.task_losses(rows, selection)
    assert len(losses) == 2 * 2 * 2  # cells x backbones x tasks
    assert set(losses.columns) >= {"config", "backbone", "task", "baseline", "score", "delta", "retention"}
    deltas = analysis.task_deltas(rows, selection)
    assert list(deltas.columns) == list(selection)
    assert deltas.loc["A", "full/binary"] == pytest.approx(-8.0)
    scores = analysis.task_scores(rows, selection)
    assert list(scores.columns) == ["fp32", *selection]
    assert scores.loc["A", "fp32"] == pytest.approx(0.8)
    assert scores.loc["A", "full/binary"] == pytest.approx(0.72)
    # truncate: mgte 0.9, mdenseon 0.6 of 0.8
    assert scores.loc["A", "truncate-4"] == pytest.approx(0.8 * 0.75)


def test_task_splits_refuses_a_task_on_two_splits():
    rows = analysis.scored_rows(make_frame())
    assert analysis.task_splits(rows) == {"A": "test", "B": "test"}
    with pytest.raises(ValueError, match="more than one split"):
        analysis.task_splits(pd.concat([rows, rows.head(1).assign(split="dev")]))


def test_corpus_sizes_from_mteb_statistics():
    sizes = analysis.corpus_sizes(
        {"ArguAna": "test", "NanoArguAnaRetrieval": "train", "Touche2020": "test", "CQADupstackRetrieval": "test"}
    )
    assert sizes.loc["ArguAna", "n_documents"] == 8674
    assert sizes.loc["NanoArguAnaRetrieval", "n_documents"] == 3635
    # Touche2020's statistics have no num_documents; the text statistics give the count.
    assert sizes.loc["Touche2020", "n_documents"] == 382545
    # The aggregate is scored per forum, each against its own corpus: the mean of twelve.
    assert sizes.loc["CQADupstackRetrieval", "n_subtasks"] == 12
    assert 2e4 < sizes.loc["CQADupstackRetrieval", "n_documents"] < 6e4


def synthetic_losses(pca_mean=None):
    tasks = [f"T{i}" for i in range(6)]
    sizes = pd.DataFrame({"task": tasks, "n_documents": [10.0 ** (3 + i) for i in range(6)]}).set_index("task")
    rows = []
    for backbone, slope in (("mgte", 0.05), ("mdenseon", 0.02)):
        for i, task in enumerate(tasks):
            rows.append(
                {"config": "pca-64", "backbone": backbone, "task": task, "retention": 1 - slope * i, "pca_mean": pca_mean}
            )
    return pd.DataFrame(rows), sizes


def test_size_correlation_finds_a_monotone_loss_per_backbone_and_on_the_mean():
    losses, sizes = synthetic_losses("restore")
    correlation = analysis.size_correlation(losses, sizes).set_index("backbone")
    assert set(correlation.index) == {"mean", "mgte", "mdenseon"}
    assert correlation.loc["mean", "spearman"] == pytest.approx(-1.0)
    assert correlation.loc["mgte", "pearson_log10"] == pytest.approx(-1.0)
    assert correlation.loc["mean", "n_tasks"] == 6
    assert correlation.loc["mean", "max_documents"] == 1e8


def test_size_correlation_refuses_both_pca_mean_modes_in_one_group():
    losses, sizes = synthetic_losses("restore")
    losses.loc[losses["backbone"] == "mgte", "pca_mean"] = "subtract"
    with pytest.raises(ValueError, match="both pca_mean modes"):
        analysis.size_correlation(losses, sizes)


def test_paired_tiers_match_a_nano_task_to_its_full_dataset():
    assert analysis.dataset_name("NanoClimateFeverRetrieval") == analysis.dataset_name("ClimateFEVER")
    assert analysis.dataset_name("NanoQuoraRetrieval") == analysis.dataset_name("QuoraRetrieval")
    base = {"config": "full/binary", "backbone": "mgte", "score": 0.5, "pca_mean": None}
    losses = pd.DataFrame(
        [
            {**base, "tier": "tier1", "task": "NanoFEVERRetrieval", "retention": 0.95},
            {**base, "tier": "tier2", "task": "FEVER", "retention": 0.85},
            {**base, "tier": "tier2", "task": "TRECCOVID", "retention": 0.9},  # no Nano counterpart
        ]
    )
    paired = analysis.paired_tiers(losses, "tier1", "tier2")
    assert len(paired) == 1
    assert paired.loc[0, "task"] == "FEVER"
    assert paired.loc[0, "retention_change"] == pytest.approx(-0.1)

    mixed = losses.assign(config="pca-64", pca_mean=["subtract", "restore", "restore"])
    with pytest.raises(ValueError, match="different pca_mean modes"):
        analysis.paired_tiers(mixed, "tier1", "tier2")


def test_every_figure_draws():
    pytest.importorskip("matplotlib")
    from geopres_grid import plots

    rows = analysis.scored_rows(make_frame())
    cells = analysis.cell_summary(rows)
    selection = {"full/binary": ("none", 0, "binary"), "pca-4": ("pca", 4, "none")}
    losses = analysis.task_losses(rows, selection)
    sizes = pd.DataFrame({"task": ["A", "B"], "n_documents": [1e4, 1e6]}).set_index("task")
    frontier = analysis.budget_frontier(cells)
    figures = [
        plots.quantization_only(analysis.quantization_only(cells), analysis.quantization_only(cells, "score")),
        plots.reduction_only(analysis.reduction_only(cells), analysis.reduction_only(cells, "score")),
        plots.budget_frontier(cells, frontier),
        plots.budget_frontier_by_backbone(cells, frontier),
        plots.ror_gain_by_backbone(analysis.ror_gain(cells)),
        plots.task_deltas(analysis.task_deltas(rows, selection), analysis.task_scores(rows, selection)["fp32"]),
        plots.task_deltas_by_backbone(losses, list(selection)),
        plots.corpus_size_loss(losses, sizes, analysis.size_correlation(losses, sizes)),
    ]
    for figure in figures:
        assert figure.axes
        plots._pyplot().close(figure)


def test_read_results_keeps_a_shared_row_once(tmp_path):
    frame = make_frame()
    restored = frame[frame["dr_method"] == "pca"].assign(run_id=lambda f: f["run_id"] + "-restore", pca_mean="restore")
    rerun = pd.concat([frame[frame["kind"] == "baseline"], restored])  # carries the baselines again
    frame.to_csv(tmp_path / "results.csv", index=False)
    rerun.to_csv(tmp_path / "results_restore.csv", index=False)
    both = analysis.read_results([tmp_path / "results.csv", tmp_path / "results_restore.csv"])
    assert len(both) == len(frame) + len(restored)
    rows = analysis.scored_rows(both, pca_mean="restore")
    assert set(rows.loc[rows["dr_method"] == "pca", "pca_mean"]) == {"restore"}


def two_tiers():
    """Tier 1 on two Nano tasks; tier 2 on their full datasets plus one without a Nano twin,
    where every binary cell loses a further 20 % of its score."""
    small = make_frame()
    small = small[small["task_type"] == "Retrieval"]
    small["task"] = small["task"].map({"A": "NanoFEVERRetrieval", "B": "NanoNQRetrieval"})
    large = small.copy()
    large["task"] = large["task"].map({"NanoFEVERRetrieval": "FEVER", "NanoNQRetrieval": "NQ"})
    large.loc[large["quant_method"] == "binary", "main_score"] *= 0.8
    extra = large[large["task"] == "NQ"].assign(task="TRECCOVID", main_score=0.1)
    return small, pd.concat([large, extra], ignore_index=True)


def paired_tiers_of(small, large):
    rows_small, rows_large = analysis.shared_datasets(analysis.scored_rows(small), analysis.scored_rows(large))
    return analysis.tier_cells(analysis.cell_summary(rows_small), analysis.cell_summary(rows_large))


def test_shared_datasets_drop_a_task_without_a_twin():
    small, large = two_tiers()
    rows_small, rows_large = analysis.shared_datasets(analysis.scored_rows(small), analysis.scored_rows(large))
    assert set(rows_small["task"]) == {"NanoFEVERRetrieval", "NanoNQRetrieval"}
    assert set(rows_large["task"]) == {"FEVER", "NQ"}


def test_tier_cells_pair_every_cell_and_measure_the_change():
    paired = paired_tiers_of(*two_tiers())
    assert len(paired) == 2 * 8  # backbones x cells
    cell = paired[(paired["backbone"] == "mgte") & (paired["label"] == "full/binary")].iloc[0]
    assert cell["retention_small"] == pytest.approx(0.9)
    assert cell["retention_large"] == pytest.approx(0.72)
    assert cell["retention_change"] == pytest.approx(-0.18)
    assert cell["score_large"] == pytest.approx(0.6 * 0.72)
    assert (paired["n_tasks_small"] == 2).all() and (paired["n_tasks_large"] == 2).all()


def test_tier_cells_refuse_pca_cells_in_different_modes():
    small, large = two_tiers()
    large = large.assign(pca_mean=lambda f: f["dr_method"].map({"pca": "restore", "pca_ror": "restore"}))
    with pytest.raises(ValueError, match="pca_mean"):
        paired_tiers_of(small, large)


def test_rank_transfer_and_scale_cost():
    paired = paired_tiers_of(*two_tiers())
    transfer = analysis.rank_transfer(paired).set_index("backbone")
    assert set(transfer.index) == {"mgte", "mdenseon", "pooled"}
    assert transfer.loc["mgte", "n_cells"] == 7  # the baseline is left out
    assert transfer.loc["pooled", "n_cells"] == 14
    cost = analysis.scale_cost(paired)
    assert cost.loc[("none", 0), "binary"] == pytest.approx(-18.0)
    assert cost.loc[("truncate", 4), "none"] == pytest.approx(0.0)
    assert list(cost.index.get_level_values(0)) == ["none", "truncate", "pca", "pca_ror", "random_selection"]


def test_budget_choice_scores_the_small_tiers_pick_on_the_large_tier():
    small, large = two_tiers()
    # On the large tier mgte's 16-byte truncation wins; on the small tier PCA does.
    large.loc[(large["backbone"] == "mgte") & (large["dr_method"] == "truncate"), "main_score"] *= 1.2
    small.loc[(small["backbone"] == "mgte") & (small["dr_method"] == "truncate"), "main_score"] *= 0.5
    choice = analysis.budget_choice(paired_tiers_of(small, large)).set_index(["backbone", "budget"])
    at_16 = choice.loc[("mgte", 16)]
    assert not at_16["same"]
    assert at_16["large_pick"] == "truncate-4/none"
    assert at_16["small_pick"] in {"pca-4/none", "pca_ror-4/none"}
    assert at_16["regret"] == pytest.approx((0.6 * 0.9 * 1.2 - 0.6 * 0.95) * 100)
    assert choice.loc[("mgte", 1), "same"]
    assert choice.loc[("mgte", 1), "regret"] == pytest.approx(0.0)
    assert list(choice.loc["mgte"].index) == [1, 16, 32]


def test_tier_figures_draw():
    pytest.importorskip("matplotlib")
    from geopres_grid import plots

    paired = paired_tiers_of(*two_tiers())
    for figure in (
        plots.tier_transfer(paired, analysis.rank_transfer(paired)),
        plots.scale_cost(analysis.scale_cost(paired)),
        plots.budget_choice(analysis.budget_choice(paired)),
    ):
        assert figure.axes
        plots._pyplot().close(figure)
