"""`scripts/thesis_figures.py`: labels, formats and the two seed summaries, on tiny frames."""

import numpy as np
import pandas as pd

from thesis_figures import (
    REFERENCE,
    best_per_budget,
    bold_best,
    config_name,
    latex_table,
    macro_name,
    prefix_gap_seeds,
    ror_gain_seeds,
)


def test_configuration_names_follow_the_plan():
    assert config_name("pca_ror", 256, "equal_count_2") == "PCA+ROR-256 / EC-2"
    assert config_name("none", 0, "binary") == "Full / binary"
    assert config_name("truncate", 128, "none") == "Truncation-128 / fp32"
    assert len(REFERENCE) == 8 and all(config_name(*v) == k for k, v in REFERENCE.items())


def test_bold_marks_the_best_unless_it_is_a_tie():
    assert bold_best([56.7, 55.0, 51.1], ["56.7", "55.0", "51.1"]) == ["\\textbf{56.7}", "55.0", "51.1"]
    assert bold_best([56.7, 56.4, 51.1], ["56.7", "56.4", "51.1"]) == ["56.7", "56.4", "51.1"]


def test_macro_names_are_letters_only():
    name = macro_name("prefixGap", "BEIR", "k64", "Mgte")
    assert name == "numPrefixGapBEIRKSixFourMgte" and name.isalpha()


def test_latex_table_is_booktabs_with_group_header():
    text = latex_table(["a", "b"], [["1", "2"], ["3", "4"]], "lr", groups=[("", 1), ("set", 1)], midrule_before=[1])
    assert "\\toprule" in text and "\\bottomrule" in text and text.count("\\midrule") == 2
    assert "\\multicolumn{1}{c}{set}" in text and "\\cmidrule(lr){2-2}" in text


def test_an_empty_group_fills_all_the_columns_it_spans():
    """T9's header: two unlabelled columns, then BEIR over three, Impresso over two."""
    text = latex_table(list("abcdefg"), [list("1234567")], "lrlrrlr", groups=[("", 2), ("BEIR", 3), ("Impresso", 2)])
    group_line = next(line for line in text.splitlines() if "multicolumn" in line)
    assert group_line.startswith(" &  & \\multicolumn{3}{c}{BEIR}")
    assert "\\cmidrule(lr){3-5} \\cmidrule(lr){6-7}" in text


def test_best_per_budget_flags_a_runner_up_within_half_a_point_as_a_tie():
    table = cells(
        [
            {"backbone": "mgte", "dr_method": "pca", "target_dim": 64, "quant_method": "equal_count_2", "bytes_per_vector": 16, "score": 0.300, "retention": 0.6},
            {"backbone": "mgte", "dr_method": "pca_ror", "target_dim": 64, "quant_method": "equal_count_2", "bytes_per_vector": 16, "score": 0.297, "retention": 0.59},
            {"backbone": "mgte", "dr_method": "pca", "target_dim": 128, "quant_method": "equal_count_2", "bytes_per_vector": 32, "score": 0.350, "retention": 0.7},
        ]
    )
    best = best_per_budget(table).set_index("budget")
    assert bool(best.loc[16, "tie"]) and best.loc[16, "runner_up"] == "PCA+ROR-64 / EC-2"
    assert not bool(best.loc[32, "tie"])  # 0.350 against 0.300: five points apart


def cells(records):
    frame = pd.DataFrame(records)
    frame["label"] = frame["dr_method"] + "-" + frame["target_dim"].astype(str) + "/" + frame["quant_method"]
    return frame


def test_prefix_gap_is_summarised_over_the_random_side_seeds():
    deterministic = cells([{"backbone": "b", "dr_method": "truncate", "target_dim": 64, "quant_method": "none", "retention": 0.80, "score": 0.4}])
    seeds = pd.DataFrame(
        {"backbone": "b", "dr_method": "random_selection", "target_dim": 64, "quant_method": "none", "dr_seed": [42, 1, 2], "retention": [0.70, 0.72, 0.68]}
    )
    gap = prefix_gap_seeds(deterministic, seeds)
    assert gap.loc[0, "n_seeds"] == 3 and np.isclose(gap.loc[0, "gap_mean"], 10.0) and bool(gap.loc[0, "resolved"])


def test_ror_gain_pairs_pca_and_ror_by_seed():
    rows = []
    for seed, (pca, ror) in enumerate([(0.50, 0.60), (0.52, 0.60)]):
        rows.append({"backbone": "b", "dr_method": "pca", "target_dim": 512, "quant_method": "binary", "dr_seed": seed, "retention": pca})
        rows.append({"backbone": "b", "dr_method": "pca_ror", "target_dim": 512, "quant_method": "binary", "dr_seed": seed, "retention": ror})
    gain = ror_gain_seeds(pd.DataFrame(rows), ["binary"])
    assert gain.loc[0, "n_seeds"] == 2 and np.isclose(gain.loc[0, "gain"], 9.0)


def test_best_per_budget_takes_the_highest_score_at_or_under_the_budget():
    table = cells(
        [
            {"backbone": "mgte", "dr_method": "pca", "target_dim": 64, "quant_method": "binary", "bytes_per_vector": 8, "score": 0.2, "retention": 0.4},
            {"backbone": "mgte", "dr_method": "pca", "target_dim": 64, "quant_method": "equal_count_2", "bytes_per_vector": 16, "score": 0.3, "retention": 0.6},
            {"backbone": "mgte", "dr_method": "truncate", "target_dim": 128, "quant_method": "binary", "bytes_per_vector": 16, "score": 0.25, "retention": 0.5},
        ]
    )
    best = best_per_budget(table).set_index("budget")
    assert best.loc[8, "configuration"] == "PCA-64 / binary"
    assert best.loc[16, "configuration"] == "PCA-64 / EC-2" and best.loc[256, "configuration"] == "PCA-64 / EC-2"


def test_seed_counts_need_the_same_seeds_in_every_cell():
    import pytest

    from thesis_figures import Ledger, seed_counts

    def frame(seed_sets):
        rows = [{"backbone": "b", "dr_method": "none", "target_dim": 0, "quant_method": "none", "dr_seed": 42}]
        for dim, seeds in zip((64, 128), seed_sets):
            rows += [{"backbone": "b", "dr_method": "random_selection", "target_dim": dim, "quant_method": "none", "dr_seed": s} for s in seeds]
        rows.append({"backbone": "b", "dr_method": "truncate", "target_dim": 64, "quant_method": "none", "dr_seed": 42})
        # quantized cells keep the default seed only, as on BEIR: not seeded, not an error
        rows.append({"backbone": "b", "dr_method": "random_selection", "target_dim": 64, "quant_method": "binary", "dr_seed": 42})
        return pd.DataFrame(rows)

    ledger = Ledger()
    counts = seed_counts(ledger, {"beir": frame([(42, 1, 2), (42, 1, 2)])})
    assert counts["n_seeds"].tolist() == [3] and counts.loc[0, "seeds"] == "42 and 1, 2"  # truncation has one seed: skipped
    assert counts.loc[0, "quantizers"] == "fp32 only" and counts.loc[0, "n_cells"] == 2
    assert {row["macro"]: row["value"] for row in ledger.rows} == {"numSeedsBEIRRS": "3", "numSeedsBEIR": "3"}
    with pytest.raises(ValueError):
        seed_counts(Ledger(), {"beir": frame([(42, 1, 2), (42, 1)])})


def test_points_never_print_negative_zero():
    from thesis_figures import pp

    assert pp(-0.003) == "+0.0" and pp(-0.06) == "-0.1" and pp(2.0) == "+2.0"


def test_pca_mean_gain_is_restored_minus_subtracted_for_cells_scored_both_ways():
    from thesis_figures import pca_mean_gain

    base = {"backbone": "b", "target_dim": 64, "quant_method": "none"}
    restored = pd.DataFrame([{**base, "dr_method": "pca", "retention": 0.90}, {**base, "dr_method": "truncate", "retention": 0.70},
                             {**base, "dr_method": "pca_ror", "quant_method": "binary", "retention": 0.60}])
    subtracted = pd.DataFrame([{**base, "dr_method": "pca", "retention": 0.85}, {**base, "dr_method": "truncate", "retention": 0.70}])
    gain = pca_mean_gain(restored, subtracted)
    assert len(gain) == 1 and gain.loc[0, "dr_method"] == "pca" and np.isclose(gain.loc[0, "gain"], 5.0)


def nonretrieval_rows():
    """Classification task A has two subsets, B one; STS one task. Raw baselines are
    unnormalized (fp16 scores above them), as for mDenseOn."""
    rows = []

    def add(task_type, task, subset, baseline, fp16, binary):
        common = {"backbone": "b", "task_type": task_type, "task": task, "split": "test", "subset": subset, "dr_seed": np.nan, "pca_mean": np.nan}
        rows.append({**common, "kind": "baseline", "dr_method": "none", "target_dim": np.nan, "quant_method": "none", "bytes_per_vector": 4096, "main_score": baseline})
        rows.append({**common, "kind": "cell", "dr_method": "none", "target_dim": np.nan, "quant_method": "fp16", "bytes_per_vector": 2048, "main_score": fp16})
        rows.append({**common, "kind": "cell", "dr_method": "none", "target_dim": np.nan, "quant_method": "binary", "bytes_per_vector": 128, "main_score": binary})

    add("Classification", "A", "en", 0.80, 0.81, 0.81 * 0.9)
    add("Classification", "A", "de", 0.60, 0.61, 0.61 * 0.9)
    add("Classification", "B", "default", 0.50, 0.50, 0.50 * 0.7)
    add("Clustering", "C", "default", 0.40, 0.40, 0.36)
    add("STS", "S", "default", 0.80, 0.80, 0.78)
    return pd.DataFrame(rows)


def test_task_types_average_subsets_first_and_take_classification_against_fp16():
    from thesis_figures import task_type_cells

    cells = task_type_cells(nonretrieval_rows()).set_index(["task_type", "quant_method"])
    binary = cells.loc[("Classification", "binary")]
    assert np.isclose(binary["retention"], 0.8)  # mean over tasks (0.9, 0.7), not over three rows
    assert np.isclose(cells.loc[("Classification", "fp16"), "retention"], 1.0)
    raw = np.mean([np.mean([0.81 / 0.80, 0.61 / 0.60]), 1.0])
    assert np.isclose(cells.loc[("Classification", "fp16"), "raw_retention"], raw)
    assert np.isclose(cells.loc[("STS", "binary"), "retention"], 0.78 / 0.80)  # STS keeps the fp32 baseline
    assert binary["n_tasks"] == 2


def test_expansion_cost_prices_every_call_and_times_the_expansion_used(tmp_path):
    import json

    from thesis_figures import expansion_cost

    folder = tmp_path / "abc"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps({"model": "m", "generation": 1}))
    calls = [
        {"query_id": "q1", "status": "incomplete", "latency_s": 9.0, "input_tokens": 100, "output_tokens": 1024},
        {"query_id": "q1", "status": "completed", "latency_s": 2.0, "input_tokens": 100, "output_tokens": 300},
        {"query_id": "q2", "status": "completed", "latency_s": 4.0, "input_tokens": 100, "output_tokens": 400},
    ]
    (folder / "expansions.jsonl").write_text("\n".join(json.dumps(c) for c in calls) + "\n")
    cost = expansion_cost(["abc"], root=tmp_path).iloc[0]
    assert cost["calls"] == 3 and cost["queries"] == 2 and cost["generation"] == 1
    assert np.isclose(cost["cost_usd"], (300 * 0.10 + 1724 * 0.50) / 1e6)
    assert np.isclose(cost["latency_median_s"], 3.0) and cost["output_tokens_median"] == 350
