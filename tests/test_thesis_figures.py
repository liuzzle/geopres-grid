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
