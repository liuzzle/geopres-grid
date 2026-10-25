"""`scripts/seed_check.py` on a small synthetic results table."""

import json

import numpy as np
import pandas as pd
import pytest

from seed_check import main, per_seed_cells, seed_spread, seeded_gain, seeded_rows

TASKS = ("NanoArguAnaRetrieval", "NanoNFCorpusRetrieval")
BASELINE = {"NanoArguAnaRetrieval": 0.5, "NanoNFCorpusRetrieval": 0.4}


def row(task, score, dr_method="none", target_dim=None, quant_method="none", dr_seed=None, kind="cell"):
    return {
        "source": "geopres-grid", "backbone": "mgte", "kind": kind, "run_id": "x",
        "dr_method": dr_method, "target_dim": target_dim, "dr_seed": dr_seed, "pca_mean": None,
        "quant_method": quant_method, "bits_per_dim": 8 if quant_method == "int8" else 32,
        "bytes_per_vector": 64 if target_dim else 768, "compression_factor": 1.0,
        "task": task, "split": "train", "subset": "default", "main_score": score, "task_type": "Retrieval",
    }


def results() -> pd.DataFrame:
    """Random selection at 64 dims retains 0.80 / 0.82 / 0.84 under seeds 42 / 1 / 2;
    random projection 0.90 under every seed; truncation 0.85."""
    rows = []
    for task, base in BASELINE.items():
        rows.append(row(task, base, kind="baseline"))
        rows += [row(task, base, quant_method=q) for q in ("fp16", "int8")]
        for quant in ("none", "int8"):
            rows.append(row(task, 0.85 * base, "truncate", 64, quant))
            for seed, retention in ((None, 0.80), (1, 0.82), (2, 0.84)):
                rows.append(row(task, retention * base, "random_selection", 64, quant, seed))
                rows.append(row(task, 0.90 * base, "random_projection", 64, quant, seed))
    rows.append(row("NanoArguAnaRetrieval", 0.4, "pca", 64) | {"pca_mean": "restore"})
    rows.append(row("ArguAna", 0.3, kind="baseline"))  # tier 2: left out
    return pd.DataFrame(rows)


def test_spread_over_seeds_and_the_default_seeds_place():
    rows = seeded_rows(results(), "tier1")
    assert set(rows["task"]) == set(TASKS) and "pca" not in set(rows["dr_method"])
    per_seed = per_seed_cells(rows)
    spread = seed_spread(per_seed, rows).set_index(["dr_method", "quant_method"])
    selection = spread.loc[("random_selection", "none")]
    assert selection["n_seeds"] == 3 and selection["n_tasks_min"] == 2
    assert selection["mean"] == pytest.approx(82.0)
    assert selection["std"] == pytest.approx(2.0)
    assert selection["default_minus_mean"] == pytest.approx(-2.0)
    assert selection["default_rank"] == 3
    assert spread.loc[("random_projection", "none"), "std"] == pytest.approx(0.0, abs=1e-9)
    assert "truncate" not in spread.index.get_level_values("dr_method")


def test_a_deterministic_side_is_compared_with_every_seed():
    per_seed = per_seed_cells(seeded_rows(results(), "tier1"))
    gain = seeded_gain(per_seed, "t", "dr_method", "truncate", "random_selection").set_index("quant_method")
    assert gain.loc["none", "n_seeds"] == 3
    assert gain.loc["none", "gain_mean"] == pytest.approx(3.0)
    assert gain.loc["none", "gain_default"] == pytest.approx(5.0)
    assert gain.loc["none", "sign_flips"] == 0
    flipped = seeded_gain(per_seed, "t", "dr_method", "random_selection", "truncate").loc[0]
    assert flipped["gain_mean"] == pytest.approx(-3.0) and not flipped["default_flips"]


def test_main_writes_every_table(tmp_path):
    source = tmp_path / "results.csv"
    results().to_csv(source, index=False)
    assert main([str(source)]) == 0
    out = tmp_path / "analysis"
    for name in ("seed_cells", "seed_spread", "comparisons", "ranking", "winners", "frontier"):
        assert (out / f"{name}.csv").exists()
    summary = json.loads((out / "summary.json").read_text())
    assert summary["seeds"] == [1, 2, 42]
    assert summary["winners_changed"] == 0  # random projection wins under every seed
    assert np.isfinite(summary["std_points"]["max"])


def test_a_single_seed_is_refused(tmp_path):
    frame = results()
    source = tmp_path / "results.csv"
    frame[frame["dr_seed"].isna()].to_csv(source, index=False)
    assert main([str(source)]) == 1
