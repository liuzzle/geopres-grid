"""Turn MTEB result files into analysis tables.

Upstream GeoPres wrote MTEB's own result files, one cache directory per method
(`<method>/<backbone>/results/<model name>/<revision>/<Task>.json`), encoded the
configuration in the model name (`<backbone>_reduced_<dim>_pca`, ...), and merged
everything into one wide CSV keyed by that name, reading only the `test` split.
Here a result slot is named by a hash (`results/geopres-grid__<key>/<run id>/`), so
the configuration cannot be parsed back out of the path. It is written next to the
results instead, as a run record in `runs/<run id>.json`, and joined back in here.

Two tables come out:

- `load_results`: tidy, one row per (run, task, split, subset), configuration in
  columns. The format for analysis and plots.
- `comparison_table`: upstream's wide layout, one row per run and one column per
  task plus per-task-type averages. The format for thesis tables.

`load_upstream_results` reads an upstream GeoPres results tree into the same tidy
schema, so old and new numbers can sit in one frame. They are not directly
comparable: upstream forced `eval_splits=["test"]` (wrong for MSMARCO, whose split
is `dev`), fitted PCA once on a general corpus (`fit_pca.py`, default C4) where we
fit per task (Kisako et al. §3.5), and shares only mGTE among its backbones.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd

from geopres_grid.backbones import all_backbones

RESULTS_DIRECTORY = "results"
RUNS_DIRECTORY = "runs"
MODEL_PREFIX = "geopres-grid__"

CONFIG_COLUMNS = (
    "source",
    "backbone",
    "kind",
    "run_id",
    "dr_method",
    "target_dim",
    "dr_seed",
    "quant_method",
    "quant_symmetric",
    "output_dim",
    "bits_per_dim",
    "bytes_per_vector",
    "compression_factor",
)


def write_run_record(results_root: str | Path, record: dict[str, Any]) -> Path:
    """Store the configuration of one result slot as `runs/<revision>.json`."""
    directory = Path(results_root) / RUNS_DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{record['revision']}.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _rows_from_result(result: dict[str, Any], base: dict[str, Any], metrics: tuple[str, ...]):
    for split, entries in result["scores"].items():
        for entry in entries:
            row = {
                **base,
                "task": result["task_name"],
                "split": split,
                "subset": entry.get("hf_subset", "default"),
                "main_score": entry["main_score"],
                "mteb_version": result.get("mteb_version"),
                "dataset_revision": result.get("dataset_revision"),
            }
            for metric in metrics:
                row[metric] = entry.get(metric)
            yield row


def _config_columns(record: dict[str, Any]) -> dict[str, Any]:
    postproc = record.get("postproc") or {}
    return {
        "source": "geopres-grid",
        "backbone": record["backbone"],
        "kind": record["kind"],
        "run_id": record["revision"],
        "dr_method": postproc.get("dr_method", "none"),
        "target_dim": postproc.get("target_dim"),
        "dr_seed": postproc.get("dr_seed"),
        "quant_method": postproc.get("quant_method", "none"),
        "quant_symmetric": postproc.get("quant_symmetric", True),
        "output_dim": record["output_dim"],
        "bits_per_dim": record["bits_per_dim"],
        "bytes_per_vector": record["bytes_per_vector"],
        "compression_factor": record["compression_factor"],
    }


def _add_task_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    """Task type and main-score name from the pinned mteb, for tasks it knows."""
    import mteb

    known: dict[str, tuple[str, str]] = {}
    for name in sorted(set(frame["task"])):
        try:
            metadata = mteb.get_tasks(tasks=[name])[0].metadata
        except (KeyError, IndexError, ValueError):
            continue
        known[name] = (metadata.type, metadata.main_score)
    frame["task_type"] = [known.get(task, (None, None))[0] for task in frame["task"]]
    frame["main_score_name"] = [known.get(task, (None, None))[1] for task in frame["task"]]
    return frame


def load_results(results_root: str | Path, *, metrics: tuple[str, ...] = ()) -> pd.DataFrame:
    """Tidy table of every result slot under `results_root`, joined to its run record.

    `metrics` adds further per-split metrics by name (e.g. `recall_at_100`). Raises
    if a result slot has no run record: a score without its configuration is not a
    usable result.
    """
    root = Path(results_root)
    records = {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in (root / RUNS_DIRECTORY).glob("*.json")
    }
    rows: list[dict[str, Any]] = []
    unregistered: set[str] = set()
    for path in sorted((root / RESULTS_DIRECTORY).glob(f"{MODEL_PREFIX}*/*/*.json")):
        if path.name == "model_meta.json":
            continue
        revision = path.parent.name
        record = records.get(revision)
        if record is None:
            unregistered.add(f"{path.parent.parent.name}/{revision}")
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(_rows_from_result(result, _config_columns(record), metrics))
    if unregistered:
        raise ValueError(
            f"result slots without a run record in {root / RUNS_DIRECTORY}: "
            f"{sorted(unregistered)}"
        )
    frame = pd.DataFrame(rows, columns=[*CONFIG_COLUMNS, "task", "split", "subset", "main_score",
                                        "mteb_version", "dataset_revision", *metrics])
    return _add_task_metadata(frame)


def comparison_table(frame: pd.DataFrame, *, tasks: list[str] | None = None) -> pd.DataFrame:
    """Upstream's wide layout: a row per run, a column per task, per-type averages.

    Each task's score is its main score averaged over splits and subsets. The
    `AVG_<TYPE>` columns average the tasks of one type; `AVG_ALL` averages those,
    as upstream's `**AVG_MTEB**` did, so no task type outweighs another by count.
    Unlike upstream's, an average is NaN when one of its tasks is missing for that
    run: a mean over whatever happened to finish is not comparable across rows.
    """
    data = frame if tasks is None else frame[frame["task"].isin(tasks)]
    index = [column for column in CONFIG_COLUMNS if column in data.columns]
    keys = data[index].astype(object).where(data[index].notna(), "-")
    wide = (
        data.assign(**{column: keys[column] for column in index})
        .groupby([*index, "task"], sort=True)["main_score"]
        .mean()
        .unstack("task")
    )
    ordered = [task for task in (tasks or sorted(wide.columns)) if task in wide.columns]
    wide = wide[ordered]
    types = data.drop_duplicates("task").set_index("task")["task_type"].reindex(ordered)
    averages = {}
    # Intrinsic rows are losses, not scores; upstream kept them out of its averages too.
    for task_type in sorted(set(types.dropna()) - {"Intrinsic"}):
        members = types.index[types == task_type]
        averages[f"AVG_{task_type.upper()}"] = wide[members].mean(axis=1, skipna=False)
    if averages:
        averages = pd.DataFrame(averages)
        averages.insert(0, "AVG_ALL", averages.mean(axis=1, skipna=False))
        wide = pd.concat([averages, wide], axis=1)
    return wide


# --- upstream GeoPres results ------------------------------------------------------

UPSTREAM_NAME = re.compile(r"^(?P<backbone>.+?)_reduced_(?P<dim>\d+)(?P<rest>.*)$")
UPSTREAM_METHODS = {
    "_pca": "pca",
    "_random_projection": "random_projection",
    "_random_selection": "random_selection",
    "_truncation": "truncate",
    "_autoencoder": "autoencoder",
}
"""Model-name suffixes of upstream's `baselines/eval_*.py`. A `_batch_...` suffix is
a trained GeoPres projection (`eval_model.py`)."""

UPSTREAM_INTRINSIC = ("spearman_loss", "angular_loss", "positional_loss")


def parse_upstream_model_name(name: str) -> dict[str, Any]:
    """Configuration encoded in an upstream result directory name."""
    match = UPSTREAM_NAME.match(name)
    if match is None:
        return {"model_id": name.replace("__", "/"), "dr_method": "none", "target_dim": None,
                "variant": ""}
    rest = match["rest"]
    if rest in UPSTREAM_METHODS:
        method, variant = UPSTREAM_METHODS[rest], ""
    elif rest.startswith("_batch_"):
        method, variant = "geopres", rest[1:]
    else:
        method, variant = "unknown", rest.lstrip("_")
    return {
        "model_id": match["backbone"].replace("__", "/"),
        "dr_method": method,
        "target_dim": int(match["dim"]),
        "variant": variant,
    }


def load_upstream_results(results_dir: str | Path) -> pd.DataFrame:
    """Read an upstream GeoPres `evaluation_results` tree into the tidy schema.

    Walks every `results/` directory as upstream's `compare_evaluation_results.py`
    did, but keeps every split instead of only `test`, keeps full precision, and
    turns `intrinsic.json` into rows of task type `Intrinsic` (losses: lower is
    better). Upstream evaluated fp32 only, so quantization columns are fp32.
    """
    keys = {backbone.model_id: backbone for backbone in all_backbones()}
    rows: list[dict[str, Any]] = []
    for root, directories, _ in os.walk(results_dir):
        if RESULTS_DIRECTORY not in directories:
            continue
        for model_directory in sorted(Path(root, RESULTS_DIRECTORY).iterdir()):
            if not model_directory.is_dir():
                continue
            parsed = parse_upstream_model_name(model_directory.name)
            backbone = keys.get(parsed["model_id"])
            source_dim = backbone.native_dim if backbone else None
            output_dim = parsed["target_dim"] or source_dim
            base = {
                "source": "geopres-upstream",
                "backbone": backbone.key if backbone else parsed["model_id"],
                "kind": "baseline" if parsed["dr_method"] == "none" else "cell",
                "run_id": model_directory.name,
                "dr_method": parsed["dr_method"],
                "target_dim": parsed["target_dim"],
                "dr_seed": None,
                "quant_method": "none",
                "quant_symmetric": True,
                "output_dim": output_dim,
                "bits_per_dim": 32,
                "bytes_per_vector": output_dim * 4 if output_dim else None,
                "compression_factor": source_dim / output_dim if source_dim and output_dim else None,
                "variant": parsed["variant"],
            }
            for path in sorted(model_directory.rglob("*.json")):
                if path.name == "model_meta.json":
                    continue
                result = json.loads(path.read_text(encoding="utf-8"))
                if path.name == "intrinsic.json":
                    for metric in UPSTREAM_INTRINSIC:
                        if result.get(metric) is not None:
                            rows.append({**base, "task": metric, "split": "test",
                                         "subset": "default", "main_score": result[metric],
                                         "task_type": "Intrinsic", "main_score_name": metric})
                    continue
                rows.extend(_rows_from_result(result, base, ()))
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    intrinsic = frame["task"].isin(UPSTREAM_INTRINSIC)
    scored = _add_task_metadata(
        frame[~intrinsic].drop(columns=["task_type", "main_score_name"], errors="ignore").copy()
    )
    return pd.concat([scored, frame[intrinsic]], ignore_index=True)
