"""The `nonretrieval` tier through the real MTEB evaluators, on injected data.

Precompute (the baseline pass with a model), a grid cell from the model-less cache,
and `load_results`, for one task of each type. What is specific to them: a
classification task encodes its training sample under `train`, outside its
`eval_splits`, and a legacy clustering task calls `encode` once per cluster set.
Nothing is downloaded; the task objects come from the pinned mteb.
"""

import mteb
import numpy as np
from datasets import Dataset, DatasetDict

from geopres_grid import analysis
from geopres_grid.backbones import get
from geopres_grid.cache import CachedBackbone
from geopres_grid.evaluation import PostProcessedBackbone, baseline_pass, evaluate_cell, load_tasks
from geopres_grid.identity import EncodeConfig, PostProcConfig
from geopres_grid.results import load_results


class FakeModel:
    def encode(self, texts, *, prompt=None, batch_size=32, **kwargs):
        rng = np.random.default_rng(sum(sum((prompt + text).encode()) for text in texts))
        return rng.normal(size=(len(texts), 8)).astype(np.float32)


def texts(count, prefix):
    return [f"{prefix} text {i}" for i in range(count)]


def tasks_with_data():
    sts, classification, clustering = load_tasks(["STS12", "ToxicConversationsClassification", "RedditClustering"])
    sts.dataset = {"default": DatasetDict({"test": Dataset.from_dict(
        {"sentence1": texts(20, "a"), "sentence2": texts(20, "b"), "score": [i % 5 for i in range(20)]}
    )})}
    labelled = lambda count, prefix: Dataset.from_dict({"text": texts(count, prefix), "label": [i % 2 for i in range(count)]})
    classification.dataset = {"default": DatasetDict({"train": labelled(100, "train"), "test": labelled(30, "test")})}
    clustering.dataset = {"default": DatasetDict({"test": Dataset.from_dict({
        # Three cluster sets sharing five sentences: one block, each sentence cached once.
        "sentences": [texts(20, f"set {s}") + texts(5, "shared") for s in range(3)],
        "labels": [[i % 3 for i in range(25)] for _ in range(3)],
    })})}
    for task in (sts, classification, clustering):
        task.data_loaded = True
    return sts, classification, clustering


def test_each_task_type_runs_from_precompute_to_retention(tmp_path):
    backbone = get("mgte")
    config = EncodeConfig(
        model_id="fake/model", revision="a" * 40, max_seq_length=512, dtype="float32",
        prompts=(("query", "query: "),), sentence_transformers_version="5.7", transformers_version="4.56",
        task_prompt_names=tuple(sorted(backbone.task_prompt_names.items())),
    )
    results = mteb.ResultCache(tmp_path / "results")
    tasks = tasks_with_data()

    with_model = CachedBackbone(FakeModel(), backbone, config, tmp_path / "cache")
    for task in tasks:
        baseline_pass(with_model, task, results_cache=results, overwrite_strategy="always")
    with_model.close()
    blocks = {
        str(path.parent.relative_to(with_model.encode_directory))
        for path in with_model.encode_directory.rglob("embeddings.npy")
    }
    assert blocks == {
        "STS12/test/default/query",
        "ToxicConversationsClassification/test/default/query",
        "ToxicConversationsClassification/train/default/query",
        "RedditClustering/test/default/query",
    }
    assert len(np.load(with_model.task_directory("RedditClustering") / "test/default/query/embeddings.npy")) == 65

    cold = CachedBackbone(None, backbone, config, tmp_path / "cache")  # a miss would raise
    for task in tasks:
        evaluate_cell(
            PostProcessedBackbone(cold, PostProcConfig(dr_method="pca", target_dim=4, quant_method="int4")),
            task,
            results_cache=results,
        )

    frame = load_results(tmp_path / "results")
    names = frame.drop_duplicates("task").set_index("task")[["task_type", "main_score_name"]]
    assert names.to_dict("index") == {
        "STS12": {"task_type": "STS", "main_score_name": "cosine_spearman"},
        "ToxicConversationsClassification": {"task_type": "Classification", "main_score_name": "accuracy"},
        "RedditClustering": {"task_type": "Clustering", "main_score_name": "v_measure"},
    }
    for task_type in ("STS", "Classification", "Clustering"):
        rows = analysis.scored_rows(frame, task_type=task_type)
        assert sorted(rows["kind"]) == ["baseline", "cell"]
        assert np.isfinite(rows["retention"]).all()


def test_fast_classification_labels_change_nothing_but_speed():
    """Same shuffled indices, same sample, same metrics as mteb 2.15.1's own methods,
    on a split with a label too rare to fill (the original then reads every row)."""
    from mteb.abstasks.classification import AbsTaskClassification

    (task,) = load_tasks(["AmazonCounterfactualClassification"])  # applies the patch
    fast_scores, fast_sample = AbsTaskClassification._calculate_scores, AbsTaskClassification._undersample_data
    assert fast_scores.geopres_fast
    rng = np.random.default_rng(0)
    for labels in (list(rng.integers(0, 3, 500)), [0] * 300 + [1] * 5 + [2] * 200):
        train = Dataset.from_dict({"text": texts(len(labels), "t"), "label": [int(label) for label in labels]})
        fast_idxs = slow_idxs = None
        for experiment in range(3):  # the shuffled order is carried from one experiment to the next
            fast, fast_idxs, fast_selected = fast_sample(task, train, experiment, fast_idxs)
            slow, slow_idxs, slow_selected = fast_sample.original(task, train, experiment, slow_idxs)
            assert fast_idxs == slow_idxs and fast_selected == slow_selected
            assert fast["label"][:] == slow["label"][:]
        y_test = train["label"]  # a lazy Column, as `_run_experiment` passes it
        y_pred = np.asarray(rng.integers(0, 3, len(labels)))
        assert fast_scores(task, y_test, y_pred) == fast_scores.original(task, y_test, y_pred)
