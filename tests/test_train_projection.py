import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from geopres_grid.identity import PostProcConfig, weights_id
from geopres_grid.reducers import PostProcessingPipeline
from geopres_grid.train_projection import holdout, projection_weights, train_projection, training_rows
from geopres_grid.trainer import EmbeddingsDataset


def write_block(root, side, values, split="train"):
    directory = root / split / "default" / side
    directory.mkdir(parents=True)
    np.save(directory / "embeddings.npy", np.asarray(values, dtype=np.float32))
    pd.DataFrame({"row_idx": range(len(values)), "text_hash": [str(r) for r in range(len(values))]}).to_parquet(
        directory / "ids.parquet", index=False
    )


def test_training_rows_pool_both_sides_cap_each_task_and_unit_norm(tmp_path):
    rng = np.random.default_rng(0)
    small, large = tmp_path / "Small", tmp_path / "Large"
    write_block(small, "query", 3 * rng.normal(size=(2, 4)))
    write_block(small, "document", 3 * rng.normal(size=(5, 4)))
    write_block(large, "query", 3 * rng.normal(size=(10, 4)))
    write_block(large, "document", 3 * rng.normal(size=(40, 4)))

    rows = training_rows([small, large], max_rows_per_task=20, seed=0)

    assert rows.shape == (7 + 20, 4)
    np.testing.assert_allclose(np.linalg.norm(rows, axis=1), 1.0, rtol=1e-6)
    np.testing.assert_array_equal(rows, training_rows([small, large], max_rows_per_task=20, seed=0))


def test_holdout_is_a_disjoint_split_of_every_row():
    values = np.arange(50, dtype=np.float32)[:, None]

    train, val = holdout(values, 0.1, seed=0)

    assert (len(train), len(val)) == (45, 5)
    assert sorted(np.concatenate([train, val])[:, 0]) == list(range(50))


def test_saved_weights_are_what_geopres_applies_and_name_the_run():
    """`nn.Linear.weight` is `(target, source)`, the layout `GeoPres` applies as `X @ W.T`."""
    torch.manual_seed(0)
    projection = nn.Linear(8, 3, bias=False)
    weights = projection_weights(projection)
    values = np.random.default_rng(0).normal(size=(6, 8)).astype(np.float32)
    config = PostProcConfig(dr_method="geopres", target_dim=3, dr_weights_id=weights_id(weights))

    pipeline = PostProcessingPipeline(config, 8, geopres_weights=weights)
    pipeline.fit(values)

    unit = values / np.linalg.norm(values, axis=1, keepdims=True)
    expected = projection(torch.from_numpy(unit)).detach().numpy()
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    np.testing.assert_allclose(pipeline.transform(values), expected, rtol=1e-5, atol=1e-6)
    # Different trained weights, different run (`test_identity` covers the hash itself).
    assert PostProcConfig(dr_method="geopres", target_dim=3, dr_weights_id=weights_id(weights + 1)).hash != config.hash


def test_training_on_cached_rows_lowers_the_held_out_loss(tmp_path):
    rng = np.random.default_rng(0)
    # Low-rank data, so a 4-dim projection can preserve its distances.
    rows = (rng.normal(size=(256, 4)) @ rng.normal(size=(4, 16))).astype(np.float32)
    rows /= np.linalg.norm(rows, axis=1, keepdims=True)
    train, val = holdout(rows, 0.25, seed=0)
    torch.manual_seed(0)
    projection = nn.Linear(16, 4, bias=False)
    before = projection_weights(projection)

    state = train_projection(
        projection, "test", EmbeddingsDataset(torch.from_numpy(train)), EmbeddingsDataset(torch.from_numpy(val)),
        train_batch_size=64, val_batch_size=len(val), backbone_model="test", spearman=False,
        spearman_local_or_global="local", epochs=30, output_path=str(tmp_path),
        eval_after_training=False, eval_steps=20, early_stopping_patience=100,
    )

    losses = [entry["eval_loss"] for entry in state.log_history if "eval_loss" in entry]
    assert losses[-1] < losses[0]
    assert state.best_metric == min(losses)
    assert not np.array_equal(projection_weights(projection), before)
    assert (tmp_path / "best_model.pt").exists()


def test_main_trains_from_the_cache_and_writes_weights_geopres_loads(tmp_path):
    import json

    from geopres_grid.backbones import PRIMARY_MAX_SEQ_LENGTH, get
    from geopres_grid.cache import backbone_cache_root
    from geopres_grid.identity import EncodeConfig
    from geopres_grid.reducers import GeoPres
    from geopres_grid.train_projection import main

    backbone = get("mgte")
    encode = EncodeConfig(
        model_id=backbone.model_id, revision=backbone.revision, max_seq_length=PRIMARY_MAX_SEQ_LENGTH,
        dtype="float32", prompts=(), sentence_transformers_version="5.7", transformers_version="4.56",
        task_prompt_names=tuple(sorted(backbone.task_prompt_names.items())),
    )
    task = backbone_cache_root(tmp_path / "cache", backbone) / encode.hash / "FakeRetrieval"
    rng = np.random.default_rng(0)
    for side, n in (("query", 8), ("document", 72)):
        write_block(task, side, rng.normal(size=(n, backbone.native_dim)))
        (task / "train" / "default" / side / "meta.json").write_text(json.dumps(encode.to_meta()))

    main([
        "--backbone", "mgte", "--target_dim", "4", "--task", "FakeRetrieval",
        "--cache_root", str(tmp_path / "cache"), "--output_root", str(tmp_path / "trained"),
        "--epochs", "2", "--train_batch_size", "16", "--eval_steps", "2", "--device", "cpu",
    ])

    (output,) = (tmp_path / "trained" / "mgte").iterdir()
    record = json.loads((output / "training.json").read_text())
    weights = GeoPres.load(output / "weights.npy").weights
    assert weights.shape == (4, backbone.native_dim)
    assert record["weights_id"] == weights_id(weights)
    assert (record["train_rows"], record["val_rows"], record["encode_hash"]) == (72, 8, encode.hash)
