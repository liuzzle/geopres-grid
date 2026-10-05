import mteb
import numpy as np
import pytest

from geopres_grid.backbones import get
from geopres_grid.cache import CachedBackbone
from geopres_grid.evaluation import BEIR_TASKS
from geopres_grid.evaluation import NANOBEIR_TASKS
from geopres_grid.evaluation import PostProcessedBackbone
from geopres_grid.evaluation import cached_blocks
from geopres_grid.evaluation import sample_rows
from geopres_grid.evaluation import task_names
from geopres_grid.identity import EncodeConfig
from geopres_grid.identity import PostProcConfig
from geopres_grid.quantizers import fake_quantize


def benchmark_tasks(name):
    return tuple(task.metadata.name for task in mteb.get_benchmark(name).tasks)


def test_tier_zero_is_small_and_mixed():
    assert task_names("tier0") == ("NanoArguAnaRetrieval", "STSBenchmark")


def test_tier_one_is_mtebs_nanobeir_benchmark():
    assert task_names("tier1") == NANOBEIR_TASKS
    assert NANOBEIR_TASKS == benchmark_tasks("NanoBEIR")


def test_tier_two_is_full_beir():
    """Meeting 16.09 §4: the fifteen BEIR datasets, not the six-task subset."""
    assert task_names("tier2") == BEIR_TASKS
    assert BEIR_TASKS == benchmark_tasks("BEIR")


@pytest.mark.parametrize("tier", ["tier0", "tier1", "tier2"])
def test_every_tier_task_resolves_in_the_pinned_mteb(tier):
    names = task_names(tier)
    assert tuple(task.metadata.name for task in mteb.get_tasks(tasks=list(names))) == names


def test_sample_rows_is_seeded_bounded_and_spans_blocks():
    blocks = [np.arange(10, dtype=np.float32)[:, None], np.arange(10, 25, dtype=np.float32)[:, None]]

    sample = sample_rows(blocks, max_rows=8, seed=3)

    assert sample.shape == (8, 1)
    assert len(np.unique(sample)) == 8
    np.testing.assert_array_equal(sample, sample_rows(blocks, max_rows=8, seed=3))
    np.testing.assert_array_equal(
        sample_rows(blocks, max_rows=100, seed=3)[:, 0], np.arange(25, dtype=np.float32)
    )


def write_block(root, split, side, values):
    directory = root / split / "default" / side
    directory.mkdir(parents=True)
    np.save(directory / "embeddings.npy", np.asarray(values, dtype=np.float32))


def test_cached_blocks_groups_by_side_and_filters_splits(tmp_path):
    write_block(tmp_path, "test", "query", np.ones((2, 3)))
    write_block(tmp_path, "test", "document", np.ones((4, 3)))
    write_block(tmp_path, "dev", "document", np.ones((5, 3)))

    blocks = cached_blocks(tmp_path, splits=["test"])

    assert [len(block) for block in blocks["query"]] == [2]
    assert [len(block) for block in blocks["document"]] == [4]
    with pytest.raises(FileNotFoundError, match="warm-up"):
        cached_blocks(tmp_path / "missing")


class FakeModel:
    def encode(self, texts, *, prompt=None, batch_size=32, **kwargs):
        rng = np.random.default_rng(sum(sum((prompt + text).encode()) for text in texts))
        return rng.normal(size=(len(texts), 8)).astype(np.float32)

    def similarity(self, embeddings1, embeddings2):
        return np.asarray(embeddings1) @ np.asarray(embeddings2).T


class FakeDataset(list):
    pass


class FakeInputs:
    def __init__(self, rows):
        self.dataset = FakeDataset(rows)


class FakeMetadata:
    name = "FakeRetrieval"


def make_cached(tmp_path):
    config = EncodeConfig(
        model_id="fake/model",
        revision="a" * 40,
        max_seq_length=512,
        dtype="float32",
        prompts=(("query", "query: "), ("document", "document: ")),
        sentence_transformers_version="5.7",
        transformers_version="4.56",
    )
    return CachedBackbone(FakeModel(), get("mdenseon"), config, tmp_path)


def encode(model, rows, side):
    return model.encode(
        FakeInputs(rows),
        task_metadata=FakeMetadata(),
        hf_split="test",
        hf_subset="default",
        prompt_type=side,
    )


QUERIES = [{"id": f"q{i}", "text": f"query {i}"} for i in range(4)]
DOCUMENTS = [{"id": f"d{i}", "text": f"document {i}"} for i in range(40)]


def warm(cached):
    encode(cached, QUERIES, "query")
    encode(cached, DOCUMENTS, "document")
    return cached_blocks(cached.task_directory(FakeMetadata.name))


def test_cell_applies_the_fitted_pipeline_and_table(tmp_path):
    cached = make_cached(tmp_path)
    blocks = warm(cached)
    cell = PostProcessedBackbone(
        cached, PostProcConfig(dr_method="pca_ror", target_dim=4, quant_method="equal_count_4")
    )

    with pytest.raises(RuntimeError, match="fit_task"):
        encode(cell, DOCUMENTS, "document")
    fitted = cell.fit_task(FakeMetadata.name, blocks)
    result = encode(cell, DOCUMENTS, "document")

    raw = encode(cached, DOCUMENTS, "document")
    expected = fake_quantize(fitted.pipeline.transform(raw), fitted.quantizers["document"])
    np.testing.assert_array_equal(result, expected)
    assert result.shape == (40, 4)
    assert fitted.quantizers["query"] is fitted.quantizers["document"]


def test_per_side_calibration_fits_one_table_per_side(tmp_path):
    cached = make_cached(tmp_path)
    cell = PostProcessedBackbone(cached, PostProcConfig(quant_method="int4", quant_symmetric=False))

    fitted = cell.fit_task(FakeMetadata.name, warm(cached))

    assert fitted.quantizers["query"] is not fitted.quantizers["document"]


def test_each_cell_gets_its_own_result_slot(tmp_path):
    cached = make_cached(tmp_path)
    int8 = PostProcessedBackbone(cached, PostProcConfig(quant_method="int8"))
    int4 = PostProcessedBackbone(cached, PostProcConfig(quant_method="int4"))

    assert int8.mteb_model_meta.revision == int8.run_id.value
    assert len({cached.mteb_model_meta.revision, int8.mteb_model_meta.revision, int4.mteb_model_meta.revision}) == 3
