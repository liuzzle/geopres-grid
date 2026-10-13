"""Version-pinned evaluation task tiers, and the evaluation path for one grid cell.

A grid cell is `RunId(encode, postproc)`. `PostProcessedBackbone` serves it to
MTEB: cached fp32 vectors -> normalize -> reduce -> normalize -> fake
quantization, with every fitted part fitted once per task before scoring, on that
task's own inputs -- the calibration protocol of Kisako, Tsukagoshi & Sasano
(arXiv:2606.01074, §3.5). The protocol is transductive; their Limitations section
says it "may overestimate performance" against fitting once on a separate
calibration corpus, and the write-up should carry the same caveat.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from geopres_grid.cache import CachedBackbone, run_model_meta, side_of
from geopres_grid.identity import PostProcConfig, RunId
from geopres_grid.precompute import leaf_tasks
from geopres_grid.quantizers import Quantizer, fake_quantize, fit_quantizer
from geopres_grid.reducers import PostProcessingPipeline
from geopres_grid.results import write_run_record

Tier = Literal["tier0", "tier1", "tier2", "nonretrieval", "mldr"]

NANOBEIR_TASKS: tuple[str, ...] = (
    "NanoArguAnaRetrieval",
    "NanoClimateFeverRetrieval",
    "NanoDBPediaRetrieval",
    "NanoFEVERRetrieval",
    "NanoFiQA2018Retrieval",
    "NanoHotpotQARetrieval",
    "NanoMSMARCORetrieval",
    "NanoNFCorpusRetrieval",
    "NanoNQRetrieval",
    "NanoQuoraRetrieval",
    "NanoSCIDOCSRetrieval",
    "NanoSciFactRetrieval",
    "NanoTouche2020Retrieval",
)
"""`mteb.get_benchmark("NanoBEIR")` in mteb 2.15.1. Their only split is `train`."""

BEIR_TASKS: tuple[str, ...] = (
    "TRECCOVID",
    "NFCorpus",
    "NQ",
    "HotpotQA",
    "FiQA2018",
    "ArguAna",
    "Touche2020",
    "CQADupstackRetrieval",
    "QuoraRetrieval",
    "DBPedia",
    "SCIDOCS",
    "FEVER",
    "ClimateFEVER",
    "SciFact",
    "MSMARCO",
)
"""`mteb.get_benchmark("BEIR")` in mteb 2.15.1: the fifteen public BEIR datasets
(Thakur et al., arXiv:2104.08663). Meeting 16.09 §4 replaced Konstantinos' six,
a strict and mostly expensive subset, with these. MSMARCO is scored on its
declared `dev` split; nothing here overrides `eval_splits`."""

NON_RETRIEVAL_TASKS: tuple[str, ...] = (
    "STS12",
    "STS13",
    "STS14",
    "STS15",
    "STS16",
    "STSBenchmark",
    "SICK-R",
    "AmazonCounterfactualClassification",
    "AmazonReviewsClassification",
    "ImdbClassification",
    "ToxicConversationsClassification",
    "AmazonPolarityClassification",
    "ArxivClusteringS2S",
    "RedditClustering",
    "StackExchangeClustering",
)
"""Konstantinos' non-retrieval set from upstream GeoPres (`eval_utils.py`), under
the same names -- not the `.v2` or hierarchical successors mteb 2.15.1 points to,
so the task definitions match upstream's. Seven STS, five classification, three
clustering. Added 07.10.2026 (Andrianos, answering 06.10 §7)."""

MLDR_TASKS: tuple[str, ...] = ("MultiLongDocRetrieval",)
"""MLDR (Chen et al., arXiv:2402.03216), thirteen languages of long documents.
Added 07.10.2026; at 512 tokens it measured truncation, so it waited on LFM2.5
leaving and `PRIMARY_MAX_SEQ_LENGTH` rising to 8192."""

TIER_TASKS: dict[Tier, tuple[str, ...]] = {
    "tier0": ("NanoArguAnaRetrieval", "STSBenchmark"),
    "tier1": NANOBEIR_TASKS,
    "tier2": BEIR_TASKS,
    "nonretrieval": NON_RETRIEVAL_TASKS,
    "mldr": MLDR_TASKS,
}
"""Task sets by cost. Tiers 0-2 grow by orders of magnitude and run on fewer
grid cells as they grow; `nonretrieval` and `mldr` are supplementary sets
beside tier 2 -- the first cheap, the second long-context."""

TASK_EVAL_SPLITS: dict[str, tuple[str, ...]] = {
    "AmazonCounterfactualClassification": ("test",),
    "AmazonReviewsClassification": ("test",),
    "MultiLongDocRetrieval": ("test",),
}
"""Overrides of a task's declared `eval_splits`, for the tasks that also declare
a validation/dev split. Upstream GeoPres scored every non-retrieval task on `test`
only; MLDR follows BEIR's convention of `test`. Every other task keeps its own
declaration -- NanoBEIR's only split is `train`, MSMARCO's is `dev`. All language
subsets are kept, as upstream did (no `languages` filter)."""


def task_names(tier: Tier) -> tuple[str, ...]:
    """Return the immutable task list for one evaluation tier."""
    return TIER_TASKS[tier]


def load_tasks(names: Any) -> list[Any]:
    """MTEB task objects for `names`, with `TASK_EVAL_SPLITS` applied.

    Every entry point goes through this, so precompute, evaluation and the cache
    estimate agree on which splits exist.
    """
    import mteb

    return [
        mteb.get_task(name, eval_splits=list(TASK_EVAL_SPLITS[name]) if name in TASK_EVAL_SPLITS else None)
        for name in names
    ]


# --- calibration ---------------------------------------------------------------

CALIBRATION_MAX_ROWS = 10_000
"""Kisako et al. §3.5 fit PCA on up to 10,000 embeddings sampled from the task's
warm-up cache, or on all of them when there are fewer. The same sample feeds the
quantization tables, which they fit on the reduced calibration embeddings."""

CALIBRATION_SEED = 0


def cached_blocks(
    task_directory: Path,
    *,
    splits: list[str] | None = None,
    subsets: list[str] | None = None,
) -> dict[str, list[np.ndarray]]:
    """Memory-mapped cache blocks of one task (`CachedBackbone.task_directory`), by side.

    Raises when the task has no cached vectors: calibration must come from a
    finished warm-up pass, never from whatever happened to be cached so far.
    """
    blocks: dict[str, list[np.ndarray]] = {"query": [], "document": []}
    for path in sorted(Path(task_directory).glob("*/*/*/embeddings.npy")):
        side_dir = path.parent
        subset, split = side_dir.parent.name, side_dir.parent.parent.name
        if splits and split not in splits:
            continue
        if subsets and subset not in subsets:
            continue
        blocks[side_dir.name].append(np.load(path, mmap_mode="r"))
    if not any(blocks.values()):
        raise FileNotFoundError(
            f"No cached embeddings under {task_directory}; "
            "run the warm-up pass (or scripts/precompute.py) first"
        )
    return blocks


def sample_rows(blocks: list[np.ndarray], max_rows: int, seed: int) -> np.ndarray:
    """Uniform sample without replacement over the rows of several blocks.

    Gathers per block with sorted indices, so a memory-mapped corpus is never
    concatenated into RAM.
    """
    sizes = [len(block) for block in blocks]
    total = sum(sizes)
    if total == 0:
        raise ValueError("cannot sample calibration rows from empty blocks")
    if total <= max_rows:
        chosen = np.arange(total)
    else:
        chosen = np.sort(np.random.default_rng(seed).choice(total, max_rows, replace=False))
    offsets = np.concatenate(([0], np.cumsum(sizes)))
    parts = []
    for block, start, stop in zip(blocks, offsets[:-1], offsets[1:]):
        local = chosen[(chosen >= start) & (chosen < stop)] - start
        if len(local):
            parts.append(np.asarray(block[local], dtype=np.float32))
    return np.vstack(parts)


# --- one grid cell ----------------------------------------------------------------


@dataclass
class FittedTask:
    """Everything fitted for one task: the shared DR pipeline and a table per side."""

    pipeline: PostProcessingPipeline
    quantizers: dict[str, Quantizer]


class PostProcessedBackbone:
    """MTEB encoder for one grid cell, reading through a `CachedBackbone`.

    DR is fitted on both sides together and always shared -- a projection that
    differed between queries and documents would not put them in one space. The
    quantization table is shared too (meeting 26.08, Kisako et al. §3.5) unless
    `quant_symmetric=False`, which fits one table per side for the symmetry
    ablation. Nothing is refitted inside `encode`.
    """

    def __init__(
        self,
        cached: CachedBackbone,
        config: PostProcConfig,
        *,
        geopres_weights: np.ndarray | None = None,
    ) -> None:
        self.cached = cached
        self.config = config
        self.geopres_weights = geopres_weights
        self.fitted: dict[str, FittedTask] = {}

    @property
    def run_id(self) -> RunId:
        return RunId(self.cached.encode_config, self.config)

    @property
    def mteb_model_meta(self) -> Any:
        """Results are stored under the run id, one slot per grid cell."""
        return run_model_meta(self.cached.backbone, self.run_id.value)

    def load_model(self) -> "PostProcessedBackbone":
        return self

    def similarity(self, embeddings1: Any, embeddings2: Any) -> Any:
        return self.cached.similarity(embeddings1, embeddings2)

    def similarity_pairwise(self, embeddings1: Any, embeddings2: Any) -> Any:
        return self.cached.similarity_pairwise(embeddings1, embeddings2)

    def fit_task(
        self,
        task_name: str,
        blocks: dict[str, list[np.ndarray]],
        *,
        max_rows: int = CALIBRATION_MAX_ROWS,
        seed: int = CALIBRATION_SEED,
    ) -> FittedTask:
        """Fit DR and quantization for one task from its cached blocks."""
        both = sample_rows(blocks["query"] + blocks["document"], max_rows, seed)
        pipeline = PostProcessingPipeline(
            self.config, both.shape[1], geopres_weights=self.geopres_weights
        )
        pipeline.fit(both)
        if self.config.quant_symmetric:
            shared = fit_quantizer(self.config, pipeline.transform(both))
            quantizers = {"query": shared, "document": shared}
        else:
            quantizers = {
                side: fit_quantizer(
                    self.config, pipeline.transform(sample_rows(rows, max_rows, seed))
                )
                for side, rows in blocks.items()
                if rows
            }
        fitted = FittedTask(pipeline, quantizers)
        self.fitted[task_name] = fitted
        return fitted

    def encode(
        self,
        inputs: Any,
        *,
        task_metadata: Any,
        hf_split: str,
        hf_subset: str,
        prompt_type: Any = None,
        **kwargs: Any,
    ) -> np.ndarray:
        fitted = self.fitted.get(task_metadata.name)
        if fitted is None:
            raise RuntimeError(
                f"{task_metadata.name} has no fitted post-processing; call fit_task first"
            )
        values = self.cached.encode(
            inputs,
            task_metadata=task_metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            prompt_type=prompt_type,
            **kwargs,
        )
        quantizer = fitted.quantizers[side_of(prompt_type)]
        return fake_quantize(fitted.pipeline.transform(values), quantizer)


def run_record(cached: CachedBackbone, config: PostProcConfig | None) -> dict[str, Any]:
    """What one result slot holds, for `results.load_results` to join back in.

    `config=None` is the warm-up pass: the backbone's raw fp32 output, which scores
    identically to `PostProcConfig()` under cosine.
    """
    source_dim = cached.backbone.native_dim
    record = {
        "backbone": cached.backbone.key,
        "model_id": cached.backbone.model_id,
        "encode": cached.encode_config.to_meta(),
    }
    if config is None:
        return {
            **record,
            "kind": "baseline",
            "revision": cached.encode_config.hash,
            "postproc": None,
            "output_dim": source_dim,
            "bits_per_dim": 32,
            "bytes_per_vector": source_dim * 4,
            "compression_factor": 1.0,
            "calibration": None,
        }
    return {
        **record,
        "kind": "cell",
        "revision": RunId(cached.encode_config, config).value,
        "postproc": asdict(config),
        "output_dim": config.output_dim(source_dim),
        "bits_per_dim": config.bits_per_dim,
        "bytes_per_vector": config.bytes_per_vector(source_dim),
        "compression_factor": config.compression_factor(source_dim),
        "calibration": {
            "max_rows": CALIBRATION_MAX_ROWS,
            "seed": CALIBRATION_SEED,
            "sample": "the task's cached inputs, both sides (Kisako et al. §3.5)",
        },
    }


def baseline_pass(
    cached: CachedBackbone,
    task: Any,
    *,
    results_cache: Any = None,
    encode_kwargs: dict[str, Any] | None = None,
    overwrite_strategy: str = "only-missing",
) -> Any:
    """Score one task through the plain cache wrapper: the backbone's fp32 baseline,
    stored under the encode hash with its run record.

    On a GPU node with a model loaded, this is how non-retrieval tasks are
    precomputed -- what STS, classification and clustering encode depends on
    MTEB's own sampling, so letting MTEB drive the wrapper is the only way to
    cache exactly the inputs evaluation will ask for. Precompute passes
    `overwrite_strategy="always"`, so an existing result cannot skip the encode
    and leave the cache cold.
    """
    import mteb

    cache = results_cache if results_cache is not None else mteb.ResultCache()
    write_run_record(cache.cache_path, run_record(cached, None))
    return mteb.evaluate(
        cached,
        task,
        cache=cache,
        encode_kwargs=encode_kwargs,
        overwrite_strategy=overwrite_strategy,
    )


def evaluate_cell(
    cell: PostProcessedBackbone,
    task: Any,
    *,
    results_cache: Any = None,
    encode_kwargs: dict[str, Any] | None = None,
) -> tuple[Any, Any]:
    """Warm up, calibrate and score one task for one grid cell.

    1. Warm-up pass through the plain cache wrapper (`baseline_pass`). It is the
       fp32 baseline of this backbone, stored under the encode hash; on a warm
       cache it reads only cached vectors, and when precompute already scored it,
       MTEB returns the stored result without encoding at all.
    2. Fit on each leaf task's cached inputs (Kisako et al. §3.5).
    3. Score the cell, stored under the run id.
    Each result slot gets its run record in `runs/` beside MTEB's `results/`.
    Returns the two `ModelResult`s, warm-up first. An identity cell
    (`PostProcConfig.is_identity`) would score exactly as the warm-up, in a
    second slot that duplicates the baseline row; it is skipped and the warm-up
    is returned twice.
    """
    import mteb

    cache = results_cache if results_cache is not None else mteb.ResultCache()
    baseline = baseline_pass(cell.cached, task, results_cache=cache, encode_kwargs=encode_kwargs)
    if cell.config.is_identity:
        return baseline, baseline
    for leaf in leaf_tasks(task):
        blocks = cached_blocks(
            cell.cached.task_directory(leaf.metadata.name),
            splits=list(leaf.eval_splits),
            subsets=list(leaf.hf_subsets),
        )
        cell.fit_task(leaf.metadata.name, blocks)
    write_run_record(cache.cache_path, run_record(cell.cached, cell.config))
    scored = mteb.evaluate(cell, task, cache=cache, encode_kwargs=encode_kwargs)
    return baseline, scored
