"""Run-identity invariants.

The hash names a directory holding GPU-hours of embeddings. Two properties have to
hold or the cache is worse than useless: equal configs must hash equally *across
processes*, and unequal configs must hash differently.
"""

import json
import subprocess
import sys

import pytest

from geopres_grid.identity import (
    BITS_PER_DIM,
    EncodeConfig,
    PostProcConfig,
    RunId,
    canonical_json,
    config_hash,
    external_code_repos,
    minor_version,
)


def make_encode(**overrides) -> EncodeConfig:
    base = dict(
        model_id="Alibaba-NLP/gte-multilingual-base",
        revision="9bbca17d9273fd0d03d5725c7a4b0f6b45142062",
        max_seq_length=512,
        dtype="float32",
        prompts=(),
        sentence_transformers_version="5.7",
        transformers_version="4.56",
    )
    base.update(overrides)
    return EncodeConfig(**base)


# --- determinism --------------------------------------------------------------


def test_same_config_same_hash():
    assert make_encode().hash == make_encode().hash


def test_hash_is_stable_across_processes():
    """The exit criterion. Python's built-in hash() is salted; ours must not be.

    Run under two different PYTHONHASHSEED values in fresh interpreters. If anything
    in the chain ever falls back to `hash()` or to unordered iteration, this fails.
    """
    script = (
        "from geopres_grid.identity import EncodeConfig;"
        "print(EncodeConfig("
        "model_id='m', revision='r', max_seq_length=512, dtype='float32',"
        "prompts=(('query','q: '),('document','d: ')),"
        "sentence_transformers_version='5.7', transformers_version='4.56',"
        ").hash)"
    )
    digests = set()
    for seed in ("0", "1", "12345"):
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
            check=True,
        )
        digests.add(result.stdout.strip())
    assert len(digests) == 1, f"hash varies with PYTHONHASHSEED: {digests}"


def test_prompt_ordering_does_not_change_the_hash():
    a = make_encode(prompts=(("query", "q: "), ("document", "d: ")))
    b = make_encode(prompts=(("document", "d: "), ("query", "q: ")))
    assert a.hash == b.hash


def test_canonical_json_sorts_keys():
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'


# --- sensitivity --------------------------------------------------------------


@pytest.mark.parametrize(
    "override",
    [
        {"max_seq_length": 8192},
        {"revision": "0" * 40},
        {"dtype": "float16"},
        {"prompts": (("query", "query: "),)},
        {"transformers_version": "5.16"},
        {"sentence_transformers_version": "5.6"},
        {"external_code_revisions": (("Alibaba-NLP/new-impl", "abc123"),)},
    ],
    ids=lambda o: next(iter(o)),
)
def test_meaningful_changes_change_the_hash(override):
    assert make_encode().hash != make_encode(**override).hash


@pytest.mark.parametrize(
    "override",
    [{"torch_version": "9.9.9"}, {"device": "cuda"}, {"batch_size": 4096}],
    ids=lambda o: next(iter(o)),
)
def test_recorded_only_fields_do_not_change_the_hash(override):
    """Provenance that moves low-order bits must not fragment the cache."""
    assert make_encode().hash == make_encode(**override).hash


def test_meta_is_json_serialisable_and_carries_provenance():
    meta = make_encode(device="cuda", torch_version="2.13.0", batch_size=256).to_meta()
    round_tripped = json.loads(json.dumps(meta))
    assert round_tripped["encode_config_hash"] == make_encode().hash
    assert round_tripped["recorded_only"]["device"] == "cuda"


# --- the remote-code gap ------------------------------------------------------


def test_external_code_repos_distinguishes_in_repo_from_external():
    # mGTE loads from a different repo, which its own revision does not pin.
    assert external_code_repos(
        {"AutoModel": "Alibaba-NLP/new-impl--modeling.NewModel"}
    ) == ["Alibaba-NLP/new-impl"]
    # LFM2.5 loads from its own repo, so the model revision already pins it.
    assert external_code_repos({"AutoModel": "modeling_lfm2.Model"}) == []
    assert external_code_repos(None) == []


def test_unpinned_external_code_is_visible():
    pinned = make_encode(external_code_revisions=(("Alibaba-NLP/new-impl", "abc123"),))
    unpinned = make_encode(external_code_revisions=(("Alibaba-NLP/new-impl", ""),))
    assert not pinned.has_unpinned_code
    assert unpinned.has_unpinned_code


def test_minor_version_drops_the_patch():
    assert minor_version("4.56.0") == "4.56"
    assert minor_version("5.16.1") == "5.16"


# --- post-processing ----------------------------------------------------------


def test_postproc_rejects_incoherent_configs():
    with pytest.raises(ValueError, match="requires a target_dim"):
        PostProcConfig(dr_method="pca")
    with pytest.raises(ValueError, match="meaningless"):
        PostProcConfig(dr_method="none", target_dim=256)
    with pytest.raises(ValueError, match="Unknown dr_method"):
        PostProcConfig(dr_method="magic", target_dim=64)
    with pytest.raises(ValueError, match="Unknown quant_method"):
        PostProcConfig(quant_method="int3")


def test_seed_is_part_of_the_identity():
    """Same method, different rotation, different transform."""
    a = PostProcConfig(dr_method="pca_ror", target_dim=256, dr_seed=1)
    b = PostProcConfig(dr_method="pca_ror", target_dim=256, dr_seed=2)
    assert a.hash != b.hash


def test_symmetry_is_part_of_the_identity():
    shared = PostProcConfig(quant_method="int8", quant_symmetric=True)
    per_side = PostProcConfig(quant_method="int8", quant_symmetric=False)
    assert shared.hash != per_side.hash


@pytest.mark.parametrize(
    ("quant", "dim", "expected_bytes"),
    [
        ("none", 768, 3072),      # fp32 baseline
        ("fp16", 768, 1536),
        ("int8", 768, 768),
        ("int4", 768, 384),
        ("binary", 768, 96),      # 8 dimensions per byte
        ("binary", 100, 13),      # rounds up, not down
    ],
)
def test_storage_math(quant, dim, expected_bytes):
    cfg = PostProcConfig(
        dr_method="truncate" if dim != 768 else "none",
        target_dim=dim if dim != 768 else None,
        quant_method=quant,
    )
    assert cfg.bytes_per_vector(768) == expected_bytes


def test_compression_factor_matches_the_configuration_table():
    """The rows from the first meeting's table, recomputed from the config."""
    baseline = PostProcConfig()
    assert baseline.bytes_per_vector(768) == 3072
    assert baseline.compression_factor(768) == 1

    dr_only = PostProcConfig(dr_method="truncate", target_dim=192)
    assert dr_only.bytes_per_vector(768) == 768
    assert dr_only.compression_factor(768) == 4

    quant_only = PostProcConfig(quant_method="int8")
    assert quant_only.bytes_per_vector(768) == 768
    assert quant_only.compression_factor(768) == 4

    combined = PostProcConfig(dr_method="truncate", target_dim=384, quant_method="int8")
    assert combined.bytes_per_vector(768) == 384
    assert combined.compression_factor(768) == 8

    aggressive = PostProcConfig(dr_method="pca", target_dim=256, quant_method="int4")
    assert aggressive.bytes_per_vector(768) == 128
    assert aggressive.compression_factor(768) == 24


def test_index_bytes_scales():
    cfg = PostProcConfig(dr_method="truncate", target_dim=256, quant_method="int8")
    assert cfg.index_bytes(768, 10_000_000) == 2_560_000_000


def test_every_quant_method_has_a_bit_width():
    for method in BITS_PER_DIM:
        PostProcConfig(quant_method=method)


def test_run_id_combines_both_halves():
    run = RunId(make_encode(), PostProcConfig(quant_method="int8"))
    assert run.value == f"{run.encode.hash}-{run.postproc.hash}"
    assert len(run.value) == 33


def test_config_hash_is_short_and_hex():
    digest = config_hash({"a": 1})
    assert len(digest) == 16
    int(digest, 16)
