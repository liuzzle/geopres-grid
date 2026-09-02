"""Registry invariants. No network, no model downloads.

`scripts/smoke_backbones.py` checks the registry against the real models; this
checks the registry against itself, so a typo fails directly
"""

import re

import pytest

from geopres_grid import backbones

SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def test_keys_match_declared_keys():
    for key, backbone in backbones.BACKBONES.items():
        assert key == backbone.key


def test_model_ids_are_unique():
    model_ids = [b.model_id for b in backbones.all_backbones()]
    assert len(model_ids) == len(set(model_ids))


@pytest.mark.parametrize("backbone", backbones.all_backbones(), ids=lambda b: b.key)
def test_revision_is_a_full_commit_sha(backbone):
    """Never track a branch. WP-B hashes this into the run id."""
    assert SHA_RE.match(backbone.revision), (
        f"{backbone.key} revision {backbone.revision!r} is not a 40-char sha"
    )


@pytest.mark.parametrize("backbone", backbones.all_backbones(), ids=lambda b: b.key)
def test_dimension_is_positive(backbone):
    assert backbone.native_dim > 0


@pytest.mark.parametrize("backbone", backbones.all_backbones(), ids=lambda b: b.key)
def test_primary_max_seq_length_is_reachable(backbone):
    """The shared max_seq_length must not exceed any model's native ceiling."""
    if backbone.native_max_seq_length is None:
        return
    assert backbones.PRIMARY_MAX_SEQ_LENGTH <= backbone.native_max_seq_length, (
        f"{backbone.key} caps at {backbone.native_max_seq_length}, below the shared "
        f"{backbones.PRIMARY_MAX_SEQ_LENGTH}"
    )


def test_lookup_by_key_and_by_model_id():
    assert backbones.get("mdenseon").model_id == "lightonai/mDenseOn"
    assert backbones.get("lightonai/mDenseOn").key == "mdenseon"
    with pytest.raises(KeyError):
        backbones.get("nope")


def test_asymmetric_prompt_detection():
    """The models this flag is wrong about are the ones the cache key corrupts."""
    # mGTE declares no non-empty prompt at all.
    assert not backbones.get("mgte").has_asymmetric_prompts
    # Differing query/document prefixes.
    assert backbones.get("mdenseon").has_asymmetric_prompts
    assert backbones.get("lfm25").has_asymmetric_prompts
    # Instruct-style query prompts only, so documents go unprefixed.
    assert backbones.get("harrier-270m").has_asymmetric_prompts
    assert backbones.get("harrier-06b").has_asymmetric_prompts


# --- Matryoshka support -------------------------------------------------------
# Two of the five backbones are MRL-trained. 


@pytest.mark.parametrize("backbone", backbones.all_backbones(), ids=lambda b: b.key)
def test_mrl_claim_is_cited(backbone):
    if backbone.mrl_dims:
        assert backbone.mrl_source, f"{backbone.key} claims MRL dims with no source"


@pytest.mark.parametrize("backbone", backbones.all_backbones(), ids=lambda b: b.key)
def test_mrl_dims_fit_the_model(backbone):
    for dim in backbone.mrl_dims or ():
        assert 0 < dim <= backbone.native_dim, (
            f"{backbone.key} lists MRL dim {dim} outside 1..{backbone.native_dim}"
        )


def test_mrl_dims_match_the_papers():
    # mGTE: D = {32k | k >= 1, 32k <= 768}
    assert backbones.get("mgte").mrl_dims == tuple(range(32, 769, 32))
    # mDenseOn: exactly the four dimensions named in appendix C.3
    assert backbones.get("mdenseon").mrl_dims == (128, 256, 512, 768)


def test_mrl_validity_check():
    mgte = backbones.get("mgte")
    mdenseon = backbones.get("mdenseon")
    assert mgte.mrl_valid(192) and mdenseon.mrl_valid(256)
    # 192 is a multiple of 32 but not one of mDenseOn's four trained dimensions.
    assert not mdenseon.mrl_valid(192)
    # A model with no documented MRL support is never MRL-valid.
    assert not backbones.get("lfm25").mrl_valid(256)


def test_shared_mrl_dims():
    """The dimensions where an MRL comparison across both models is legitimate."""
    assert backbones.shared_mrl_dims(["mgte", "mdenseon"]) == [128, 256, 512, 768]
    # Not every backbone has documented MRL support, so there is no set for all five.
    assert backbones.shared_mrl_dims() == []
