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
