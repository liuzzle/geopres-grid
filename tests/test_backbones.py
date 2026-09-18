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


def test_prompt_names_are_resolved_per_side():
    """The query prompt is not always called "query".

    Both harriers name theirs `web_search_query`. sentence-transformers raises on an
    unknown prompt name (`base/model.py:303-309`), but the reference `CachedEncoder`
    catches it first and returns None, so wrapping the model turns that error into
    queries encoded with no prefix at all and a number that is quietly a few points
    low. A wrong-but-plausible number is the worst kind, so the side -> name mapping
    is a registry fact rather than something each call site guesses.
    """
    assert backbones.get("mgte").prompt_name_for("query") is None
    assert backbones.get("mgte").prompt_name_for("document") is None

    for key in ("mdenseon", "lfm25"):
        b = backbones.get(key)
        assert b.prompt_name_for("query") == "query"
        assert b.prompt_name_for("document") == "document"
        assert b.prompt_for("query") == "query: "
        assert b.prompt_for("document") == "document: "

    for key in ("harrier-270m", "harrier-06b"):
        b = backbones.get(key)
        assert b.prompt_name_for("query") == "web_search_query"
        # Not an omission: the model card encodes documents with no prompt.
        assert b.prompt_name_for("document") is None
        assert b.prompt_for("query").startswith("Instruct: Given a web search query")
        assert b.prompt_for("document") == ""


@pytest.mark.parametrize("backbone", backbones.all_backbones(), ids=lambda b: b.key)
def test_prompt_names_exist_in_the_prompts_dict(backbone):
    """A named prompt that the model does not declare would raise at encode time."""
    for side in ("query", "document"):
        name = backbone.prompt_name_for(side)
        if name is not None:
            assert backbone.prompts and name in backbone.prompts
    # Resolution must not raise for either side.
    backbone.prompt_for("query")
    backbone.prompt_for("document")


@pytest.mark.parametrize("backbone", backbones.all_backbones(), ids=lambda b: b.key)
def test_prompt_split_is_cited(backbone):
    assert backbone.prompt_source, f"{backbone.key} states a prompt split with no source"


def test_prompt_side_argument_is_validated():
    with pytest.raises(ValueError):
        backbones.get("mgte").prompt_name_for("passage")


@pytest.mark.parametrize("backbone", backbones.all_backbones(), ids=lambda b: b.key)
def test_prompted_text_is_what_a_cache_must_key_on(backbone):
    """Same text, two sides -- the keys must differ exactly when the prompts do.

    This is the C1 defect in one assertion: MTEB's `CachedEmbeddingWrapper` keys on
    `sha256(text)`, so on the four asymmetric backbones a query and a document with
    identical text share a cache entry and one of them gets the other's vector.
    """
    import hashlib

    text = "what is the capital of france"
    key_q = hashlib.sha256((backbone.prompt_for("query") + text).encode()).hexdigest()
    key_d = hashlib.sha256((backbone.prompt_for("document") + text).encode()).hexdigest()
    assert (key_q != key_d) == backbone.has_asymmetric_prompts


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


def test_mrl_support_has_been_investigated_for_every_backbone():
    """No backbone may sit in the 'nobody checked' state.

    This is the 02.09 lesson encoded as a test: silence in a model card was read
    as absence, and both models that were called non-MRL turned out to be
    MRL-trained per their papers.
    """
    assert backbones.unchecked_mrl() == []


@pytest.mark.parametrize("backbone", backbones.all_backbones(), ids=lambda b: b.key)
def test_mrl_claims_cite_a_source(backbone):
    """Both directions need a citation: the dimension set, and the negative."""
    if backbone.mrl_dims or backbone.mrl_checked:
        assert backbone.mrl_source, f"{backbone.key} states an MRL claim with no source"


def test_backbones_without_documented_mrl():
    """Checked 15.09.2026; truncation is a naive baseline for these three."""
    no_mrl = {b.key for b in backbones.all_backbones() if not b.mrl_dims}
    assert no_mrl == {"lfm25", "harrier-270m", "harrier-06b"}


@pytest.mark.parametrize("backbone", backbones.all_backbones(), ids=lambda b: b.key)
def test_every_backbone_has_an_environment(backbone):
    assert backbone.transformers_majors
    for major in backbone.transformers_majors:
        assert major in backbones.ENVIRONMENTS, (
            f"{backbone.key} claims transformers {major}.x, which no environment provides"
        )


def test_the_environment_split_is_exactly_mgte_and_mdenseon():
    """If this ever fails, the split has changed and envs/README.md is stale."""
    assert backbones.get("mgte").transformers_majors == (4,)
    assert backbones.get("mdenseon").transformers_majors == (5,)
    assert {b.key for b in backbones.runnable_backbones(4)} == {
        "mgte", "lfm25", "harrier-270m", "harrier-06b"
    }
    assert {b.key for b in backbones.runnable_backbones(5)} == {
        "mdenseon", "lfm25", "harrier-270m", "harrier-06b"
    }


def test_no_single_environment_runs_every_backbone():
    """The claim the second environment exists for. Stated as a test so that an
    upstream fix making it false shows up as a failure to celebrate."""
    for major in backbones.ENVIRONMENTS:
        assert len(backbones.runnable_backbones(major)) < len(backbones.all_backbones())


def test_require_runnable_names_the_fix():
    with pytest.raises(RuntimeError, match="envs/transformers5"):
        backbones.require_runnable(backbones.get("mdenseon"), major=4)
    with pytest.raises(RuntimeError, match="uv sync"):
        backbones.require_runnable(backbones.get("mgte"), major=5)
    # A backbone in the right environment passes silently.
    backbones.require_runnable(backbones.get("lfm25"), major=4)
    backbones.require_runnable(backbones.get("lfm25"), major=5)


def test_prompt_name_keys_fragment_the_cache_where_prompted_text_does_not():
    """Why the cache key is `sha256(prompted_text)` and not `(prompt_name, text)`.

    The reference `CachedEncoder` keys on the prompt *name*. LFM2.5 declares eight
    names -- `document`, `positive`, `negative_0..6` -- that all resolve to the same
    `"document: "` prefix, because they are the role names from its training script.
    Name-keying therefore stores up to eight identical vectors for one document and
    misses the cache seven times out of eight; hashing the prompted text collapses
    them to one entry. That is a throughput argument on top of the correctness
    argument in the test above, and it points the same way.
    """
    lfm = backbones.get("lfm25")
    document_prefix = lfm.prompt_for("document")
    aliases = [n for n, p in (lfm.prompts or {}).items() if p == document_prefix]
    assert len(aliases) > 1, "expected LFM2.5's training-time role aliases"
    assert {"document", "positive"} <= set(aliases)

    text = "Paris is the capital of France."
    by_name = {(name, text) for name in aliases}
    by_prompted_text = {document_prefix + text for _ in aliases}
    assert len(by_name) == len(aliases)
    assert len(by_prompted_text) == 1


# --- the empirical probe ------------------------------------------------------
# `mrl_dims` is what a vendor documents; `mrl_probe` is what was measured. They
# answer different questions and are allowed to disagree.


@pytest.mark.parametrize("backbone", backbones.all_backbones(), ids=lambda b: b.key)
def test_probe_results_carry_their_provenance(backbone):
    """A probe result without the script, the tasks and the date is not a result."""
    if not backbone.mrl_probe:
        return
    for token in ("probe_mrl.py", "Nano", "2026"):
        assert token in backbone.mrl_probe, (
            f"{backbone.key}'s mrl_probe does not record {token!r}"
        )


def test_probe_does_not_silently_become_the_documented_claim():
    """Measuring a model does not license describing it as MRL-trained.

    `mrl_valid()` gates which truncation dimensions the grid may report as a real
    MRL reduction, and it keys off `mrl_dims` -- the documented set -- alone. Six
    NanoBEIR queries' worth of evidence is not a training-recipe citation, and the
    write-up has to keep the two apart.
    """
    for key in ("lfm25", "harrier-270m", "harrier-06b"):
        b = backbones.get(key)
        assert b.mrl_probe, f"{key} should have been probed"
        assert b.mrl_dims is None
        assert not b.mrl_valid(128)
    assert backbones.shared_mrl_dims() == []


def test_the_two_known_mrl_models_are_the_controls():
    """The probe is only interpretable if the documented models went through it.

    Whatever the statistic reads on mGTE and mDenseOn *is* what MRL-trained looks
    like under this measurement. Without them a number for harrier-0.6b would need
    a threshold picked by hand -- which, on the first attempt, was wrong.
    """
    for key in ("mgte", "mdenseon"):
        b = backbones.get(key)
        assert b.mrl_dims, f"{key} is supposed to be a documented-MRL control"
        assert b.mrl_probe, f"control {key} was never run through the probe"
