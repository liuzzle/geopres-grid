"""The backbone registry.

One place for every fact about the five backbones that the rest of the pipeline
branches on. Everything here was read off the models' own HuggingFace configs on
2026-09-01; `scripts/smoke_backbones.py` re-checks it against the loaded model, so
a silent upstream change shows up as a failed assertion rather than as a bad number
three work packages later.

Two fields exist because of findings in the smoke test:

  `normalizes` -- three of the five stacks end in a `Normalize` module and two do
  not. Appending a projection after the stack therefore means "project unit
  vectors, return unnormalised" for some models and "project raw pooled vectors"
  for others. WP-D removes the ambiguity by caching pre-`Normalize` vectors and
  applying `normalize -> reduce -> normalize` explicitly; this flag is what lets
  the cache layer know what it is looking at.

  `prompts` -- three of the five use asymmetric query/document prompts. MTEB's
  own `CachedEmbeddingWrapper` keys its cache on `sha256(text)` alone, so for
  these models a query and a document with identical text collide. WP-C keys on
  the *prompted* text and scopes by `prompt_type`; this field is the list of
  models for which that matters.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Backbone:
    """Everything the pipeline needs to know about one backbone."""

    key: str
    """Short slug. Used in paths and CLI arguments."""

    model_id: str
    """HuggingFace model id."""

    revision: str
    """Pinned commit sha. Never track `main` -- WP-B hashes this into the run id."""

    native_dim: int
    """Output dimension of the backbone, before any reduction."""

    native_max_seq_length: int | None
    """`max_seq_length` the model ships with, or None when it declares none.

    None means the model has no `sentence_bert_config.json` and sentence-transformers
    falls back to the tokenizer's `model_max_length`, which is not a deliberate
    choice by the model author. Both harrier checkpoints are in this position.
    """

    normalizes: bool
    """Whether the ST module stack ends in a `Normalize` module."""

    trust_remote_code: bool
    """Whether loading executes code from the model repo."""

    transformers_majors: tuple[int, ...] = (4, 5)
    """Major `transformers` versions this model actually runs on.

    Not a guess from metadata -- every entry here was established by loading the
    model and encoding with it. Two of the five are single-major: mGTE only works
    on 4.x (on 5.x it loads and then dies inside `forward`, see its notes) and
    mDenseOn only on 5.x (its tokenizer class does not exist before 5.0). There is
    therefore no single environment that runs the whole model set; see
    `ENVIRONMENTS` below.
    """

    prompts: dict[str, str] | None = None
    """Prompt prefixes from `config_sentence_transformers.json`, if any."""

    query_prompt_name: str | None = None
    """Key into `prompts` to use when encoding a **query**, or None for bare text.

    A name rather than the prefix itself, because that is what
    `SentenceTransformer.encode(prompt_name=...)` takes and what the model cards
    document. Note that it is not always `"query"`: both harriers call their
    retrieval prompt `web_search_query`, so code that hard-codes `"query"` gets
    silently no prefix on those models. See `prompt_source`.
    """

    document_prompt_name: str | None = None
    """Key into `prompts` to use when encoding a **document**, or None for bare text.

    None alongside a non-None `query_prompt_name` is a deliberate asymmetry, not a
    gap in the registry: the harrier cards encode documents with no prompt at all.
    Adding one would be off-distribution for the model.
    """

    prompt_source: str = ""
    """Citation for the two prompt-name fields -- where the query/document split is
    documented, in the model's own words."""

    mrl_dims: tuple[int, ...] | None = None
    """Dimensions the model was actually Matryoshka-trained at, per its paper.

    None means MRL is not documented for this model -- which is not the same as
    "not MRL-trained", only that we have no source. Truncating an MRL-trained model
    at a dimension outside this set is not a supported use of MRL; truncating a
    non-MRL model at any dimension is a naive baseline. The distinction is the whole
    reason truncation can be reported as a near-zero-overhead reduction method for
    some models and only as a control for others.
    """

    mrl_source: str = ""
    """Citation for `mrl_dims` -- or, when `mrl_dims` is None and `mrl_checked` is
    True, the record of where we looked and found nothing."""

    mrl_checked: bool = False
    """Whether the MRL question has actually been investigated for this model.

    `mrl_dims=None, mrl_checked=False` means unknown. `mrl_dims=None,
    mrl_checked=True` means we looked and found no documented MRL training. The
    distinction exists because reading silence as absence is exactly the mistake
    made on 02.09: mGTE's and mDenseOn's model cards say nothing about Matryoshka
    and both turned out to be MRL-trained, per their papers.
    """

    mrl_probe: str = ""
    """What the *empirical* prefix probe found, or "" if it has never been run.

    `mrl_dims` and `mrl_source` record what a vendor documents. This records what
    `scripts/probe_mrl.py` measured: at a fixed budget k, whether the first k
    coordinates outperform an arbitrary k columns (the null) and how far they close
    the gap to PCA (the reference). The two fields answer different questions and
    can legitimately disagree -- an undocumented model can still behave like an MRL
    model, which is exactly the possibility this field exists to settle.

    `mrl_valid()` deliberately still keys off `mrl_dims` alone. A measurement on six
    NanoBEIR tasks is evidence about the grid, not a licence to describe a model as
    MRL-trained in the write-up.
    """

    notes: str = ""

    @property
    def slug(self) -> str:
        """Filesystem-safe identifier, matching the old repo's convention."""
        return self.model_id.replace("/", "__")

    def mrl_valid(self, dim: int) -> bool:
        """Whether truncating to `dim` is a documented use of MRL for this model."""
        return bool(self.mrl_dims) and dim in self.mrl_dims

    def prompt_name_for(self, side: str) -> str | None:
        """Prompt name to pass to `encode` for `side` in ("query", "document")."""
        if side == "query":
            return self.query_prompt_name
        if side == "document":
            return self.document_prompt_name
        raise ValueError(f"side must be 'query' or 'document', got {side!r}")

    def prompt_for(self, side: str) -> str:
        """The literal prefix prepended to text on `side`. Empty string when bare.

        This is the string the cache key has to be built over: WP-C hashes the
        *prompted* text, so `prompt_for(side) + text` is the thing that gets
        hashed, not `text`.
        """
        name = self.prompt_name_for(side)
        if name is None:
            return ""
        if not self.prompts or name not in self.prompts:
            raise KeyError(
                f"{self.key} names prompt {name!r} for the {side} side, but its "
                f"prompts dict has {sorted(self.prompts or {})}"
            )
        return self.prompts[name]

    @property
    def has_asymmetric_prompts(self) -> bool:
        """True when the prefix applied depends on which side is being encoded.

        A model that names a prompt for only one side counts as asymmetric: the
        query gets a prefix and the document goes bare, which is exactly the case
        a shared-text cache key gets wrong.

        Resolved from `query_prompt_name` / `document_prompt_name` rather than by
        guessing key names out of `prompts`. The earlier version looked up
        `prompts["query"]` and `prompts["document"]`, which happened to return the
        right answer for the harriers only because *neither* key exists there and
        it fell through to True -- the right answer for the wrong reason, and it
        would have been the wrong answer for any model naming only a document
        prompt.
        """
        return self.prompt_for("query") != self.prompt_for("document")


PROBE = (
    "Probed 18.09.2026, scripts/probe_mrl.py, NanoSciFact + NanoNFCorpus "
    "(100 queries), max_seq_length=512, fp32. "
)
"""Shared provenance prefix for the `mrl_probe` strings below."""


BACKBONES: dict[str, Backbone] = {
    b.key: b
    for b in [
        Backbone(
            key="mgte",
            model_id="Alibaba-NLP/gte-multilingual-base",
            revision="9bbca17d9273fd0d03d5725c7a4b0f6b45142062",
            native_dim=768,
            native_max_seq_length=8192,
            normalizes=True,
            trust_remote_code=True,
            prompts=None,
            query_prompt_name=None,
            document_prompt_name=None,
            prompt_source=(
                "No prompts. config_sentence_transformers.json declares none and the "
                "model card's usage example calls encode() bare on both sides. The "
                "only symmetric backbone of the five, so it is the one model whose "
                "cache key is safe on raw text."
            ),
            transformers_majors=(4,),
            # Paper eq. 3: D = {32k | k in N, k >= 1, 32k <= H}, H = 768.
            mrl_dims=tuple(range(32, 768 + 1, 32)),
            mrl_source=(
                "mGTE, arXiv:2407.19669, §2.2 'Matryoshka Embedding' and eq. 3; "
                "elastic-embedding results in §3.2"
            ),
            mrl_checked=True,
            mrl_probe=(
                PROBE + "POSITIVE CONTROL, and it behaves like one: truncation beats "
                "random columns at k=32/64/128 with the 95% paired bootstrap on "
                "trunc-rand excluding zero (+0.072 [+0.029, +0.117] at k=32), "
                "mrl_gain +0.46/+0.51/+1.12. Variance in the first 32 coordinates is "
                "0.071 against 0.042 for uniform (ratio 1.63), decaying toward 1.0 as "
                "k grows -- front-loading, as the paper's eq. 3 implies. Loses "
                "resolution at k>=256 where PCA stops beating random columns. This is "
                "the band an undocumented model has to be read against."
            ),
            notes=(
                "Custom `NewModel` architecture loaded via auto_map. The reason the "
                "repo stays on the transformers 4.x line. Retested on 5.16.1 "
                "(02.09): it loads cleanly and then raises inside forward at "
                "modeling.py:392, `rope_cos[position_ids]` with a garbage index, "
                "because transformers 5 changed how position_ids are defaulted and "
                "the vendored code reads them raw. No upstream fix is coming -- "
                "Alibaba-NLP/new-impl's last commit predates the 5.x line. "
                "Caveat for WP-B: its remote code is fetched from the separate "
                "`Alibaba-NLP/new-impl` repo, which the `revision` above does not "
                "pin -- so a pinned sha here does not fully pin the encode path."
            ),
        ),
        Backbone(
            key="mdenseon",
            model_id="lightonai/mDenseOn",
            revision="a5fdb000f7a21da96c3bddde3a782ef777316df3",
            native_dim=768,
            native_max_seq_length=8192,
            normalizes=False,
            trust_remote_code=False,
            prompts={"query": "query: ", "document": "document: "},
            query_prompt_name="query",
            document_prompt_name="document",
            prompt_source=(
                "Model card usage example: "
                "`model.encode(queries, prompt_name=\"query\")` / "
                "`model.encode(documents, prompt_name=\"document\")`; "
                "prefixes read from config_sentence_transformers.json."
            ),
            transformers_majors=(5,),
            mrl_dims=(128, 256, 512, 768),
            mrl_source=(
                "DenseOn/LateOn, arXiv:2607.27178, §2 and appendix C.3: "
                "'we apply Matryoshka Representation Learning (MRL) on the InfoNCE "
                "loss with truncation dimensions {128, 256, 512, 768}'"
            ),
            mrl_checked=True,
            mrl_probe=(
                PROBE + "POSITIVE CONTROL, and the more informative of the two, "
                "because it has a documented boundary the probe can be checked "
                "against: its trained set is {128, 256, 512, 768}, so k=32 and k=64 "
                "are *outside* it. That is roughly what comes out. trunc-rand is "
                "+0.045 [+0.006, +0.086] at k=128 -- the bottom of the documented set, "
                "and the only k where the interval excludes zero -- against +0.034 "
                "[-0.020, +0.088] at k=32 and +0.037 [-0.006, +0.081] at k=64. "
                "mrl_gain +0.39/+0.50/+0.93. Read carefully: the point estimates below "
                "128 are positive and the intervals only just include zero, so this is "
                "'not detectable at 100 queries', not 'absent'. The honest reading is "
                "that the probe recovers the documented boundary without being told "
                "where it is, which is the closest thing to a calibration this design "
                "can offer."
            ),
            notes=(
                "ModernBERT. Its modules.json uses the refactored "
                "`sentence_transformers.base.modules.*` paths, which do not exist "
                "before sentence-transformers 5.4. "
                "Its tokenizer_config.json declares `tokenizer_class: "
                "TokenizersBackend`, introduced in transformers 5.0.0, so it cannot "
                "load on the 4.x line at all -- AutoTokenizer raises 'Unrecognized "
                "processing class'. mGTE cannot load on 5.x. The two are therefore "
                "permanently split across environments; this one belongs to "
                "`envs/transformers5`. Verified there on 02.09: loads, encodes, and "
                "every field above matches the live model (768-d, query:/document: "
                "prompts, Transformer -> Pooling with no Normalize, norms ~43)."
            ),
        ),
        Backbone(
            key="lfm25",
            model_id="LiquidAI/LFM2.5-Embedding-350M",
            revision="f35ae2c91d687658dbf1f2b449382f0b019b9808",
            native_dim=1024,
            native_max_seq_length=512,
            normalizes=False,
            trust_remote_code=True,
            prompts={
                "query": "query: ",
                "document": "document: ",
                # Training-time aliases the checkpoint ships; all map to the
                # document prefix and are recorded so the registry matches the
                # loaded model exactly.
                "positive": "document: ",
                **{f"negative_{i}": "document: " for i in range(7)},
            },
            query_prompt_name="query",
            document_prompt_name="document",
            prompt_source=(
                "Model card, 'Encoding queries and documents': 'Always pass "
                "prompt_name=\"query\" for queries and prompt_name=\"document\" for "
                "passages -- the model was trained with these prefixes, and omitting "
                "them silently degrades retrieval quality.' The card's own training "
                "snippet passes prompts={'query': 'query: ', 'positive': 'document: '}, "
                "which is where the `positive`/`negative_i` aliases in `prompts` come "
                "from: they are training-time role names, all resolving to the document "
                "prefix. Checked 17.09.2026."
            ),
            mrl_checked=True,
            mrl_source=(
                "No MRL documented. Checked 15.09.2026: HF model card (LiquidAI/LFM2.5-Embedding-350M), the GGUF card, the release blog 'LFM2.5 Retrievers: Bi-directional LFMs for Fast Multilingual Search', and the Liquid docs page. The blog enumerates the full training recipe -- (1) English contrastive pretraining, (2) multilingual/cross-lingual distillation, (3) fine-tuning on hard-mined negatives -- with no nested or Matryoshka objective. No technical report exists. NOTE: web search attributes Matryoshka dims {2048, 1024, 512, 256} to *LFM2.5-230M*, a different (generative) model; it does not transfer to this checkpoint, and 2048 is not even reachable from this model's 1024-d output."
            ),
            mrl_probe=(
                PROBE + "NO PREFIX PRIVILEGE. Truncation is statistically "
                "indistinguishable from keeping k random columns at every k probed: "
                "trunc-rand is +0.023 [-0.025, +0.071] at k=32 and within noise of "
                "zero at 64/128/256, mrl_gain +0.15/-0.04/-0.06/+0.05. Variance in "
                "the first 32 coordinates is 0.032 against 0.031 for uniform -- flat "
                "to three decimals, no front-loading at all. So the documented "
                "negative (mrl_dims=None) is confirmed by measurement rather than "
                "inferred from silence. Truncating LFM2.5 is a naive baseline; it is "
                "not, however, actively harmful the way it is on the harriers."
            ),
            notes=(
                "Shortest context of the five at 512 tokens, which is what sets "
                "PRIMARY_MAX_SEQ_LENGTH below."
            ),
        ),
        Backbone(
            key="harrier-270m",
            model_id="microsoft/harrier-oss-v1-270m",
            revision="31de22b673913c7d658c0f03f792d77c2dcf8ebd",
            native_dim=640,
            native_max_seq_length=None,
            normalizes=True,
            trust_remote_code=False,
            prompts={
                "web_search_query": (
                    "Instruct: Given a web search query, retrieve relevant passages "
                    "that answer the query\nQuery: "
                ),
                "sts_query": "Instruct: Retrieve semantically similar text\nQuery: ",
                "bitext_query": "Instruct: Retrieve parallel sentences\nQuery: ",
            },
            query_prompt_name="web_search_query",
            document_prompt_name=None,
            prompt_source=(
                "Model card usage example, which is the only documentation of the "
                "split: `query_embeddings = model.encode(queries, "
                "prompt_name=\"web_search_query\")` followed by "
                "`document_embeddings = model.encode(documents)` -- documents take no "
                "prompt at all. Confirmed by the maintainers: 'Yes, this is how the "
                "model is trained, otherwise you will see a performance degradation.' "
                "The three declared prompts are all *query*-side and task-specific; "
                "`web_search_query` is the retrieval one. Checked 17.09.2026."
            ),
            mrl_checked=True,
            mrl_source=(
                "No MRL documented. Checked 15.09.2026: HF model cards for both harrier checkpoints, the Microsoft Foundry Labs page, and the Bing blog release post. Training is described as contrastive learning plus knowledge distillation from a larger teacher, with no nested objective. No arXiv technical report exists for harrier-oss-v1, so unlike mGTE and mDenseOn there is no paper that could contradict the cards -- which is why this is recorded as 'no evidence found' rather than as a settled negative."
            ),
            mrl_probe=(
                PROBE + "WORSE THAN RANDOM. Truncation loses to keeping k random "
                "columns: trunc-rand -0.087 [-0.130, -0.043] at k=32 and -0.046 "
                "[-0.088, -0.005] at k=128, both intervals excluding zero; k=64 is "
                "not distinguishable. mrl_gain -0.52/-0.23/-0.75. Variance in the "
                "first 32 coordinates 0.055 against 0.050 uniform -- essentially "
                "flat. Truncation is therefore not merely a naive baseline here, it "
                "is worse than the naive baseline, and the grid should say so."
            ),
            notes=(
                "Gemma-3 based. Carries the same instruct-style query prompts as the "
                "0.6b checkpoint -- the registry originally recorded none, which the "
                "smoke test caught. "
                "The Gemma family has a history of fp16 overflow and a "
                "T4 cannot do bf16, so test this one in fp16 on the target GPU early "
                "(plan risk table). Ships no sentence_bert_config.json."
            ),
        ),
        Backbone(
            key="harrier-06b",
            model_id="microsoft/harrier-oss-v1-0.6b",
            revision="f9b9dc8d367d443f2479d27aa5d8d2850c0774ee",
            native_dim=1024,
            native_max_seq_length=None,
            normalizes=True,
            trust_remote_code=False,
            prompts={
                "web_search_query": (
                    "Instruct: Given a web search query, retrieve relevant passages "
                    "that answer the query\nQuery: "
                ),
                "sts_query": "Instruct: Retrieve semantically similar text\nQuery: ",
                "bitext_query": "Instruct: Retrieve parallel sentences\nQuery: ",
            },
            query_prompt_name="web_search_query",
            document_prompt_name=None,
            prompt_source=(
                "Model card usage example, which is the only documentation of the "
                "split: `query_embeddings = model.encode(queries, "
                "prompt_name=\"web_search_query\")` followed by "
                "`document_embeddings = model.encode(documents)` -- documents take no "
                "prompt at all. Confirmed by the maintainers: 'Yes, this is how the "
                "model is trained, otherwise you will see a performance degradation.' "
                "The three declared prompts are all *query*-side and task-specific; "
                "`web_search_query` is the retrieval one. Checked 17.09.2026."
            ),
            mrl_checked=True,
            mrl_source=(
                "No MRL documented. Checked 15.09.2026: HF model cards for both harrier checkpoints, the Microsoft Foundry Labs page, and the Bing blog release post. Training is described as contrastive learning plus knowledge distillation from a larger teacher, with no nested objective. No arXiv technical report exists for harrier-oss-v1, so unlike mGTE and mDenseOn there is no paper that could contradict the cards -- which is why this is recorded as 'no evidence found' rather than as a settled negative."
            ),
            mrl_probe=(
                PROBE + "WORSE THAN RANDOM, by the largest margin of the five, which "
                "settles the Qwen3 lineage question in `notes` against inheritance. "
                "trunc-rand is -0.075 [-0.120, -0.029] at k=32, -0.169 [-0.225, "
                "-0.120] at k=64 and -0.137 [-0.191, -0.087] at k=128, every interval "
                "excluding zero; mrl_gain -0.29/-1.34/-3.76. In absolute terms k=32 "
                "keeps 0.087 of a full-width 0.593. The mechanism is unexciting and "
                "visible in the cached vectors: variance in the first 32 coordinates "
                "is 0.025 against 0.031 for uniform (ratio 0.81), so the prefix is a "
                "slightly *below*-average slice and random columns sample the whole "
                "spectrum instead. No massive-activation story required."
            ),
            notes=(
                "Qwen3 based. Instruct-style query prompts with no matching document "
                "prompt, so the query side is prefixed and the document side is not. "
                "Ships no sentence_bert_config.json. "
                "LINEAGE, because it is the one place an undocumented MRL property "
                "could have arrived from: Qwen3-Embedding-0.6B is MRL-trained (its "
                "card's model table, 'MRL Support: Yes', defined there as support for "
                "custom output dimensions), it is also 1024-d and Qwen3-based, and "
                "harrier's `web_search_query` prefix is the Qwen3-Embedding query "
                "prompt copied verbatim -- 'Instruct: Given a web search query, "
                "retrieve relevant passages that answer the query\nQuery:' -- where "
                "Qwen3-Embedding pairs it with an explicit empty document prompt, "
                "which is harrier's bare document side written down. But the config "
                "matches the *base* LLM, not the embedding model: vocab_size 151936 "
                "and eos_token_id 151645 are Qwen/Qwen3-0.6B's, against 151669 and "
                "151643 for Qwen/Qwen3-Embedding-0.6B (everything else -- 28 layers, "
                "1024 hidden, 3072 intermediate, 16/8 heads, head_dim 128, rope_theta "
                "1e6 -- is shared by all three and so distinguishes nothing). Read "
                "that way, harrier inherited the prompt convention and the backbone "
                "but not the contrastive stage that MRL lives in. Checked 17.09.2026 "
                "against the three HF configs. It is an inference from metadata, not "
                "a statement by Microsoft, which is why `mrl_probe` measures it "
                "instead of resting on it."
            ),
        ),
    ]
}


PRIMARY_MAX_SEQ_LENGTH = 512
"""One `max_seq_length` for every backbone in the primary results.

512 is LFM2.5's ceiling and therefore the only value all five can actually reach.
Fixing it means every model sees the same effective corpus, so a cross-model
difference is a property of the model rather than of how much text it was allowed
to read. Native lengths (8192 / 8192 / 512 / tokenizer / tokenizer) are a
supplementary run, not the headline -- meeting 20.08 §6, "cross-model comparison
needs a fixed max_seq_length or an explicit caveat".
"""


ENVIRONMENTS: dict[int, str] = {
    4: "the project root environment (`uv sync`)",
    5: "`envs/transformers5` (`uv sync --project envs/transformers5`)",
}
"""Which environment provides which `transformers` major.

The split is not a preference. mGTE cannot run on 5.x and mDenseOn cannot run on
4.x, both established by loading them rather than by reading metadata, and neither
has an upstream fix pending. Three of the five backbones run in either, so the only
cross-environment cost is that a run covering the whole model set is two commands.

Everything downstream of the cache is unaffected: the cache stores plain fp32
arrays, so DR, quantization and scoring all happen in one environment regardless of
which one produced the embeddings.
"""


def installed_transformers_major() -> int:
    """Major version of the `transformers` actually installed in this interpreter."""
    import transformers

    return int(transformers.__version__.split(".")[0])


def runnable_backbones(major: int | None = None) -> list[Backbone]:
    """Backbones that can load under this `transformers` major version."""
    major = installed_transformers_major() if major is None else major
    return [b for b in all_backbones() if major in b.transformers_majors]


def require_runnable(backbone: Backbone, major: int | None = None) -> None:
    """Fail early, and with the fix, when a backbone is in the wrong environment.

    Without this the failure surfaces as `Unrecognized processing class` (mDenseOn
    on 4.x) or as an IndexError inside `forward` with a nonsense index (mGTE on
    5.x), neither of which names the actual problem.
    """
    major = installed_transformers_major() if major is None else major
    if major in backbone.transformers_majors:
        return
    supported = ", ".join(
        f"{m}.x -> {ENVIRONMENTS.get(m, 'unregistered environment')}"
        for m in backbone.transformers_majors
    )
    raise RuntimeError(
        f"{backbone.key} ({backbone.model_id}) does not run on transformers "
        f"{major}.x. It needs: {supported}"
    )


def unchecked_mrl() -> list[Backbone]:
    """Backbones whose MRL support has never been investigated.

    Empty is the goal. A model in this list must not be described as "not
    MRL-trained" -- only as unknown.
    """
    return [b for b in all_backbones() if not b.mrl_checked and not b.mrl_dims]


def shared_mrl_dims(keys: list[str] | None = None) -> list[int]:
    """Dimensions at which every named backbone supports MRL truncation.

    Empty when any of them has no documented MRL support.
    """
    selected = [get(k) for k in keys] if keys else all_backbones()
    if any(not b.mrl_dims for b in selected):
        return []
    common = set(selected[0].mrl_dims)
    for b in selected[1:]:
        common &= set(b.mrl_dims)
    return sorted(common)


def get(key: str) -> Backbone:
    """Look a backbone up by short key or by full model id."""
    if key in BACKBONES:
        return BACKBONES[key]
    for backbone in BACKBONES.values():
        if backbone.model_id == key:
            return backbone
    raise KeyError(
        f"Unknown backbone {key!r}. Known: {', '.join(sorted(BACKBONES))}"
    )


def all_backbones() -> list[Backbone]:
    """Every registered backbone, in declaration order."""
    return list(BACKBONES.values())
