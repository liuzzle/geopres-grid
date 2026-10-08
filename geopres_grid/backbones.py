"""The backbone registry.

One place for every fact about the five backbones that the rest of the pipeline
branches on. Everything here was read off the models' own HuggingFace configs on
2026-09-01; `scripts/smoke_backbones.py` re-checks it against the loaded model, so
a silent upstream change shows up as a failed assertion rather than as a bad number
three work packages later.

Two fields exist because of findings in the smoke test:

  `normalizes` -- four of the five stacks end in a `Normalize` module and
  mDenseOn's does not. Appending a projection after the stack therefore means
  "project unit vectors, return unnormalised" for some models and "project raw
  pooled vectors" for others. WP-D removes the ambiguity by caching
  pre-`Normalize` vectors and applying `normalize -> reduce -> normalize`
  explicitly; this flag is what lets the cache layer know what it is looking at.

  `prompts` -- four of the five use asymmetric query/document prompts. MTEB's
  own `CachedEmbeddingWrapper` keys its cache on `sha256(text)` alone, so for
  these models a query and a document with identical text collide. WP-C keys on
  the *prompted* text and scopes by `prompt_type`; this field is the list of
  models for which that matters.

The set changed on 07.10.2026 (Andrianos, answering the 06.10 notes):
`LiquidAI/LFM2.5-Embedding-350M` was dropped -- its 512-token ceiling kept MLDR
out and capped every other model's context -- and `google/embeddinggemma-2` was
added. LFM2.5's probe results and prompt findings stay in `logging.md`.
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

    None means the model declares none, and sentence-transformers falls back to the
    tokenizer's `model_max_length`, which is not a deliberate choice by the model
    author. Both harrier checkpoints ship no `sentence_bert_config.json`;
    EmbeddingGemma-2 ships one without the key and its tokenizer says 1e30.
    """

    context_window: int
    """Documented context window in tokens, from the model card or paper -- the
    ceiling for `max_seq_length`, which `native_max_seq_length` is not where that
    is None. `PRIMARY_MAX_SEQ_LENGTH` must not exceed any of these."""

    normalizes: bool
    """Whether the ST module stack ends in a `Normalize` module."""

    trust_remote_code: bool
    """Whether loading executes code from the model repo."""

    transformers_majors: tuple[int, ...] = (4, 5)
    """Major `transformers` versions this model actually runs on.

    Not a guess from metadata -- every entry here was established by loading the
    model and encoding with it. Three of the five are single-major: mGTE only works
    on 4.x (on 5.x it loads and then dies inside `forward`, see its notes),
    mDenseOn and EmbeddingGemma-2 only on 5.x. There is therefore no single
    environment that runs the whole model set; see `ENVIRONMENTS` below.
    """

    sentence_transformers_majors: tuple[int, ...] = (5,)
    """Major `sentence-transformers` versions this model actually runs on.

    5.x is the project pin and the supervisor's ceiling. EmbeddingGemma-2 is the
    one exception: its `modules.json` names `sentence_transformers.base.modules.
    normalize`, which 5.7 does not have, so it gets its own 6.x environment rather
    than lifting the ceiling for every model. No other model has been tested on
    6.x, which is why the default is (5,) and not (5, 6).
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

    task_prompt_names: dict[str, str] = field(default_factory=dict)
    """MTEB task type -> key into `prompts`, for the tasks MTEB encodes without a
    `prompt_type` (STS, Classification, Clustering), where every text goes through
    one side. A type missing here falls back to the query prompt.

    The rule (Andrianos, 06.10.2026, stated for STS): a model with a dedicated
    prompt for the task type uses it; every other model uses its query prompt --
    read as "what follows is a short text" -- even where that costs it, because
    the rule is applied consistently and never tuned per model. Extended per type
    on 07.10.2026 when classification and clustering joined the task set: one
    prompt for all symmetric tasks would have given harrier's "Retrieve
    semantically similar text" to a sentiment classifier. Cite the dedicated
    prompts in `prompt_source`. Mirrored into `EncodeConfig.task_prompt_names`.
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

    def prompt_name_for(self, side: str, task_type: str | None = None) -> str | None:
        """Prompt name to pass to `encode` for `side` in ("query", "document",
        "symmetric"). "symmetric" is a task without a `prompt_type`; its prompt
        depends on the MTEB `task_type` (`task_prompt_names`)."""
        if side == "query":
            return self.query_prompt_name
        if side == "document":
            return self.document_prompt_name
        if side == "symmetric":
            return self.task_prompt_names.get(task_type or "") or self.query_prompt_name
        raise ValueError(f"side must be 'query', 'document' or 'symmetric', got {side!r}")

    def prompt_for(self, side: str, task_type: str | None = None) -> str:
        """The literal prefix prepended to text on `side`. Empty string when bare.

        This is the string the cache key has to be built over: WP-C hashes the
        *prompted* text, so `prompt_for(side) + text` is the thing that gets
        hashed, not `text`.
        """
        name = self.prompt_name_for(side, task_type)
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
            context_window=8192,
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
            context_window=8192,
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
            key="harrier-270m",
            model_id="microsoft/harrier-oss-v1-270m",
            revision="31de22b673913c7d658c0f03f792d77c2dcf8ebd",
            native_dim=640,
            native_max_seq_length=None,
            # Model card, "Max Tokens" column of the harrier-oss-v1 table.
            context_window=32768,
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
            task_prompt_names={"STS": "sts_query"},
            prompt_source=(
                "Model card usage example, which is the only documentation of the "
                "split: `query_embeddings = model.encode(queries, "
                "prompt_name=\"web_search_query\")` followed by "
                "`document_embeddings = model.encode(documents)` -- documents take no "
                "prompt at all. Confirmed by the maintainers: 'Yes, this is how the "
                "model is trained, otherwise you will see a performance degradation.' "
                "The three declared prompts are all *query*-side and task-specific; "
                "`web_search_query` is the retrieval one. Checked 17.09.2026. "
                "Symmetric tasks (STS) take `sts_query` on both texts: it is the "
                "instruction mteb 2.15.1's own harrier implementation "
                "(`harrier_models.py`, `harrier_task_prompts`) gives STSBenchmark, "
                "STS12-17 and SICK-R alike; the retrieval prompt was being applied "
                "there before 06.10.2026. Classification and clustering have no "
                "declared prompt, so they take the query prompt by the same rule; "
                "mteb's own per-task instructions for them are not in the checkpoint."
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
            # Model card, "Max Tokens" column of the harrier-oss-v1 table.
            context_window=32768,
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
            task_prompt_names={"STS": "sts_query"},
            prompt_source=(
                "Model card usage example, which is the only documentation of the "
                "split: `query_embeddings = model.encode(queries, "
                "prompt_name=\"web_search_query\")` followed by "
                "`document_embeddings = model.encode(documents)` -- documents take no "
                "prompt at all. Confirmed by the maintainers: 'Yes, this is how the "
                "model is trained, otherwise you will see a performance degradation.' "
                "The three declared prompts are all *query*-side and task-specific; "
                "`web_search_query` is the retrieval one. Checked 17.09.2026. "
                "Symmetric tasks (STS) take `sts_query` on both texts: it is the "
                "instruction mteb 2.15.1's own harrier implementation "
                "(`harrier_models.py`, `harrier_task_prompts`) gives STSBenchmark, "
                "STS12-17 and SICK-R alike; the retrieval prompt was being applied "
                "there before 06.10.2026. Classification and clustering have no "
                "declared prompt, so they take the query prompt by the same rule; "
                "mteb's own per-task instructions for them are not in the checkpoint."
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
        Backbone(
            key="embeddinggemma-2",
            model_id="google/embeddinggemma-2",
            revision="914f7f89142e33e77833254d9c9b90c3cef7303b",
            native_dim=768,
            native_max_seq_length=None,
            # Model card, "Context Window: 8,192 tokens", shared by all modalities.
            context_window=8192,
            normalizes=True,
            trust_remote_code=False,
            transformers_majors=(5,),
            sentence_transformers_majors=(6,),
            prompts={
                "BitextMining": "task: search result | query: ",
                "Classification": "task: classification | query: ",
                "Clustering": "task: clustering | query: ",
                "CodeRetrieval": "task: code retrieval | query: ",
                "Document": "title: none | text: ",
                "FactChecking": "task: fact checking | query: ",
                "InstructionRetrieval": "task: code retrieval | query: ",
                "MultilabelClassification": "task: classification | query: ",
                "PairClassification": "task: sentence similarity | query: ",
                "QuestionAnswering": "task: question answering | query: ",
                "Reranking": "task: search result | query: ",
                "Retrieval": "task: search result | query: ",
                "Retrieval-document": "title: none | text: ",
                "Retrieval-query": "task: search result | query: ",
                "STS": "task: sentence similarity | query: ",
                "SearchQuery": "task: search result | query: ",
                "SentenceSimilarity": "task: sentence similarity | query: ",
                "Summarization": "task: sentence similarity | query: ",
                "document": "title: none | text: ",
                "query": "task: search result | query: ",
            },
            query_prompt_name="query",
            document_prompt_name="document",
            task_prompt_names={
                "STS": "STS",
                "Classification": "Classification",
                "Clustering": "Clustering",
            },
            prompt_source=(
                "Model card usage example: `model.encode(query, "
                "prompt_name=\"SearchQuery\")` / `model.encode(document, "
                "prompt_name=\"Document\")`; `query` and `document` resolve to the "
                "same two prefixes and are what the other models use. The card's "
                "prompt table gives the symmetric tasks their own prompts -- "
                "Classification, Clustering, and SentenceSimilarity (declared as `STS` "
                "too) -- so all three are dedicated under the task-prompt rule. "
                "Prefixes read from config_sentence_transformers.json at the pinned "
                "sha. Checked 07.10.2026."
            ),
            mrl_dims=(128, 256, 512, 768),
            mrl_source=(
                "Model card, 'Matryoshka Representation Learning (MRL): Native support "
                "for truncated embeddings across 128d, 256d, 512d, and 768d', with a "
                "per-dimension MTEB table and the instruction to re-normalise after "
                "truncating. No paper yet (checked 07.10.2026), so unlike mGTE and "
                "mDenseOn this is the vendor's card, not a training-recipe citation."
            ),
            mrl_checked=True,
            notes=(
                "Released 06.10.2026. Multimodal (text, image, video, audio into one "
                "768-d space); only the text path is used here, but the checkpoint "
                "carries the vision and audio towers (740M parameters in total). "
                "Needs transformers >= 5.19 (first release with `embedding_gemma2`) "
                "and sentence-transformers 6.x: on 5.7 loading fails with "
                "`No module named sentence_transformers.base.modules.normalize`. It "
                "runs in `envs/sentence_transformers6`; verified there on 07.10.2026 "
                "with transformers 5.19.0 and sentence-transformers 6.1.0 (loads, "
                "768-d, unit norm, cosine). The card forbids float16 -- 'the model "
                "returns NaN or silently degraded embeddings' -- so precompute in "
                "float32 or bfloat16 only. Its sentence_bert_config.json declares no "
                "max_seq_length and the tokenizer reports 1e30, so leaving it unset "
                "is not an option. Caveat for titled corpora: the card asks for "
                "`title: {title} | text: {body}`, while MTEB prepares "
                "`title + ' ' + body` and the prefix says `title: none` -- the same "
                "prepared text every backbone gets, slightly off this model's format."
            ),
        ),
    ]
}


PRIMARY_MAX_SEQ_LENGTH = 8192
"""One `max_seq_length` for every backbone, in every run.

8192 is the smallest documented `context_window` of the five (mGTE, mDenseOn and
EmbeddingGemma-2; the harriers reach 32768), so every model can actually reach it.
Fixing one value means every model sees the same effective corpus, so a
cross-model difference is a property of the model rather than of how much text it
was allowed to read -- meeting 20.08 §6, "cross-model comparison needs a fixed
max_seq_length or an explicit caveat".

It has to be set, not left to the model: harrier and EmbeddingGemma-2 declare
none, and sentence-transformers then falls back to the tokenizer's
`model_max_length` -- 32768 and 1e30, past what EmbeddingGemma-2 was trained on.

Until 07.10.2026 this was 512, LFM2.5's ceiling. Dropping LFM2.5 lifted it, and
with it the separate native-length run: MLDR, a long-document benchmark that only
made sense there, now runs at the same length as everything else.
"""


ENVIRONMENTS: dict[tuple[int, int], str] = {
    (4, 5): "the project root environment (`uv sync`)",
    (5, 5): "`envs/transformers5` (`uv sync --project envs/transformers5`)",
    (5, 6): "`envs/sentence_transformers6` (`uv sync --project envs/sentence_transformers6`)",
}
"""Which environment provides which (`transformers`, `sentence-transformers`) major.

The split is not a preference. mGTE cannot run on transformers 5.x, mDenseOn
cannot run on 4.x, and EmbeddingGemma-2 cannot run on sentence-transformers 5.x,
all established by loading them rather than by reading metadata. The two harriers
run in either of the first two, so a run covering the whole model set is three
precompute commands.

Everything downstream of the cache is unaffected: the cache stores plain fp32
arrays, so DR, quantization and scoring all happen in the root environment
regardless of which one produced the embeddings.
"""


def installed_transformers_major() -> int:
    """Major version of the `transformers` actually installed in this interpreter."""
    import transformers

    return int(transformers.__version__.split(".")[0])


def installed_sentence_transformers_major() -> int:
    """Major version of the `sentence-transformers` installed in this interpreter."""
    import sentence_transformers

    return int(sentence_transformers.__version__.split(".")[0])


def environments_for(backbone: Backbone) -> list[tuple[int, int]]:
    """The registered environments `backbone` runs in."""
    return [
        key
        for key in ENVIRONMENTS
        if key[0] in backbone.transformers_majors
        and key[1] in backbone.sentence_transformers_majors
    ]


def _installed(major: int | None, st_major: int | None) -> tuple[int, int]:
    return (
        installed_transformers_major() if major is None else major,
        installed_sentence_transformers_major() if st_major is None else st_major,
    )


def runnable_backbones(major: int | None = None, st_major: int | None = None) -> list[Backbone]:
    """Backbones that can load under these `transformers` / `sentence-transformers`
    majors, by default the installed ones."""
    key = _installed(major, st_major)
    return [b for b in all_backbones() if key in environments_for(b)]


def require_runnable(
    backbone: Backbone, major: int | None = None, st_major: int | None = None
) -> None:
    """Fail early, and with the fix, when a backbone is in the wrong environment.

    Without this the failure surfaces as `Unrecognized processing class` (mDenseOn
    on 4.x), as an IndexError inside `forward` with a nonsense index (mGTE on
    5.x), or as a missing `sentence_transformers.base.modules.normalize`
    (EmbeddingGemma-2 on sentence-transformers 5.x), none of which names the
    actual problem.
    """
    key = _installed(major, st_major)
    if key in environments_for(backbone):
        return
    supported = ", ".join(
        f"transformers {t}.x + sentence-transformers {st}.x -> {ENVIRONMENTS[(t, st)]}"
        for t, st in environments_for(backbone)
    )
    raise RuntimeError(
        f"{backbone.key} ({backbone.model_id}) does not run on transformers {key[0]}.x "
        f"with sentence-transformers {key[1]}.x. It needs: {supported or 'no registered environment'}"
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
