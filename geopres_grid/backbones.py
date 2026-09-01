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

    prompts: dict[str, str] | None = None
    """Prompt prefixes from `config_sentence_transformers.json`, if any."""

    notes: str = ""

    @property
    def slug(self) -> str:
        """Filesystem-safe identifier, matching the old repo's convention."""
        return self.model_id.replace("/", "__")

    @property
    def has_asymmetric_prompts(self) -> bool:
        """True when the prefix applied depends on which side is being encoded.

        A model that names a prompt for only one side counts as asymmetric: the
        query gets a prefix and the document goes bare, which is exactly the case
        a shared-text cache key gets wrong.
        """
        if not self.prompts:
            return False
        query_prompt = self.prompts.get("query")
        document_prompt = self.prompts.get("document")
        if query_prompt is not None and document_prompt is not None:
            return query_prompt != document_prompt
        return True


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
            notes=(
                "Custom `NewModel` architecture loaded via auto_map. The reason the "
                "repo pins transformers==4.57.6; newer versions break it at encode "
                "time. Caveat for WP-B: its remote code is fetched from the separate "
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
            notes=(
                "ModernBERT. Its modules.json uses the refactored "
                "`sentence_transformers.base.modules.*` paths, which do not exist "
                "before sentence-transformers 5.4. "
                "BLOCKED: its tokenizer_config.json declares "
                "`tokenizer_class: TokenizersBackend`, which was introduced in "
                "transformers 5.0.0 and does not exist in the pinned 4.57.6 -- "
                "AutoTokenizer raises 'Unrecognized processing class'. mGTE needs "
                "the 4.57.6 pin, so the two models cannot share an environment "
                "until that pin is retested against transformers 5.x. "
                "Prompts below are from config_sentence_transformers.json and are "
                "unverified against a load."
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
            notes=(
                "Qwen3 based. Instruct-style query prompts with no matching document "
                "prompt, so the query side is prefixed and the document side is not. "
                "Ships no sentence_bert_config.json."
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
