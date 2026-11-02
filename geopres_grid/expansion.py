"""Document-agnostic LLM query expansion for the Impresso tier (test-time scaling).

Andrianos' suggestion (logging.md §71): an LLM rewrites each query from the query
alone -- it never sees a document, so retrieval stays standard dense retrieval and
the corpus side is untouched. The expansions are produced once, offline, by
`scripts/expand_queries.py`, and stored per `ExpansionConfig` hash; the expanded
Impresso tasks (`impresso.expanded_task`) read them back and re-encode only their
queries, their document blocks being copies of the original tasks'.

Variants (`VARIANTS`):

* `pseudodoc` -- query2doc (Wang et al., arXiv:2303.07678) / HyDE (Gao et al.,
  arXiv:2212.10496): a short passage that would answer the query, in the query's
  language. The newspaper-archive framing is collection-level, as HyDE's per-task
  instructions are ("write a news passage"), not document-level.
* `pseudodoc_doclang` -- the same passage written in the collection's language. For
  the crosslingual sets this also translates, so a gain there is not expansion
  alone; for the multilingual sets it differs from `pseudodoc` only in the prompt.
* `keywords` -- query-term expansion: related terms, synonyms and spelling
  variants (period spellings, OCR-prone forms), comma-separated.

How an expansion meets the query (`COMBINE_MODES`) is the task's business, not the
LLM call's, so one expansion file serves both: `replace` encodes the expansion
alone, `concat` encodes `query + "\\n" + expansion` (query2doc's dense setting).
Either way the text is encoded on the *query* side with the backbone's query
prompt; HyDE proper would encode a pseudo-document with the document prompt.

This module imports neither `mteb` nor `openai`: the expansion script needs no
MTEB, and evaluation needs no OpenAI SDK (`openai` is the `expansion` extra).
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

from geopres_grid.identity import HASH_LENGTH, config_hash

# --- prompts ---------------------------------------------------------------------

COLLECTION_LANGUAGES = {"DE": "German", "FR": "French"}

_ARCHIVE = (
    "You help a search engine over an archive of historical Swiss newspapers "
    "(German- and French-language articles from the 19th and 20th centuries, digitised by OCR). "
)


@dataclass(frozen=True)
class PromptTemplate:
    """One versioned prompt. `version` is for people; `sha` is what identifies it.

    `input_template` may use `{query}` and `{collection_language}` (German/French,
    the language of the corpus the query is run against -- known per query set,
    not per document).
    """

    version: str
    instructions: str
    input_template: str

    @property
    def sha(self) -> str:
        text = f"{self.instructions}\x00{self.input_template}"
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:HASH_LENGTH]

    def render(self, query: str, collection_language: str) -> str:
        return self.input_template.format(query=query, collection_language=collection_language)


VARIANTS: dict[str, PromptTemplate] = {
    "pseudodoc": PromptTemplate(
        version="pseudodoc-v1",
        instructions=_ARCHIVE
        + "Given a search query, write a short newspaper passage (about 100 words) that would "
        "answer it, in the style of such an article. Write in the same language as the query. "
        "If you do not know the facts, write a plausible passage anyway. "
        "Return only the passage, with no title and no commentary.",
        input_template="Query: {query}",
    ),
    "pseudodoc_doclang": PromptTemplate(
        version="pseudodoc-doclang-v1",
        instructions=_ARCHIVE
        + "Given a search query, write a short newspaper passage (about 100 words) that would "
        "answer it, in the style of such an article. Write the passage in the language named "
        "in the request, whatever the language of the query. If you do not know the facts, "
        "write a plausible passage anyway. Return only the passage, with no title and no commentary.",
        input_template="Language of the passage: {collection_language}\nQuery: {query}",
    ),
    "keywords": PromptTemplate(
        version="keywords-v1",
        instructions=_ARCHIVE
        + "Given a search query, list 10 to 20 search terms that a relevant article would "
        "contain: key entities, synonyms, related concepts, and spelling variants a period "
        "newspaper might use. Use the language of the query. "
        "Return only the terms, separated by commas.",
        input_template="Query: {query}",
    ),
}

COMBINE_MODES: tuple[str, ...] = ("replace", "concat")


def combine(query: str, expansion: str, mode: str) -> str:
    """The text a backbone encodes for one expanded query."""
    expansion = expansion.strip()
    if mode == "replace":
        return expansion
    if mode == "concat":
        return f"{query}\n{expansion}"
    raise ValueError(f"unknown combine mode {mode!r}; known: {', '.join(COMBINE_MODES)}")


# --- configuration -----------------------------------------------------------------


@dataclass(frozen=True)
class ExpansionConfig:
    """Everything that decides what the LLM returns for a query; its hash names the file.

    `temperature` and `reasoning_effort` are sent only when set: reasoning models
    reject `temperature`, and a non-reasoning model rejects `reasoning`. Sampling is
    one completion per query (HyDE averages eight; that is eight times the cost).

    `generation` repeats the same request as an independent sample: generation 1, 2,
    ... get their own hash, file and tasks, so the spread of retrieval scores across
    generations shows whether one generation is stable. It is in the hash only when
    non-zero, so generation 0 keeps the hash it had before the field existed. The
    request itself is identical; the samples differ only by the model's sampling
    (none at temperature 0).
    """

    model: str
    variant: str
    max_output_tokens: int = 512
    temperature: float | None = None
    reasoning_effort: str | None = None
    generation: int = 0

    def __post_init__(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError(f"unknown variant {self.variant!r}; known: {', '.join(VARIANTS)}")
        if self.max_output_tokens < 16:
            raise ValueError("max_output_tokens below 16 cannot hold an expansion")
        if self.generation < 0:
            raise ValueError("generation must be 0 or positive")

    @property
    def prompt(self) -> PromptTemplate:
        return VARIANTS[self.variant]

    def payload(self) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "variant": self.variant,
            "prompt_version": self.prompt.version,
            "prompt_sha": self.prompt.sha,
            "max_output_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            "reasoning_effort": self.reasoning_effort,
        }
        if self.generation:
            payload["generation"] = self.generation
        return payload

    @property
    def hash(self) -> str:
        return config_hash(self.payload())

    def to_meta(self) -> dict[str, Any]:
        """What `config.json` holds: the hashed payload plus the prompt texts themselves."""
        return {
            "hash": self.hash,
            **self.payload(),
            "instructions": self.prompt.instructions,
            "input_template": self.prompt.input_template,
        }

    @classmethod
    def from_meta(cls, meta: dict[str, Any]) -> "ExpansionConfig":
        config = cls(
            model=meta["model"],
            variant=meta["variant"],
            max_output_tokens=meta["max_output_tokens"],
            temperature=meta.get("temperature"),
            reasoning_effort=meta.get("reasoning_effort"),
            generation=meta.get("generation", 0),
        )
        if config.prompt.sha != meta["prompt_sha"]:
            raise ValueError(
                f"expansion {meta.get('hash')}: prompt {meta['variant']} has changed since it was "
                f"run (sha {meta['prompt_sha']}, now {config.prompt.sha}); a changed prompt "
                "needs a new version, never an edit in place"
            )
        return config


# --- the queries ------------------------------------------------------------------

FORMS: tuple[str, ...] = ("crosslingual", "multilingual")
LANGUAGES: tuple[str, ...] = ("DE", "FR")


@dataclass(frozen=True)
class Query:
    """One Impresso query. `query_id` is `<form>/<language>/q<row>`; the `q<row>`
    part is the id `impresso.load_split` gives it inside its task."""

    query_id: str
    form: str
    language: str
    text: str

    @property
    def task_query_id(self) -> str:
        return self.query_id.rsplit("/", 1)[1]


def read_queries(root: str | Path, *, forms: Iterable[str] = FORMS, languages: Iterable[str] = LANGUAGES) -> list[Query]:
    """Every query of the chosen Impresso sets, in file order (no `mteb` needed)."""
    csv.field_size_limit(2**31 - 1)
    queries: list[Query] = []
    for form in forms:
        for language in languages:
            path = Path(root) / form / language / "queries.csv"
            if not path.exists():
                raise FileNotFoundError(f"{path} not found; set IMPRESSO_PATH")
            with open(path, encoding="utf-8", newline="") as handle:
                for n, row in enumerate(csv.DictReader(handle)):
                    queries.append(Query(f"{form}/{language}/q{n}", form, language, row["search_query"]))
    return queries


# --- task names -------------------------------------------------------------------

_NAME = re.compile(r"^(?P<base>Impresso[A-Za-z]+)__qe-(?P<variant>[a-z_]+)-(?P<hash>[0-9a-f]{%d})-(?P<combine>[a-z]+)$" % HASH_LENGTH)


def expanded_task_name(base: str, config: ExpansionConfig, mode: str) -> str:
    """`ImpressoMultilingualDE__qe-pseudodoc-<hash>-concat`: own cache directory, own
    MTEB result file, and readable without opening `config.json`."""
    if mode not in COMBINE_MODES:
        raise ValueError(f"unknown combine mode {mode!r}")
    return f"{base}__qe-{config.variant}-{config.hash}-{mode}"


def parse_expanded_task_name(name: str) -> dict[str, str] | None:
    """`base`, `variant`, `hash`, `combine` of an expanded task name, or None."""
    match = _NAME.match(name)
    if match is None or match["combine"] not in COMBINE_MODES:
        return None
    return match.groupdict()


# --- the store --------------------------------------------------------------------

CONFIG_FILE = "config.json"
RECORDS_FILE = "expansions.jsonl"


def expansion_directory(root: str | Path, expansion_hash: str) -> Path:
    return Path(root) / expansion_hash


def read_records(path: str | Path) -> list[dict[str, Any]]:
    """Every record of one `expansions.jsonl`. A truncated *last* line (a kill mid-write)
    is dropped; a bad line anywhere else is damage and raises."""
    path = Path(path)
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    records = []
    for number, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if number == len(lines) - 1:
                break
            raise ValueError(f"{path}:{number + 1} is not JSON")
    return records


def final_records(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """The settled record per query id: the last one with a response. Error records
    (no response) never settle a query, so a rerun retries them."""
    settled: dict[str, dict[str, Any]] = {}
    for record in records:
        if record.get("status") != "error":
            settled[record["query_id"]] = record
    return settled


def load_config(root: str | Path, expansion_hash: str) -> ExpansionConfig:
    path = expansion_directory(root, expansion_hash) / CONFIG_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"no query expansion {expansion_hash} under {root} ({path} missing); run "
            "scripts/expand_queries.py, or copy the expansion directory here (QUERY_EXPANSION_PATH)"
        )
    config = ExpansionConfig.from_meta(json.loads(path.read_text(encoding="utf-8")))
    if config.hash != expansion_hash:
        raise ValueError(f"{path} hashes to {config.hash}, not {expansion_hash}")
    return config


def load_expansions(root: str | Path, expansion_hash: str) -> dict[str, dict[str, Any]]:
    """Settled records of one expansion, by query id. Raises on a hash mismatch."""
    load_config(root, expansion_hash)
    return final_records(read_records(expansion_directory(root, expansion_hash) / RECORDS_FILE))


def usable(record: dict[str, Any]) -> bool:
    return record.get("status") == "completed" and bool(str(record.get("expansion", "")).strip())


# --- the LLM call -----------------------------------------------------------------


@dataclass
class Completion:
    text: str
    status: str
    """`completed`, or `incomplete` (e.g. `max_output_tokens` hit, reasoning included)."""
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int = 0
    response_model: str | None = None
    response_id: str | None = None
    incomplete_reason: str | None = None


class Expander(Protocol):
    def expand(self, config: ExpansionConfig, instructions: str, text: str) -> Completion: ...

    def is_fatal(self, error: BaseException) -> bool: ...


class OpenAIExpander:
    """One Responses API call per query (openai>=3.41, `client.responses.create`).

    The key comes from `OPENAI_API_KEY` and nowhere else. `store=False`: nothing is
    kept server-side. The SDK's own retries (`max_retries`, exponential backoff on
    429/5xx/timeouts) run inside the timed call, so a recorded latency includes them.
    """

    def __init__(self, *, timeout: float = 60.0, max_retries: int = 3) -> None:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise SystemExit("OPENAI_API_KEY is not set; it is the only place the key is read from")
        import openai

        self._openai = openai
        self.client = openai.OpenAI(api_key=key, timeout=timeout, max_retries=max_retries)

    def expand(self, config: ExpansionConfig, instructions: str, text: str) -> Completion:
        kwargs: dict[str, Any] = {
            "model": config.model,
            "instructions": instructions,
            "input": text,
            "max_output_tokens": config.max_output_tokens,
            "store": False,
        }
        if config.temperature is not None:
            kwargs["temperature"] = config.temperature
        if config.reasoning_effort is not None:
            kwargs["reasoning"] = {"effort": config.reasoning_effort}
        response = self.client.responses.create(**kwargs)
        usage = response.usage
        details = getattr(usage, "output_tokens_details", None) if usage else None
        incomplete = getattr(response, "incomplete_details", None)
        return Completion(
            text=response.output_text or "",
            status=str(response.status or "completed"),
            input_tokens=int(usage.input_tokens) if usage else 0,
            output_tokens=int(usage.output_tokens) if usage else 0,
            reasoning_tokens=int(getattr(details, "reasoning_tokens", 0) or 0),
            response_model=response.model,
            response_id=response.id,
            incomplete_reason=getattr(incomplete, "reason", None) if incomplete else None,
        )

    def is_fatal(self, error: BaseException) -> bool:
        """Errors no retry fixes: a bad key, a missing model, a rejected parameter."""
        errors = self._openai
        return isinstance(
            error,
            (errors.AuthenticationError, errors.PermissionDeniedError, errors.NotFoundError, errors.BadRequestError),
        )


class StubExpander:
    """Deterministic, offline expander for tests and pipeline checks. No network."""

    def __init__(self, respond: Callable[[str, str], str] | None = None, *, latency_s: float = 0.0) -> None:
        self.respond = respond or (lambda instructions, text: f"expanded: {text.split(': ', 1)[-1]}")
        self.latency_s = latency_s
        self.calls = 0
        self._lock = threading.Lock()

    def expand(self, config: ExpansionConfig, instructions: str, text: str) -> Completion:
        with self._lock:
            self.calls += 1
        if self.latency_s:
            time.sleep(self.latency_s)
        output = self.respond(instructions, text)
        return Completion(
            text=output,
            status="completed",
            input_tokens=len((instructions + " " + text).split()),
            output_tokens=len(output.split()),
            response_model=f"stub/{config.model}",
            response_id=f"stub-{self.calls}",
        )

    def is_fatal(self, error: BaseException) -> bool:
        return False


# --- tokens, prices, budget -------------------------------------------------------

REQUEST_OVERHEAD_TOKENS = 8
"""Message framing the API adds per request on top of instructions and input; a
guess, small next to ~100 tokens of instructions."""


class TokenCounter:
    """`o200k_base` (tiktoken) when installed -- the encoding of OpenAI's current models;
    a future model may use another, so counts are an estimate either way -- else 4
    characters per token, which runs high for English and low for German."""

    def __init__(self) -> None:
        try:
            import tiktoken

            self._encoding = tiktoken.get_encoding("o200k_base")
            self.method = "tiktoken o200k_base"
        except Exception:  # not installed, or its BPE file cannot be fetched offline
            self._encoding = None
            self.method = "heuristic: 4 characters per token"

    def count(self, text: str) -> int:
        if self._encoding is not None:
            return len(self._encoding.encode(text))
        return math.ceil(len(text) / 4)


@dataclass(frozen=True)
class Price:
    """USD per million tokens. Reasoning tokens are billed as output."""

    input_per_mtok: float
    output_per_mtok: float

    def cost(self, input_tokens: float, output_tokens: float) -> float:
        return (input_tokens * self.input_per_mtok + output_tokens * self.output_per_mtok) / 1e6


@dataclass
class Spend:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def add(self, record: dict[str, Any]) -> None:
        self.calls += 1
        self.input_tokens += int(record.get("input_tokens") or 0)
        self.output_tokens += int(record.get("output_tokens") or 0)


def spend_under(root: str | Path) -> Spend:
    """Every call ever recorded under `root`, all expansions together: one key, one pot."""
    spend = Spend()
    for path in sorted(Path(root).glob(f"*/{RECORDS_FILE}")):
        for record in read_records(path):
            spend.add(record)
    return spend


@dataclass
class Budget:
    """Hard caps on the cumulative spend under one expansion root.

    A call is admitted only if its worst case fits: the input estimate with a
    margin, plus all of `max_output_tokens`. Admitted calls hold a reservation until
    they settle, so concurrent workers cannot overshoot together. What can exceed a
    cap is only an input count above its estimate (margin 25 % + 32 tokens).
    """

    max_calls: int
    max_tokens: int
    max_cost_usd: float | None = None
    price: Price | None = None
    spent: Spend = field(default_factory=Spend)
    _reserved_calls: int = 0
    _reserved_input: int = 0
    _reserved_output: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        if self.max_cost_usd is not None and self.price is None:
            raise ValueError("a cost cap needs a price")

    @staticmethod
    def worst_case(estimated_input: int, max_output_tokens: int) -> tuple[int, int]:
        return math.ceil(estimated_input * 1.25) + 32, max_output_tokens

    def _fits(self, calls: int, input_tokens: int, output_tokens: int) -> str | None:
        if calls > self.max_calls:
            return f"call cap {self.max_calls}"
        if input_tokens + output_tokens > self.max_tokens:
            return f"token cap {self.max_tokens:,}"
        if self.max_cost_usd is not None and self.price.cost(input_tokens, output_tokens) > self.max_cost_usd:
            return f"cost cap ${self.max_cost_usd:.2f}"
        return None

    def reserve(self, input_tokens: int, output_tokens: int) -> str | None:
        """Reserve one call; returns None on success, else which cap it would break."""
        with self._lock:
            refused = self._fits(
                self.spent.calls + self._reserved_calls + 1,
                self.spent.input_tokens + self._reserved_input + input_tokens,
                self.spent.output_tokens + self._reserved_output + output_tokens,
            )
            if refused is None:
                self._reserved_calls += 1
                self._reserved_input += input_tokens
                self._reserved_output += output_tokens
            return refused

    def settle(self, reserved: tuple[int, int], record: dict[str, Any]) -> None:
        with self._lock:
            self._reserved_calls -= 1
            self._reserved_input -= reserved[0]
            self._reserved_output -= reserved[1]
            self.spent.add(record)


class RateLimiter:
    """Call starts at least `60 / requests_per_minute` seconds apart, across threads."""

    def __init__(self, requests_per_minute: float) -> None:
        self.interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0.0
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next)
            self._next = start + self.interval
        if start > now:
            time.sleep(start - now)


# --- the run ----------------------------------------------------------------------


def request_text(config: ExpansionConfig, query: Query) -> str:
    return config.prompt.render(query.text, COLLECTION_LANGUAGES[query.language])


def pending_queries(
    queries: list[Query], settled: dict[str, dict[str, Any]], *, retry_unusable: bool = False
) -> list[Query]:
    """Queries without a settled record -- or, with `retry_unusable`, without a usable one
    (an empty or truncated answer; asking again with the same config costs again)."""
    return [
        query
        for query in queries
        if query.query_id not in settled or (retry_unusable and not usable(settled[query.query_id]))
    ]


def estimate(
    config: ExpansionConfig,
    queries: list[Query],
    counter: TokenCounter,
    *,
    expected_output_tokens: int,
    price: Price | None = None,
) -> dict[str, Any]:
    """Token counts and cost of expanding `queries` with `config`, before any call.

    Input is counted exactly as sent (instructions + rendered input + framing
    overhead). Output can only be assumed: `expected_output_tokens` per call, and
    the worst case, every call using all of `max_output_tokens` (which, on a
    reasoning model, includes its reasoning tokens).
    """
    instructions = counter.count(config.prompt.instructions)
    inputs = sum(instructions + counter.count(request_text(config, q)) + REQUEST_OVERHEAD_TOKENS for q in queries)
    expected_out = expected_output_tokens * len(queries)
    worst_out = config.max_output_tokens * len(queries)
    result: dict[str, Any] = {
        "expansion": config.hash,
        "model": config.model,
        "variant": config.variant,
        "calls": len(queries),
        "input_tokens": inputs,
        "expected_output_tokens": expected_out,
        "worst_output_tokens": worst_out,
        "token_count": counter.method,
    }
    if price is not None:
        result["expected_cost_usd"] = price.cost(inputs, expected_out)
        result["worst_cost_usd"] = price.cost(inputs, worst_out)
    return result


def write_config(directory: Path, config: ExpansionConfig) -> None:
    path = directory / CONFIG_FILE
    meta = config.to_meta()
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != meta:
            raise ValueError(f"{path} describes another config than {config.hash}")
        return
    directory.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")


def run_expansion(
    config: ExpansionConfig,
    queries: list[Query],
    expander: Expander,
    root: str | Path,
    budget: Budget,
    *,
    requests_per_minute: float = 60.0,
    workers: int = 4,
    retry_unusable: bool = False,
    max_consecutive_errors: int = 5,
    counter: TokenCounter | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Expand every pending query of `queries`, appending one JSON line per call.

    Resumable: settled queries are skipped, and each line is written and flushed as
    its call returns, so a killed run loses at most the calls in flight. Stops
    early -- never mid-line -- when the budget refuses the next call, on an error the
    expander calls fatal, or after `max_consecutive_errors` errors in a row. The
    budget must already hold the spend under `root` (`spend_under`).
    """
    directory = expansion_directory(root, config.hash)
    write_config(directory, config)
    records_path = directory / RECORDS_FILE
    settled = final_records(read_records(records_path))
    todo = pending_queries(queries, settled, retry_unusable=retry_unusable)
    counter = counter or TokenCounter()
    instructions = config.prompt.instructions
    instruction_tokens = counter.count(instructions)
    limiter = RateLimiter(requests_per_minute)

    lock = threading.Lock()
    stop = threading.Event()
    state: dict[str, Any] = {"next": 0, "new": 0, "errors": 0, "consecutive": 0, "stopped": None}

    def take() -> Query | None:
        with lock:
            if stop.is_set() or state["next"] >= len(todo):
                return None
            query = todo[state["next"]]
            state["next"] += 1
            return query

    def halt(reason: str) -> None:
        with lock:
            if state["stopped"] is None:
                state["stopped"] = reason
        stop.set()

    def work(handle: Any) -> None:
        while (query := take()) is not None:
            text = request_text(config, query)
            reserved = Budget.worst_case(
                instruction_tokens + counter.count(text) + REQUEST_OVERHEAD_TOKENS, config.max_output_tokens
            )
            refused = budget.reserve(*reserved)
            if refused is not None:
                halt(f"budget: the next call could exceed the {refused}")
                return
            limiter.wait()
            record: dict[str, Any] = {
                "query_id": query.query_id,
                "query": query.text,
                "expansion_hash": config.hash,
                "model": config.model,
                "variant": config.variant,
                "prompt_sha": config.prompt.sha,
                "started_at": time.time(),
            }
            started = time.perf_counter()
            try:
                completion = expander.expand(config, instructions, text)
            except Exception as error:  # recorded, counted as a call, retried on the next run
                record.update(
                    status="error",
                    error=f"{type(error).__name__}: {error}",
                    latency_s=time.perf_counter() - started,
                    input_tokens=0,
                    output_tokens=0,
                )
                fatal = expander.is_fatal(error)
                with lock:
                    state["errors"] += 1
                    state["consecutive"] += 1
                    too_many = state["consecutive"] >= max_consecutive_errors
                if fatal or too_many:
                    halt(f"{'fatal error' if fatal else f'{max_consecutive_errors} errors in a row'}: {record['error']}")
            else:
                record.update(
                    status=completion.status,
                    expansion=completion.text,
                    latency_s=time.perf_counter() - started,
                    input_tokens=completion.input_tokens,
                    output_tokens=completion.output_tokens,
                    reasoning_tokens=completion.reasoning_tokens,
                    response_model=completion.response_model,
                    response_id=completion.response_id,
                    incomplete_reason=completion.incomplete_reason,
                )
                with lock:
                    state["consecutive"] = 0
            budget.settle(reserved, record)
            with lock:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                if record["status"] != "error":
                    state["new"] += 1
                done = state["new"]
            if progress is not None and done and done % 50 == 0:
                progress(f"{config.hash}: {done}/{len(todo)} expanded, {budget.spent.calls} calls under the root")

    directory.mkdir(parents=True, exist_ok=True)
    with open(records_path, "a", encoding="utf-8") as handle:
        threads = [threading.Thread(target=work, args=(handle,), daemon=True) for _ in range(max(1, workers))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    settled = final_records(read_records(records_path))
    wanted = {query.query_id for query in queries}
    return {
        "expansion": config.hash,
        "directory": str(directory),
        "queries": len(queries),
        "pending_at_start": len(todo),
        "newly_expanded": state["new"],
        "errors": state["errors"],
        "usable": sum(1 for key in wanted if key in settled and usable(settled[key])),
        "stopped": state["stopped"],
        "spent": asdict(budget.spent),
    }


# --- latency ----------------------------------------------------------------------


def percentiles(values: list[float], points: Iterable[float] = (50, 95)) -> dict[str, float]:
    """Nearest-rank-interpolated percentiles (numpy's default), plus mean and n."""
    import numpy as np

    if not values:
        return {"n": 0}
    array = np.asarray(values, dtype=float)
    return {"n": len(values), "mean": float(array.mean()), **{f"p{p:g}": float(np.percentile(array, p)) for p in points}}


def api_latency(root: str | Path, expansion_hash: str) -> dict[str, Any]:
    """Wall-clock of the recorded calls of one expansion that returned a response."""
    records = read_records(expansion_directory(root, expansion_hash) / RECORDS_FILE)
    return percentiles([float(r["latency_s"]) for r in records if r.get("status") != "error" and "latency_s" in r])
