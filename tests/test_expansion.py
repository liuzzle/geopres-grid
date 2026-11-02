import json

import numpy as np
import pytest

from geopres_grid.expansion import (
    VARIANTS,
    Budget,
    ExpansionConfig,
    Price,
    Query,
    RateLimiter,
    StubExpander,
    TokenCounter,
    api_latency,
    combine,
    estimate,
    expanded_task_name,
    expansion_directory,
    final_records,
    load_config,
    load_expansions,
    parse_expanded_task_name,
    percentiles,
    read_records,
    run_expansion as run_unlimited,
    spend_under,
)

def run_expansion(*args, **kwargs):
    return run_unlimited(*args, requests_per_minute=0, **kwargs)


QUERIES = [Query(f"multilingual/DE/q{n}", "multilingual", "DE", f"frage {n}") for n in range(6)]


def make_budget(root, *, calls=1000, tokens=10**9, **kwargs):
    return Budget(max_calls=calls, max_tokens=tokens, spent=spend_under(root), **kwargs)


def config(variant="pseudodoc", **kwargs):
    return ExpansionConfig(model="gpt-test", variant=variant, **kwargs)


def test_the_hash_covers_model_prompt_and_sampling_but_is_stable():
    assert config().hash == config().hash
    assert len({config().hash, config("keywords").hash, ExpansionConfig("other", "pseudodoc").hash,
                config(max_output_tokens=300).hash, config(temperature=0.0).hash,
                config(reasoning_effort="low").hash}) == 6
    assert config().payload()["prompt_sha"] == VARIANTS["pseudodoc"].sha


def test_an_unknown_variant_is_refused():
    with pytest.raises(ValueError, match="unknown variant"):
        ExpansionConfig("m", "summary")


def test_a_prompt_edited_in_place_is_refused(tmp_path):
    meta = config().to_meta()
    meta["prompt_sha"] = "0" * 16

    with pytest.raises(ValueError, match="needs a new version"):
        ExpansionConfig.from_meta(meta)


def test_combine_modes():
    assert combine("q", " passage \n", "replace") == "passage"
    assert combine("q", "passage", "concat") == "q\npassage"
    with pytest.raises(ValueError):
        combine("q", "p", "prepend")


def test_task_names_round_trip():
    name = expanded_task_name("ImpressoCrosslingualFR", config("pseudodoc_doclang"), "concat")

    assert parse_expanded_task_name(name) == {
        "base": "ImpressoCrosslingualFR",
        "variant": "pseudodoc_doclang",
        "hash": config("pseudodoc_doclang").hash,
        "combine": "concat",
    }
    assert parse_expanded_task_name("ImpressoCrosslingualFR") is None
    assert parse_expanded_task_name(name.replace("concat", "prepend")) is None


def test_a_run_writes_one_keyed_record_per_query_and_resumes(tmp_path):
    expander = StubExpander()
    summary = run_expansion(config(), QUERIES, expander, tmp_path, make_budget(tmp_path), workers=3)

    assert summary["newly_expanded"] == summary["usable"] == 6 and summary["stopped"] is None
    records = read_records(expansion_directory(tmp_path, config().hash) / "expansions.jsonl")
    assert sorted(r["query_id"] for r in records) == sorted(q.query_id for q in QUERIES)
    first = next(r for r in records if r["query_id"] == "multilingual/DE/q0")
    assert first["model"] == "gpt-test" and first["prompt_sha"] == VARIANTS["pseudodoc"].sha
    assert first["expansion"] == "expanded: frage 0" and first["latency_s"] >= 0
    assert load_config(tmp_path, config().hash) == config()

    again = run_expansion(config(), QUERIES, expander, tmp_path, make_budget(tmp_path))
    assert expander.calls == 6 and again["pending_at_start"] == 0 and again["usable"] == 6


def test_the_call_cap_is_hard_and_counts_every_expansion_under_the_root(tmp_path):
    summary = run_expansion(config(), QUERIES, StubExpander(), tmp_path, make_budget(tmp_path, calls=4), workers=4)

    assert summary["newly_expanded"] == 4 and "call cap" in summary["stopped"]
    # Another variant shares the pot: 4 of 4 calls are already spent.
    other = run_expansion(config("keywords"), QUERIES, StubExpander(), tmp_path, make_budget(tmp_path, calls=4))
    assert other["newly_expanded"] == 0 and "call cap" in other["stopped"]
    assert spend_under(tmp_path).calls == 4


def test_the_token_cap_reserves_the_worst_case(tmp_path):
    from geopres_grid.expansion import REQUEST_OVERHEAD_TOKENS, request_text

    counter = TokenCounter()
    expansion = config(max_output_tokens=100)
    estimated = counter.count(expansion.prompt.instructions) + counter.count(request_text(expansion, QUERIES[0]))
    reserve = sum(Budget.worst_case(estimated + REQUEST_OVERHEAD_TOKENS, 100))

    refused = run_expansion(
        expansion, QUERIES, StubExpander(), tmp_path / "a", make_budget(tmp_path / "a", tokens=reserve - 1), counter=counter
    )
    # The first call's worst case fits; once its (stub) usage of ~60 tokens is spent, the second's does not.
    one = run_expansion(
        expansion, QUERIES, StubExpander(), tmp_path / "b", make_budget(tmp_path / "b", tokens=reserve + 10),
        counter=counter, workers=1,
    )

    assert refused["newly_expanded"] == 0 and "token cap" in refused["stopped"]
    assert one["newly_expanded"] == 1 and "token cap" in one["stopped"]


def test_the_cost_cap_needs_a_price_and_stops_the_run(tmp_path):
    with pytest.raises(ValueError, match="price"):
        Budget(max_calls=1, max_tokens=1, max_cost_usd=1.0)
    budget = make_budget(tmp_path, max_cost_usd=0.0, price=Price(1.0, 1.0))

    summary = run_expansion(config(), QUERIES, StubExpander(), tmp_path, budget)

    assert summary["newly_expanded"] == 0 and "cost cap" in summary["stopped"]


class Flaky(StubExpander):
    def __init__(self, failures, fatal=False):
        super().__init__()
        self.failures, self.fatal = failures, fatal

    def expand(self, config, instructions, text):
        if self.failures:
            self.failures -= 1
            raise RuntimeError("503")
        return super().expand(config, instructions, text)

    def is_fatal(self, error):
        return self.fatal


def test_errors_are_recorded_counted_and_retried_on_the_next_run(tmp_path):
    summary = run_expansion(config(), QUERIES, Flaky(2), tmp_path, make_budget(tmp_path), workers=1)

    assert summary["errors"] == 2 and summary["usable"] == 4 and summary["stopped"] is None
    assert spend_under(tmp_path).calls == 6  # the two failed calls count

    again = run_expansion(config(), QUERIES, StubExpander(), tmp_path, make_budget(tmp_path))
    assert again["pending_at_start"] == 2 and again["usable"] == 6


def test_a_fatal_error_stops_at_once(tmp_path):
    summary = run_expansion(config(), QUERIES, Flaky(10, fatal=True), tmp_path, make_budget(tmp_path), workers=1)

    assert summary["errors"] == 1 and summary["stopped"].startswith("fatal error")


def test_too_many_errors_in_a_row_stop_the_run(tmp_path):
    summary = run_expansion(
        config(), QUERIES, Flaky(10), tmp_path, make_budget(tmp_path), workers=1, max_consecutive_errors=3
    )

    assert summary["errors"] == 3 and "3 errors in a row" in summary["stopped"]


def test_an_empty_answer_settles_but_is_unusable_until_retried(tmp_path):
    run_expansion(config(), QUERIES, StubExpander(lambda i, t: ""), tmp_path, make_budget(tmp_path))
    assert run_expansion(config(), QUERIES, StubExpander(), tmp_path, make_budget(tmp_path))["usable"] == 0

    summary = run_expansion(config(), QUERIES, StubExpander(), tmp_path, make_budget(tmp_path), retry_unusable=True)

    assert summary["usable"] == 6
    assert all(r["expansion"] for r in load_expansions(tmp_path, config().hash).values())


def test_a_truncated_last_line_is_dropped_and_damage_elsewhere_raises(tmp_path):
    path = tmp_path / "expansions.jsonl"
    path.write_text('{"query_id": "a", "status": "completed"}\n{"query_id": "b", "sta')

    assert [r["query_id"] for r in read_records(path)] == ["a"]
    path.write_text('{"query_id": "a", "sta\n{"query_id": "b", "status": "completed"}\n')
    with pytest.raises(ValueError, match="not JSON"):
        read_records(path)


def test_final_records_keep_the_last_answer():
    records = [
        {"query_id": "a", "status": "completed", "expansion": "old"},
        {"query_id": "a", "status": "error"},
        {"query_id": "a", "status": "completed", "expansion": "new"},
    ]
    assert final_records(records)["a"]["expansion"] == "new"


def test_the_estimate_counts_inputs_and_bounds_the_output():
    counter = TokenCounter()
    row = estimate(config(max_output_tokens=200), QUERIES, counter, expected_output_tokens=50, price=Price(2.0, 10.0))

    assert row["calls"] == 6
    assert row["input_tokens"] > 6 * counter.count(VARIANTS["pseudodoc"].instructions)
    assert row["worst_output_tokens"] == 1200 and row["expected_output_tokens"] == 300
    assert row["worst_cost_usd"] == pytest.approx((row["input_tokens"] * 2 + 1200 * 10) / 1e6)


def test_the_rate_limiter_spaces_call_starts():
    import time

    limiter = RateLimiter(requests_per_minute=600)  # 0.1 s apart
    started = time.monotonic()
    for _ in range(4):
        limiter.wait()

    assert time.monotonic() - started >= 0.29


def test_api_latency_reads_the_recorded_calls(tmp_path):
    run_expansion(config(), QUERIES, StubExpander(latency_s=0.01), tmp_path, make_budget(tmp_path), workers=2)

    stats = api_latency(tmp_path, config().hash)
    assert stats["n"] == 6 and 0.01 <= stats["p50"] <= stats["p95"]
    assert percentiles([]) == {"n": 0}


# --- the script -------------------------------------------------------------------


def write_queries(root):
    import csv

    for form in ("crosslingual", "multilingual"):
        for language in ("DE", "FR"):
            folder = root / form / language
            folder.mkdir(parents=True)
            with open(folder / "queries.csv", "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["search_query", "idx"])
                writer.writeheader()
                writer.writerows([{"search_query": f"{form} {language} {n}", "idx": str(n)} for n in range(3)])


def test_the_dry_run_makes_no_call_and_needs_no_key(tmp_path, monkeypatch, capsys):
    import expand_queries

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(expand_queries, "OpenAIExpander", None)  # any use would raise
    write_queries(tmp_path / "impresso")

    assert expand_queries.main([
        "--model", "gpt-x", "--dry-run", "--price-input", "1", "--price-output", "4",
        "--impresso-root", str(tmp_path / "impresso"), "--out-root", str(tmp_path / "out"),
    ]) == 0

    out = capsys.readouterr().out
    rows = [json.loads(line) for line in out.splitlines() if line.startswith("{")]
    assert [row["variant"] for row in rows] == ["pseudodoc", "keywords"] and rows[0]["calls"] == 12
    assert "worst_cost_usd" in out and not (tmp_path / "out").exists()


def test_a_live_run_needs_caps_and_the_key(tmp_path, monkeypatch):
    import expand_queries

    write_queries(tmp_path / "impresso")
    common = ["--model", "gpt-x", "--impresso-root", str(tmp_path / "impresso"), "--out-root", str(tmp_path / "out")]
    with pytest.raises(SystemExit, match="hard cap"):
        expand_queries.main(common)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="OPENAI_API_KEY"):
        expand_queries.main(common + ["--max-total-calls", "5", "--max-total-tokens", "1000"])


def test_the_stub_run_goes_end_to_end_through_the_script(tmp_path, monkeypatch):
    import expand_queries

    write_queries(tmp_path / "impresso")
    args = ["--model", "gpt-x", "--stub", "--limit", "2", "--impresso-root", str(tmp_path / "impresso"),
            "--out-root", str(tmp_path / "out"), "--max-total-calls", "100", "--max-total-tokens", "100000",
            "--requests-per-minute", "0"]

    assert expand_queries.main(args) == 0
    assert spend_under(tmp_path / "out").calls == 2 * 4 * 2  # 2 variants, 4 sets, 2 queries each
    assert expand_queries.main(args) == 0
    assert spend_under(tmp_path / "out").calls == 16


def test_search_kernels_rank_the_corpus():
    from estimate_latency import search_kernel, search_latency

    rng = np.random.default_rng(0)
    assert len(search_kernel("binary", 128, 50, rng)()) == 10
    assert len(search_kernel("int4", 64, 5, rng)()) == 5
    rows = search_latency([64], ["none", "binary"], [100], repeats=3)
    assert [row["kernel"] for row in rows] == ["float32 matvec (dequantized)", "xor+popcount"]
    assert rows[1]["index_bytes"] == 100 * 64 // 8


def test_generation_zero_keeps_the_hash_and_others_get_their_own():
    first = config()
    assert first.hash == ExpansionConfig(model="gpt-test", variant="pseudodoc", generation=0).hash
    assert "generation" not in first.payload()
    second, third = config(generation=1), config(generation=2)
    assert len({first.hash, second.hash, third.hash}) == 3
    assert ExpansionConfig.from_meta(third.to_meta()) == third
    with pytest.raises(ValueError, match="generation"):
        config(generation=-1)
