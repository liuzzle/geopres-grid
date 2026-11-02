#!/usr/bin/env python
"""Expand the Impresso queries with one OpenAI model, once, offline (logging.md §71).

Each query is sent alone -- no document -- with a versioned prompt
(`expansion.VARIANTS`); the answers go to
`$QUERY_EXPANSION_PATH/<expansion hash>/expansions.jsonl`, one line per call, keyed
by query id, model and prompt sha, with token usage and wall-clock latency. A rerun
resumes: settled queries are skipped, calls that raised are retried.

Spending is capped by `--max-total-calls` and `--max-total-tokens` (and
`--max-total-cost-usd` with prices), counted over *every* expansion under the root,
so neither a rerun nor a second variant can spend past them. The key is read from
`OPENAI_API_KEY` only. `--dry-run` makes no call and prints the cost estimate.

    uv run --extra expansion python scripts/expand_queries.py --model <model> --variant pseudodoc \\
        --dry-run --price-input 2 --price-output 10
    OPENAI_API_KEY=... uv run --extra expansion python scripts/expand_queries.py --model <model> \\
        --variant pseudodoc --max-total-calls 6000 --max-total-tokens 4000000 \\
        --price-input 2 --price-output 10 --max-total-cost-usd 20

`--stub` runs the whole pipeline with `expansion.StubExpander` instead of the API
(into a separate root unless `--out-root` says otherwise), for checking the path
end to end without a key.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from geopres_grid import config
from geopres_grid.expansion import (
    COMBINE_MODES,
    FORMS,
    LANGUAGES,
    VARIANTS,
    Budget,
    ExpansionConfig,
    OpenAIExpander,
    Price,
    StubExpander,
    TokenCounter,
    estimate,
    expanded_task_name,
    expansion_directory,
    final_records,
    pending_queries,
    read_queries,
    read_records,
    run_expansion,
    spend_under,
)

EXPECTED_OUTPUT_TOKENS = {"pseudodoc": 170, "pseudodoc_doclang": 190, "keywords": 70}
"""Assumed visible output per call for the dry run: ~100 words of English is ~130
tokens, German and French run longer; 10-20 comma-separated terms ~50-80. Reasoning
tokens come on top and are unknown before a call -- the worst case covers them."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="OpenAI model name, e.g. the one Andrianos' key is for")
    parser.add_argument("--variant", action="append", choices=sorted(VARIANTS), help="Repeatable; default pseudodoc and keywords")
    parser.add_argument("--form", action="append", choices=FORMS, help="Query sets; default both")
    parser.add_argument("--language", action="append", choices=LANGUAGES, help="Default both")
    parser.add_argument("--limit", type=int, help="Only the first N queries of each set (a pilot)")
    parser.add_argument("--max-output-tokens", type=int, help="Per call, reasoning included; default 512 (keywords 256)")
    parser.add_argument("--temperature", type=float, help="Sent only when given (reasoning models reject it)")
    parser.add_argument("--reasoning-effort", help="Sent only when given, e.g. minimal/low (reasoning models)")
    parser.add_argument("--dry-run", action="store_true", help="No call: print token counts and the cost estimate")
    parser.add_argument("--stub", action="store_true", help="Use the offline stub expander, not the API")
    parser.add_argument("--price-input", type=float, help="USD per 1M input tokens")
    parser.add_argument("--price-output", type=float, help="USD per 1M output tokens (reasoning included)")
    parser.add_argument("--max-total-calls", type=int, help="Hard cap, all expansions under the root together")
    parser.add_argument("--max-total-tokens", type=int, help="Hard cap on input + output tokens, all expansions")
    parser.add_argument("--max-total-cost-usd", type=float, help="Hard cap; needs both prices")
    parser.add_argument("--requests-per-minute", type=float, default=60.0)
    parser.add_argument("--workers", type=int, default=4, help="Concurrent calls")
    parser.add_argument("--timeout", type=float, default=60.0, help="Seconds per API attempt")
    parser.add_argument("--max-retries", type=int, default=3, help="SDK retries per call (429, 5xx, timeouts)")
    parser.add_argument("--retry-unusable", action="store_true", help="Ask again for empty or truncated answers")
    parser.add_argument("--impresso-root", default=config.IMPRESSO_PATH)
    parser.add_argument("--out-root", help=f"Default {config.QUERY_EXPANSION_PATH} ({config.QUERY_EXPANSION_PATH}_stub with --stub)")
    return parser


def expansion_config(args: argparse.Namespace, variant: str) -> ExpansionConfig:
    default_max = 256 if variant == "keywords" else 512
    return ExpansionConfig(
        # A stub answer must never sit under a real model's hash, whatever --out-root is.
        model=f"stub:{args.model}" if args.stub else args.model,
        variant=variant,
        max_output_tokens=args.max_output_tokens or default_max,
        temperature=args.temperature,
        reasoning_effort=args.reasoning_effort,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    variants = args.variant or ["pseudodoc", "keywords"]
    root = Path(args.out_root or (config.QUERY_EXPANSION_PATH + ("_stub" if args.stub else "")))
    if (args.price_input is None) != (args.price_output is None):
        raise SystemExit("give both --price-input and --price-output, or neither")
    price = Price(args.price_input, args.price_output) if args.price_input is not None else None

    queries = read_queries(args.impresso_root, forms=args.form or FORMS, languages=args.language or LANGUAGES)
    if args.limit:
        queries = [q for q in queries if int(q.task_query_id[1:]) < args.limit]
    counter = TokenCounter()
    configs = [expansion_config(args, variant) for variant in variants]

    if args.dry_run:
        totals = {"calls": 0, "input_tokens": 0, "expected_output_tokens": 0, "worst_output_tokens": 0}
        print(f"DRY RUN -- no API call. {len(queries)} queries; tokens by {counter.method}; root {root}")
        for expansion in configs:
            settled = final_records(read_records(expansion_directory(root, expansion.hash) / "expansions.jsonl"))
            todo = pending_queries(queries, settled, retry_unusable=args.retry_unusable)
            row = estimate(
                expansion, todo, counter, expected_output_tokens=EXPECTED_OUTPUT_TOKENS[expansion.variant], price=price
            )
            for key in totals:
                totals[key] += row[key]
            print(json.dumps(row))
        if price is not None:
            totals["expected_cost_usd"] = price.cost(totals["input_tokens"], totals["expected_output_tokens"])
            totals["worst_cost_usd"] = price.cost(totals["input_tokens"], totals["worst_output_tokens"])
        print("TOTAL " + json.dumps(totals))
        print(f"Already spent under {root}: {spend_under(root)}")
        return 0

    if args.max_total_calls is None or args.max_total_tokens is None:
        raise SystemExit("a live run needs --max-total-calls and --max-total-tokens (a hard cap is not optional)")
    budget = Budget(
        max_calls=args.max_total_calls,
        max_tokens=args.max_total_tokens,
        max_cost_usd=args.max_total_cost_usd,
        price=price,
        spent=spend_under(root),
    )
    expander = StubExpander() if args.stub else OpenAIExpander(timeout=args.timeout, max_retries=args.max_retries)
    print(f"Root {root}; already spent {budget.spent}", flush=True)
    exit_code = 0
    for expansion in configs:
        summary = run_expansion(
            expansion,
            queries,
            expander,
            root,
            budget,
            requests_per_minute=args.requests_per_minute,
            workers=args.workers,
            retry_unusable=args.retry_unusable,
            counter=counter,
            progress=lambda line: print(line, flush=True),
        )
        print(json.dumps(summary), flush=True)
        if summary["usable"] == summary["queries"]:
            bases = sorted({f"Impresso{q.form.capitalize()}{q.language}" for q in queries})
            print(f"{expansion.variant} complete: --query-expansion {expansion.hash}, e.g. task "
                  f"{expanded_task_name(bases[0], expansion, COMBINE_MODES[1])}")
        else:
            exit_code = 1
        if summary["stopped"]:
            print(f"Stopped: {summary['stopped']}", file=sys.stderr)
            return 1
    if price is not None:
        print(f"Spent under {root}: {budget.spent} = ${price.cost(budget.spent.input_tokens, budget.spent.output_tokens):.4f}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
