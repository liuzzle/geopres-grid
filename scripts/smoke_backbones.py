"""WP-A exit criterion: load every backbone on CPU and check the registry against it.

This is the test that proves the sentence-transformers 5.3 -> 5.7 bump was both
necessary and sufficient. Necessary because `lightonai/mDenseOn` declares its
modules under the refactored `sentence_transformers.base.modules.*` paths, which
do not exist before 5.4. Sufficient because the other four still declare the
legacy `sentence_transformers.models.*` paths, which 5.7 resolves transparently
through its deprecation shim onto the same new classes, so they keep working.

Everything here runs on CPU. No GPU is needed to establish that the model set loads.

    uv run python scripts/smoke_backbones.py
    uv run python scripts/smoke_backbones.py --models mdenseon lfm25

Backbones that cannot run under the installed `transformers` major are skipped with
the command that would run them, rather than counted as failures -- the model set
is split across two environments on purpose (see `envs/README.md`).

    uv run --project envs/transformers5 python scripts/smoke_backbones.py
"""

import argparse
import logging
import sys
import time

import torch
from sentence_transformers import SentenceTransformer

from geopres_grid import backbones
from geopres_grid.identity import EncodeConfig

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

PROBE = "Compressing an embedding index trades storage for retrieval quality."


class CheckFailed(Exception):
    """A registry claim did not survive contact with the loaded model."""


def check(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailed(message)


def module_classes(model: SentenceTransformer) -> list[str]:
    """Fully qualified class path of each module in the stack, in order."""
    return [
        f"{type(m).__module__}.{type(m).__name__}" for m in model._modules.values()
    ]


def probe(backbone: backbones.Backbone, max_seq_length: int | None) -> dict:
    """Load one backbone, verify the registry against it, and encode one sentence."""
    started = time.perf_counter()

    model = SentenceTransformer(
        backbone.model_id,
        revision=backbone.revision,
        trust_remote_code=backbone.trust_remote_code,
        device="cpu",
    )
    load_seconds = time.perf_counter() - started

    classes = module_classes(model)
    # Renamed in sentence-transformers 5.7; the old name still works but warns.
    get_dim = getattr(
        model, "get_embedding_dimension", model.get_sentence_embedding_dimension
    )
    reported_dim = get_dim()
    normalizes = classes[-1].endswith("Normalize")
    # sentence-transformers 5.7 injects empty `query`/`document` defaults for models
    # that declare none. An empty prefix changes nothing, so compare only real ones.
    prompts = {k: v for k, v in (model.prompts or {}).items() if v} or None

    check(
        reported_dim == backbone.native_dim,
        f"registry says d={backbone.native_dim}, model reports d={reported_dim}",
    )
    check(
        normalizes == backbone.normalizes,
        f"registry says normalizes={backbone.normalizes}, stack ends in {classes[-1]}",
    )
    check(
        bool(prompts) == bool(backbone.prompts),
        f"registry says prompts={backbone.prompts!r}, model reports {prompts!r}",
    )
    if backbone.prompts:
        check(
            prompts == backbone.prompts,
            f"prompt mismatch:\n  registry: {backbone.prompts!r}\n  model:    {prompts!r}",
        )

    # The per-side names are what encode() is actually called with, so a name the
    # model does not declare is a ValueError at encode time -- except when a cache
    # wrapper swallows it, in which case it is a silently unprefixed query instead.
    # Catch it here, against the loaded model, rather than three work packages later.
    for side in ("query", "document"):
        name = backbone.prompt_name_for(side)
        if name is None:
            continue
        check(
            name in (model.prompts or {}),
            f"registry uses prompt_name={name!r} for the {side} side, but the loaded "
            f"model declares {sorted(model.prompts or {})}",
        )
        check(
            (model.prompts or {}).get(name) == backbone.prompt_for(side),
            f"{side} prefix mismatch for {name!r}:\n"
            f"  registry: {backbone.prompt_for(side)!r}\n"
            f"  model:    {(model.prompts or {}).get(name)!r}",
        )

    native_max_seq = model.max_seq_length
    if backbone.native_max_seq_length is not None:
        check(
            native_max_seq == backbone.native_max_seq_length,
            f"registry says max_seq_length={backbone.native_max_seq_length}, "
            f"model reports {native_max_seq}",
        )

    if max_seq_length is not None:
        model.max_seq_length = max_seq_length
        check(
            model.max_seq_length == max_seq_length,
            f"setting max_seq_length={max_seq_length} did not take effect "
            f"(still {model.max_seq_length})",
        )

    vector = model.encode(PROBE, convert_to_numpy=True)
    check(
        vector.shape == (backbone.native_dim,),
        f"encode returned shape {vector.shape}, expected ({backbone.native_dim},)",
    )
    check(bool(torch.isfinite(torch.from_numpy(vector)).all()), "encode produced non-finite values")

    norm = float((vector**2).sum() ** 0.5)
    if backbone.normalizes:
        check(abs(norm - 1.0) < 1e-3, f"stack ends in Normalize but ||v|| = {norm:.4f}")

    # WP-B: the cache key this model would produce, built from the live object
    # rather than from the registry, so it records what the encoder will actually do.
    encode_config = EncodeConfig.from_model(
        model,
        model_id=backbone.model_id,
        revision=backbone.revision,
        device="cpu",
    )

    del model
    return {
        "encode_config": encode_config,
        "classes": classes,
        "dim": reported_dim,
        "native_max_seq": native_max_seq,
        "norm": norm,
        "load_seconds": load_seconds,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Backbone keys to probe. Defaults to all of them.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=backbones.PRIMARY_MAX_SEQ_LENGTH,
        help="Value to apply and verify. Pass 0 to skip and use each model's native value.",
    )
    args = parser.parse_args()

    # Each model can take minutes to fetch; stream progress instead of buffering
    # it all until the end when stdout is redirected to a file.
    sys.stdout.reconfigure(line_buffering=True)

    selected = (
        [backbones.get(k) for k in args.models]
        if args.models
        else backbones.all_backbones()
    )
    max_seq_length = args.max_seq_length or None

    major = backbones.installed_transformers_major()
    runnable = [b for b in selected if major in b.transformers_majors]
    skipped = [b for b in selected if major not in b.transformers_majors]

    print(f"sentence-transformers {__import__('sentence_transformers').__version__}")
    print(f"transformers          {__import__('transformers').__version__}")
    print(f"torch                 {torch.__version__}  (device: cpu)")
    print(f"applying max_seq_length = {max_seq_length or 'native'}\n")

    if skipped:
        print("skipped -- wrong environment for these:")
        for backbone in skipped:
            wants = ", ".join(
                backbones.ENVIRONMENTS.get(m, f"transformers {m}.x")
                for m in backbone.transformers_majors
            )
            print(f"  {backbone.key:<14} needs {wants}")
        print()

    failures: list[tuple[str, str]] = []

    for backbone in runnable:
        print(f"{'=' * 72}\n{backbone.key}  --  {backbone.model_id}\n{'=' * 72}")
        try:
            result = probe(backbone, max_seq_length)
        except Exception as exc:  # noqa: BLE001 - the report is the product
            failures.append((backbone.key, f"{type(exc).__name__}: {exc}"))
            print(f"  FAIL  {type(exc).__name__}: {exc}\n")
            continue

        print(f"  dim                {result['dim']}")
        print(f"  native max_seq     {result['native_max_seq']}"
              + ("  (from tokenizer, model declares none)"
                 if backbone.native_max_seq_length is None else ""))
        print(f"  ||encode(probe)||  {result['norm']:.4f}")
        print(f"  asymmetric prompts {backbone.has_asymmetric_prompts}"
              f"  (query={backbone.prompt_name_for('query')!r}, "
              f"document={backbone.prompt_name_for('document')!r})")
        print(f"  loaded in          {result['load_seconds']:.1f}s")
        config = result["encode_config"]
        print(f"  encode_config_hash {config.hash}")
        for repo, sha in config.external_code_revisions:
            state = sha[:12] if sha else "UNRESOLVED"
            print(f"  external code      {repo} @ {state}")
        if config.has_unpinned_code:
            print("                     ^ revision could not be pinned")
        print("  modules")
        for i, cls in enumerate(result["classes"]):
            print(f"    [{i}] {cls}")
        print("  PASS\n")

    print("=" * 72)
    if failures:
        print(f"{len(failures)} of {len(runnable)} backbones FAILED:")
        for key, message in failures:
            print(f"  {key}: {message}")
        return 1

    tail = f" ({len(skipped)} skipped: other environment)" if skipped else ""
    print(
        f"All {len(runnable)} backbones runnable on transformers {major}.x "
        f"load and encode on CPU.{tail}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
