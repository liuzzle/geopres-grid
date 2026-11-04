"""`scripts/probe_long_batch.py`: the probe batches it encodes (no GPU needed)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "probe_long_batch.py"


@pytest.fixture(scope="module")
def probe():
    spec = importlib.util.spec_from_file_location("probe_long_batch", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("batch_size", [2, 4, 32])
def test_probe_batches_are_padded_by_one_shorter_document(probe, batch_size):
    texts = probe.probe_texts(batch_size)
    assert len(texts) == batch_size
    assert len(set(texts[:-1])) == 1
    assert len(texts[-1]) < len(texts[0])


def test_a_single_document_batch_is_one_long_document(probe):
    (text,) = probe.probe_texts(1)
    assert text == probe.probe_texts(2)[0]


def test_long_documents_reach_the_cap_in_a_backbone_tokenizer(probe):
    tokenizers = pytest.importorskip("tokenizers")
    snapshots = Path.home() / ".cache" / "huggingface" / "hub" / "models--microsoft--harrier-oss-v1-0.6b"
    found = next(snapshots.glob("snapshots/*/tokenizer.json"), None)
    if found is None:
        pytest.skip("harrier-06b tokenizer not in the local HF cache")
    tokenizer = tokenizers.Tokenizer.from_file(str(found))
    assert len(tokenizer.encode(probe.probe_texts(1)[0]).ids) > 8192
