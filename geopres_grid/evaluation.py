"""Version-pinned evaluation task tiers."""

from __future__ import annotations

from typing import Literal

Tier = Literal["tier0", "tier1", "tier2"]

NANOBEIR_TASKS: tuple[str, ...] = (
    "NanoArguAnaRetrieval",
    "NanoClimateFeverRetrieval",
    "NanoDBPediaRetrieval",
    "NanoFEVERRetrieval",
    "NanoFiQA2018Retrieval",
    "NanoHotpotQARetrieval",
    "NanoMSMARCOREtrieval",
    "NanoNFCorpusRetrieval",
    "NanoNQRetrieval",
    "NanoQuoraRetrieval",
    "NanoSCIDOCSRetrieval",
    "NanoSciFactRetrieval",
    "NanoTouche2020Retrieval",
)

TIER_TASKS: dict[Tier, tuple[str, ...]] = {
    "tier0": ("NanoArguAnaRetrieval", "STSBenchmark"),
    "tier1": NANOBEIR_TASKS,
    "tier2": (
        "ArguAna",
        "QuoraRetrieval",
        "HotpotQA",
        "NQ",
        "MSMARCO",
        "DBPedia",
    ),
}


def task_names(tier: Tier) -> tuple[str, ...]:
    """Return the immutable task list for one evaluation tier."""
    return TIER_TASKS[tier]