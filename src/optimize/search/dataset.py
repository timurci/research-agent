"""Search-suite optimization dataset loader.

Loads the Hugging Face ``tcakmako/research_queries`` **train** split and
maps rows to domain ``ResearchQuery`` values and DSPy examples. No gold
papers or relevance labels are present.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import dspy
from datasets import load_dataset

from research_agent.search.models import ResearchQuery

if TYPE_CHECKING:
    from collections.abc import Mapping

SEARCH_HF_PATH: str = "tcakmako/research_queries"
SEARCH_SPLIT: str = "train"


def research_query_from_row(row: Mapping[str, object]) -> ResearchQuery:
    """Map one HF row to a ``ResearchQuery``.

    Expects ``text: str`` and optional ``domains: list[str]``.
    """
    text, domains = row["text"], row.get("domains")
    if not isinstance(text, str):
        msg = f"text must be str, got {type(text).__name__}"
        raise TypeError(msg)
    if domains is not None and not (
        isinstance(domains, list) and all(isinstance(d, str) for d in domains)
    ):
        msg = "domains must be list[str] or None"
        raise TypeError(msg)
    return ResearchQuery.model_validate(
        {"text": text, "domains": domains or None},
    )


def load_search_queries(
    *,
    path: str = SEARCH_HF_PATH,
    split: str = SEARCH_SPLIT,
) -> list[ResearchQuery]:
    """Load search-suite queries from Hugging Face.

    Args:
        path: Hugging Face dataset path.
        split: Dataset split name (default ``train``).

    Returns:
        Domain queries in dataset order.
    """
    dataset = load_dataset(path, split=split)
    return [research_query_from_row(row) for row in dataset]


def load_search_trainset(
    *,
    path: str = SEARCH_HF_PATH,
    split: str = SEARCH_SPLIT,
) -> list[dspy.Example]:
    """Load DSPy examples for search optimization.

    Each example carries ``research_query`` (matching
    ``SearchAgentSignature`` input) marked with ``.with_inputs()``.

    Args:
        path: Hugging Face dataset path.
        split: Dataset split name (default ``train``).

    Returns:
        DSPy examples suitable for GEPA / other optimizers.
    """
    return [
        dspy.Example(research_query=query).with_inputs("research_query")
        for query in load_search_queries(path=path, split=split)
    ]
