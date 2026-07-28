"""Search-suite evaluation dataset loader.

Loads the Hugging Face ``tcakmako/research_queries`` **test** split and
maps rows to domain ``ResearchQuery`` values for query-only Opik
evaluation. No gold papers or relevance labels are present.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from datasets import load_dataset

from research_agent.search.models import ResearchQuery

if TYPE_CHECKING:
    from collections.abc import Mapping

SEARCH_HF_PATH: str = "tcakmako/research_queries"
SEARCH_SPLIT: str = "test"


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
        split: Dataset split name (default ``test``).

    Returns:
        Domain queries in dataset order.
    """
    dataset = load_dataset(path, split=split)
    return [research_query_from_row(row) for row in dataset]


def load_search_eval_data(
    *,
    path: str = SEARCH_HF_PATH,
    split: str = SEARCH_SPLIT,
) -> list[dict[str, object]]:
    """Load Opik ``evaluate`` rows for the search suite.

    Each row is query-only::

        {"query": ResearchQuery(...)}

    Args:
        path: Hugging Face dataset path.
        split: Dataset split name (default ``test``).

    Returns:
        Rows suitable for ``opik.evaluate(dataset=...)``.
    """
    queries = load_search_queries(path=path, split=split)
    return [{"query": q.model_dump()} for q in queries]
