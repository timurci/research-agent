"""Unit tests for the unified ``LiteratureSearch`` dispatcher.

Verifies ``LiteratureSearch.__call__`` routes to the correct private
index handler for each ``SearchIndexType`` and wires the per-index API
keys through to the matching handler.  Per-index normalisation is
covered by ``tests/unit/search/test_*.py``; the async ``__call__`` paths
of each handler are covered by ``tests/unit/search/test_call_paths.py``.
"""

from __future__ import annotations

from typing import Any

import pytest

from research_agent.search.models import SearchIndexType
from research_agent.search.tools import (
    LiteratureSearch,
    _ArXivSearch,
    _CrossRefSearch,
    _PubMedSearch,
    _SemanticScholarSearch,
)


@pytest.mark.asyncio
async def test_call_dispatches_to_semantic_scholar() -> None:
    captured: dict[str, Any] = {}

    class _Stub:
        async def __call__(self, query: str, *, limit: int) -> list[Any]:
            captured["query"] = query
            captured["limit"] = limit
            return ["s2-result"]

    tool = LiteratureSearch()
    tool._handlers[SearchIndexType.SEMANTIC_SCHOLAR] = _Stub()  # type: ignore[method-assign]
    result = await tool(SearchIndexType.SEMANTIC_SCHOLAR, "q", limit=7)

    assert result == ["s2-result"]
    assert captured == {"query": "q", "limit": 7}


def test_init_wires_all_four_handlers() -> None:
    tool = LiteratureSearch()

    s2 = tool._handlers[SearchIndexType.SEMANTIC_SCHOLAR]
    arxiv_h = tool._handlers[SearchIndexType.ARXIV]
    pubmed = tool._handlers[SearchIndexType.PUBMED]
    crossref = tool._handlers[SearchIndexType.CROSSREF]

    assert isinstance(s2, _SemanticScholarSearch)
    assert isinstance(arxiv_h, _ArXivSearch)
    assert isinstance(pubmed, _PubMedSearch)
    assert isinstance(crossref, _CrossRefSearch)


def test_init_forwards_semantic_scholar_api_key() -> None:
    tool = LiteratureSearch(s2_api_key="s2k")

    s2 = tool._handlers[SearchIndexType.SEMANTIC_SCHOLAR]
    assert isinstance(s2, _SemanticScholarSearch)
    assert s2._client.auth_header == {"x-api-key": "s2k"}


def test_init_accepts_pubmed_api_key() -> None:
    tool = LiteratureSearch(pubmed_api_key="pmk")

    pubmed = tool._handlers[SearchIndexType.PUBMED]
    assert isinstance(pubmed, _PubMedSearch)
