"""Unit tests for the Semantic Scholar search tool.

Exercises the pure normalization path with a fabricated SDK `Paper`,
no network required.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from semanticscholar.Paper import Paper

from research_agent.search.models import SearchIndexType, SearchResult
from research_agent.search.tools import _SemanticScholarSearch

_ABSTRACT = (
    "Test abstract that introduces a new approach to the problem under study, "
    "presenting a detailed methodology, a thorough experimental evaluation, "
    "and an analysis of the results. The findings advance the state of the art."
)
_TITLE = "Test Paper Title Long Enough"


def test_to_search_result_constructs_search_result() -> None:
    paper = Paper(
        {
            "paperId": "abc123",
            "title": _TITLE,
            "abstract": _ABSTRACT,
            "authors": [{"name": "Alice"}, {"name": None}, {"name": "Bob"}],
            "year": 2020,
            "venue": "Test Venue",
            "citationCount": 10,
            "isOpenAccess": True,
            "openAccessPdf": {"url": "http://example.com/paper.pdf"},
            "tldr": {"text": "TLDR here"},
            "externalIds": {"DOI": "10.1234/test"},
            "url": "http://example.com",
            "fieldsOfStudy": ["Computer Science"],
        }
    )
    result = _SemanticScholarSearch()._to_search_result(paper)
    assert isinstance(result, SearchResult)
    assert result.search_index_reference[0].index == SearchIndexType.SEMANTIC_SCHOLAR
    assert result.search_index_reference[0].id == "abc123"
    assert result.paper.title == _TITLE
    assert result.paper.abstract == _ABSTRACT
    assert result.paper.authors == ["Alice", "Bob"]
    assert result.paper.citation_count == 10
    assert result.paper.source.open_access is True
    assert str(result.paper.source.pdf_url) == "http://example.com/paper.pdf"
    assert result.paper.source.doi == "10.1234/test"
    assert result.paper.raw_metadata is not None
    assert result.paper.raw_metadata["venue"] == "Test Venue"
    assert result.paper.raw_metadata["fields_of_study"] == ["Computer Science"]
    assert result.paper.raw_metadata["tldr"] == "TLDR here"


def test_to_search_result_falls_back_url_when_missing() -> None:
    paper = Paper(
        {
            "paperId": "abc123",
            "title": _TITLE,
            "abstract": _ABSTRACT,
            "authors": [{"name": "Alice"}],
            "url": None,
        }
    )
    result = _SemanticScholarSearch()._to_search_result(paper)
    assert (
        str(result.paper.source.url) == "https://www.semanticscholar.org/paper/abc123"
    )


def test_to_search_result_coerces_empty_doi_to_none() -> None:
    paper = Paper(
        {
            "paperId": "abc123",
            "title": _TITLE,
            "abstract": _ABSTRACT,
            "authors": [{"name": "Alice"}],
            "externalIds": {"DOI": ""},
            "url": "http://example.com",
        }
    )
    result = _SemanticScholarSearch()._to_search_result(paper)
    assert result.paper.source.doi is None


def test_to_search_result_coerces_empty_venue_to_none() -> None:
    paper = Paper(
        {
            "paperId": "abc123",
            "title": _TITLE,
            "abstract": _ABSTRACT,
            "authors": [{"name": "Alice"}],
            "venue": "",
            "url": "http://example.com",
        }
    )
    result = _SemanticScholarSearch()._to_search_result(paper)
    assert result.paper.raw_metadata is not None
    assert result.paper.raw_metadata["venue"] is None


def test_to_search_result_coerces_empty_fields_of_study_to_none() -> None:
    paper = Paper(
        {
            "paperId": "abc123",
            "title": _TITLE,
            "abstract": _ABSTRACT,
            "authors": [{"name": "Alice"}],
            "fieldsOfStudy": [],
            "url": "http://example.com",
        }
    )
    result = _SemanticScholarSearch()._to_search_result(paper)
    assert result.paper.raw_metadata is not None
    assert result.paper.raw_metadata["fields_of_study"] is None


def test_to_search_result_coerces_empty_tldr_text_to_none() -> None:
    paper = Paper(
        {
            "paperId": "abc123",
            "title": _TITLE,
            "abstract": _ABSTRACT,
            "authors": [{"name": "Alice"}],
            "tldr": {"text": "   "},
            "url": "http://example.com",
        }
    )
    result = _SemanticScholarSearch()._to_search_result(paper)
    assert result.paper.raw_metadata is not None
    assert result.paper.raw_metadata["tldr"] is None


@pytest.mark.asyncio
async def test_call_normalises_list_response() -> None:
    paper = Paper(
        {
            "paperId": "p1",
            "title": _TITLE,
            "abstract": _ABSTRACT,
            "authors": [{"name": "Alice"}],
            "url": "http://x",
        }
    )
    tool = _SemanticScholarSearch()
    tool._client.search_paper = AsyncMock(  # type: ignore[method-assign]
        return_value=_StubSearchResults([paper])
    )

    results = await tool("query", limit=5)

    assert len(results) == 1
    assert results[0].search_index_reference[0].id == "p1"


@pytest.mark.asyncio
async def test_call_normalises_single_paper_response() -> None:
    paper = Paper(
        {
            "paperId": "p2",
            "title": _TITLE,
            "abstract": _ABSTRACT,
            "authors": [{"name": "Alice"}],
            "url": "http://x",
        }
    )
    tool = _SemanticScholarSearch()
    tool._client.search_paper = AsyncMock(return_value=paper)  # type: ignore[method-assign]

    results = await tool("query", limit=5)

    assert len(results) == 1
    assert results[0].search_index_reference[0].id == "p2"


class _StubSearchResults:
    def __init__(self, items: list[Paper]) -> None:
        self.items = items
