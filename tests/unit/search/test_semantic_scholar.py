"""Unit tests for the Semantic Scholar search tool.

Exercises the pure normalization path with a fabricated SDK `Paper`,
no network required.
"""

from __future__ import annotations

from semanticscholar.Paper import Paper

from research_agent.search.tools import (
    SemanticScholarSearch,
)


def test_to_search_result_maps_all_fields() -> None:
    paper = Paper(
        {
            "paperId": "abc123",
            "title": "Test Paper",
            "abstract": "Test abstract.",
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

    tool = SemanticScholarSearch()
    result = tool._to_search_result(paper)

    assert result.title == "Test Paper"
    assert result.abstract == "Test abstract."
    assert result.authors == ["Alice", "Bob"]
    assert result.reference.source.id == "abc123"
    assert result.reference.doi == "10.1234/test"
    assert result.publication_year == 2020
    assert result.citation_count == 10
    assert result.is_open_access is True
    assert result.pdf_url == "http://example.com/paper.pdf"
    assert result.tldr == "TLDR here"
    assert result.topics == ["Computer Science"]
