"""Unit tests for the Semantic Scholar search tool.

Exercises the pure normalization path with a fabricated SDK `Paper`,
no network required.
"""

from __future__ import annotations

from semanticscholar.Paper import Paper

from research_agent.search.models import SearchResult
from research_agent.search.tools import SemanticScholarSearch


def test_to_search_result_constructs_search_result() -> None:
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
    result = SemanticScholarSearch()._to_search_result(paper)
    assert isinstance(result, SearchResult)
