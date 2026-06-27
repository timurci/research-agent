"""Unit tests for the CrossRef search tool.

Exercises the pure normalisation path with fabricated response dicts,
no network required.
"""

from __future__ import annotations

from typing import Any

from research_agent.search.models import SearchResult
from research_agent.search.tools import CrossRefSearch


def _make_item(  # noqa: PLR0913  # test helper mirroring CrossRef items dict shape
    *,
    title: str = "Test Paper",
    abstract: str = "Test abstract.",
    authors: list[dict[str, str]] | None = None,
    doi: str = "10.1234/test",
    container: str = "Test Journal",
    year: int = 2020,
    citation_count: int = 0,
    resource_url: str = "",
    url: str = "",
    item_type: str = "journal-article",
    publisher: str = "Test Publisher",
    issn: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "title": [title],
        "abstract": f"<p>{abstract}</p>",
        "author": authors or [],
        "DOI": doi,
        "container-title": [container],
        "published-print": {"date-parts": [[year]]},
        "is-referenced-by-count": citation_count,
        "type": item_type,
        "publisher": publisher,
        "URL": url or f"https://doi.org/{doi}",
        "resource": {"primary": {"URL": resource_url}},
        "ISSN": issn or [],
    }


def test_to_search_result_strips_jats_from_abstract() -> None:
    item = _make_item(abstract="")
    item["abstract"] = (
        "<jats:p>Background: <jats:italic>Important</jats:italic> findings.</jats:p>"
    )
    sr = CrossRefSearch._to_search_result(item)
    assert isinstance(sr, SearchResult)
    assert sr.abstract == "Background: Important findings."


def test_to_search_result_handles_missing_title() -> None:
    item = _make_item(title="")
    item["title"] = []
    sr = CrossRefSearch._to_search_result(item)
    assert isinstance(sr, SearchResult)
    assert sr.title is None


def test_to_search_result_handles_missing_abstract() -> None:
    item = _make_item(abstract="")
    item["abstract"] = ""
    sr = CrossRefSearch._to_search_result(item)
    assert isinstance(sr, SearchResult)
    assert sr.abstract is None


def test_to_search_result_falls_back_on_published_online() -> None:
    item = _make_item(year=2020)
    del item["published-print"]
    item["published-online"] = {"date-parts": [[2022]]}
    sr = CrossRefSearch._to_search_result(item)
    assert isinstance(sr, SearchResult)
    assert sr.publication_year == 2022


def test_to_search_result_falls_back_on_issued() -> None:
    item = _make_item(year=2020)
    del item["published-print"]
    item["issued"] = {"date-parts": [[2019]]}
    sr = CrossRefSearch._to_search_result(item)
    assert isinstance(sr, SearchResult)
    assert sr.publication_year == 2019


def test_to_search_result_stores_raw_metadata() -> None:
    item = _make_item(
        item_type="book-chapter",
        publisher="Oxford University Press",
        issn=["1234-5678"],
    )
    sr = CrossRefSearch._to_search_result(item)
    assert isinstance(sr, SearchResult)
    assert sr.raw_metadata is not None
    assert sr.raw_metadata["type"] == "book-chapter"
    assert sr.raw_metadata["publisher"] == "Oxford University Press"
    assert sr.raw_metadata["issn"] == ["1234-5678"]
