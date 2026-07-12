"""Unit tests for the CrossRef search tool.

Exercises the pure normalisation path with fabricated response dicts,
no network required.
"""

from __future__ import annotations

from typing import Any

import pytest

from research_agent.search.models import SearchIndexType, SearchResult
from research_agent.search.tools import _CrossRefSearch

_DEFAULT_ABSTRACT = (
    "This paper introduces a new approach to the problem under study, "
    "presenting a detailed methodology, a thorough experimental evaluation, "
    "and an analysis of the results. The findings advance the state of the "
    "art and open several directions for future work in the area."
)
_DEFAULT_AUTHORS: list[dict[str, str]] = [{"given": "Alice", "family": "Smith"}]


def _make_item(  # noqa: PLR0913  # test helper mirroring CrossRef items dict shape
    *,
    title: str = "Test Paper",
    abstract: str = _DEFAULT_ABSTRACT,
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
        "author": authors or _DEFAULT_AUTHORS,
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
        "<jats:p>Background: <jats:italic>Important</jats:italic> findings "
        "are presented in this paper, introducing a new approach to the "
        "problem under study, presenting a detailed methodology with a "
        "thorough experimental evaluation, and an analysis of the results "
        "that advance the state of the art.</jats:p>"
    )
    sr = _CrossRefSearch._to_search_result(item)
    assert isinstance(sr, SearchResult)
    assert sr.paper.abstract == (
        "Background: Important findings are presented in this paper, "
        "introducing a new approach to the problem under study, presenting "
        "a detailed methodology with a thorough experimental evaluation, "
        "and an analysis of the results that advance the state of the art."
    )


def test_to_search_result_falls_back_on_published_online() -> None:
    item = _make_item(year=2020)
    del item["published-print"]
    item["published-online"] = {"date-parts": [[2022]]}
    sr = _CrossRefSearch._to_search_result(item)
    assert isinstance(sr, SearchResult)
    assert sr.paper.publication_year == 2022


def test_to_search_result_falls_back_on_issued() -> None:
    item = _make_item(year=2020)
    del item["published-print"]
    item["issued"] = {"date-parts": [[2019]]}
    sr = _CrossRefSearch._to_search_result(item)
    assert isinstance(sr, SearchResult)
    assert sr.paper.publication_year == 2019


def test_to_search_result_index_is_crossref() -> None:
    sr = _CrossRefSearch._to_search_result(_make_item(doi="10.9999/xyz"))
    assert sr.search_index_reference[0].index == SearchIndexType.CROSSREF
    assert sr.search_index_reference[0].id == "10.9999/xyz"
    assert sr.paper.source.doi == "10.9999/xyz"


def test_to_search_result_falls_back_url_to_doi_org() -> None:
    item = _make_item()
    item["URL"] = ""
    sr = _CrossRefSearch._to_search_result(item)
    assert str(sr.paper.source.url) == "https://doi.org/10.1234/test"


def test_to_search_result_raises_when_no_url_or_doi() -> None:
    item = _make_item()
    item["URL"] = ""
    item["DOI"] = ""
    with pytest.raises(ValueError, match="CrossRef"):
        _CrossRefSearch._to_search_result(item)


def test_to_search_result_coerces_empty_doi_to_none() -> None:
    item = _make_item(doi="")
    sr = _CrossRefSearch._to_search_result(item)
    assert sr.paper.source.doi is None
