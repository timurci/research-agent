"""Unit tests for the PubMed search tool.

Exercises the pure normalisation path with fabricated dictionaries that
match the Bio.Entrez parsed XML structure, no network required.
"""

from __future__ import annotations

from typing import Any

from research_agent.search.models import SearchIndexType, SearchResult
from research_agent.search.tools import _PubMedSearch


def _make_article(  # noqa: PLR0913  # test helper mirroring PubMed efetch record shape
    *,
    pmid: str = "12345",
    title: str = "Test Paper",
    abstract: str = "Test abstract.",
    authors: list[dict[str, str]] | None = None,
    journal: str = "Test Journal",
    year: str = "2020",
    publication_types: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "MedlineCitation": {
            "PMID": pmid,
            "Article": {
                "ArticleTitle": title,
                "Abstract": {"AbstractText": [abstract]},
                "AuthorList": authors or [],
                "Journal": {
                    "Title": journal,
                    "JournalIssue": {"PubDate": {"Year": year}},
                },
                "ELocationID": [],
                "PublicationTypeList": publication_types or [],
            },
        },
    }


def test_to_search_result_handles_abstract_list() -> None:
    article = _make_article(abstract="Part one")
    article["MedlineCitation"]["Article"]["Abstract"]["AbstractText"] = [
        "Part one.",
        "Part two.",
    ]
    sr = _PubMedSearch._to_search_result(article)
    assert isinstance(sr, SearchResult)
    assert sr.paper.abstract == "Part one. Part two."


def test_to_search_result_handles_collective_name() -> None:
    sr = _PubMedSearch._to_search_result(
        _make_article(authors=[{"CollectiveName": "The Cancer Genome Atlas Network"}]),
    )
    assert isinstance(sr, SearchResult)
    assert sr.paper.authors == ["The Cancer Genome Atlas Network"]


def test_to_search_result_handles_missing_year() -> None:
    sr = _PubMedSearch._to_search_result(_make_article(year=""))
    assert isinstance(sr, SearchResult)
    assert sr.paper.publication_year is None


def test_to_search_result_handles_missing_title_and_abstract() -> None:
    sr = _PubMedSearch._to_search_result(_make_article(title="", abstract=""))
    assert isinstance(sr, SearchResult)
    assert sr.paper.title is None
    assert sr.paper.abstract is None


def test_to_search_result_handles_non_list_abstract() -> None:
    article = _make_article(abstract="")
    article["MedlineCitation"]["Article"]["Abstract"]["AbstractText"] = "Single string"
    sr = _PubMedSearch._to_search_result(article)
    assert isinstance(sr, SearchResult)
    assert sr.paper.abstract == "Single string"


def test_to_search_result_stores_publication_types() -> None:
    sr = _PubMedSearch._to_search_result(
        _make_article(publication_types=["Journal Article", "Review"]),
    )
    assert isinstance(sr, SearchResult)
    assert sr.paper.raw_metadata is not None
    assert sr.paper.raw_metadata["publication_types"] == ["Journal Article", "Review"]


def test_to_search_result_index_is_pubmed() -> None:
    sr = _PubMedSearch._to_search_result(_make_article(pmid="99999"))
    assert sr.search_reference.index == SearchIndexType.PUBMED
    assert sr.search_reference.id == "99999"
    assert str(sr.paper.source.url) == "https://pubmed.ncbi.nlm.nih.gov/99999/"
