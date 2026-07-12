"""Unit tests for the PubMed search tool.

Exercises the pure normalisation path with fabricated dictionaries that
match the Bio.Entrez parsed XML structure, no network required.
"""

from __future__ import annotations

from typing import Any, ClassVar

from research_agent.search.models import SearchIndexType, SearchResult
from research_agent.search.tools import _PubMedSearch

_DEFAULT_ABSTRACT = (
    "This paper introduces a new approach to the problem under study, "
    "presenting a detailed methodology, a thorough experimental evaluation, "
    "and an analysis of the results. The findings advance the state of the "
    "art and open several directions for future work in the area."
)
_DEFAULT_AUTHORS: list[dict[str, str]] = [{"LastName": "Doe", "ForeName": "Jane"}]


def _make_article(  # noqa: PLR0913  # test helper mirroring PubMed efetch record shape
    *,
    pmid: str = "12345",
    title: str = "Test Paper",
    abstract: str = _DEFAULT_ABSTRACT,
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
                "AuthorList": authors or _DEFAULT_AUTHORS,
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
    article = _make_article(abstract="")
    article["MedlineCitation"]["Article"]["Abstract"]["AbstractText"] = [
        "Part one introduces a new approach to the problem under study and "
        "presents a detailed methodology that builds on prior work in the area.",
        "Part two presents a thorough experimental evaluation and an analysis "
        "of the results, with a discussion of limitations and future work.",
    ]
    sr = _PubMedSearch._to_search_result(article)
    assert isinstance(sr, SearchResult)
    assert sr.paper.abstract == (
        "Part one introduces a new approach to the problem under study and "
        "presents a detailed methodology that builds on prior work in the area. "
        "Part two presents a thorough experimental evaluation and an analysis "
        "of the results, with a discussion of limitations and future work."
    )


def test_to_search_result_handles_collective_name() -> None:
    sr = _PubMedSearch._to_search_result(
        _make_article(authors=[{"CollectiveName": "The Cancer Genome Atlas Network"}]),
    )
    assert isinstance(sr, SearchResult)
    assert list(sr.paper.authors) == ["The Cancer Genome Atlas Network"]


def test_to_search_result_handles_missing_year() -> None:
    sr = _PubMedSearch._to_search_result(_make_article(year=""))
    assert isinstance(sr, SearchResult)
    assert sr.paper.publication_year is None


def test_to_search_result_handles_non_list_abstract() -> None:
    article = _make_article(abstract="")
    article["MedlineCitation"]["Article"]["Abstract"]["AbstractText"] = (
        "A single string abstract that introduces a new approach to the "
        "problem under study, presenting a detailed methodology, a thorough "
        "experimental evaluation, and an analysis of the results that advance "
        "the state of the art and suggest several directions for future work."
    )
    sr = _PubMedSearch._to_search_result(article)
    assert isinstance(sr, SearchResult)
    assert sr.paper.abstract.startswith("A single string abstract that introduces")


def test_has_title_and_abstract_handles_non_list_abstract() -> None:
    article = _make_article(abstract="")
    article["MedlineCitation"]["Article"]["Abstract"]["AbstractText"] = (
        "A non-empty string abstract that is treated as a single text part."
    )
    assert _PubMedSearch._has_title_and_abstract(article) is True


def test_has_title_and_abstract_rejects_missing_abstract() -> None:
    article = _make_article(abstract="")
    article["MedlineCitation"]["Article"]["Abstract"]["AbstractText"] = ""
    assert _PubMedSearch._has_title_and_abstract(article) is False


def test_to_search_result_index_is_pubmed() -> None:
    sr = _PubMedSearch._to_search_result(_make_article(pmid="99999"))
    assert sr.search_index_reference[0].index == SearchIndexType.PUBMED
    assert sr.search_index_reference[0].id == "99999"
    assert str(sr.paper.source.url) == "https://pubmed.ncbi.nlm.nih.gov/99999/"


def test_to_search_result_coerces_empty_doi_to_none() -> None:
    article = _make_article()
    article["MedlineCitation"]["Article"]["ELocationID"] = [
        _EmptyDoiEId(),
    ]
    sr = _PubMedSearch._to_search_result(article)
    assert sr.paper.source.doi is None


class _EmptyDoiEId:
    attributes: ClassVar[dict[str, str]] = {"EIdType": "doi"}

    def __str__(self) -> str:
        return ""
