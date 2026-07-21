"""Unit tests for the unified ``LiteratureSearch`` dispatcher.

All tests go through the public ``LiteratureSearch`` class. Per-index
upstream SDKs (``Bio.Entrez``, ``habanero.Crossref``) are stubbed at
the boundary with ``monkeypatch``; the private per-index handler
classes (``_PubMedSearch``, ``_CrossRefSearch``) are not imported here.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from Bio import Entrez
from habanero import Crossref

from research_agent.search.models import SearchIndexType
from research_agent.search.tools import LiteratureSearch

_ABSTRACT = (
    "An abstract that introduces a new approach to the problem under study, "
    "presenting a detailed methodology, a thorough experimental evaluation, "
    "and an analysis of the results. The findings advance the state of the art."
)
_TITLE = "Consolidated Paper Title Long Enough"


def _noop_init(_self: object, **_kwargs: object) -> None:
    """Stand-in for ``Crossref.__init__`` that records nothing."""


# --- Init ---


def test_init_wires_two_handlers() -> None:
    tool = LiteratureSearch()

    pubmed = tool._handlers[SearchIndexType.PUBMED]
    crossref = tool._handlers[SearchIndexType.CROSSREF]

    assert pubmed is not None
    assert crossref is not None
    assert set(tool._handlers) == {
        SearchIndexType.PUBMED,
        SearchIndexType.CROSSREF,
    }


# --- PubMed ---


def _make_pubmed_article(  # noqa: PLR0913  # test helper mirroring PubMed efetch record shape
    *,
    pmid: str = "12345",
    title: str = _TITLE,
    abstract: str = _ABSTRACT,
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
                "AuthorList": authors or [{"LastName": "Doe", "ForeName": "Jane"}],
                "Journal": {
                    "Title": journal,
                    "JournalIssue": {"PubDate": {"Year": year}},
                },
                "ELocationID": [],
                "PublicationTypeList": publication_types or [],
            },
        },
    }


class _MockEId:
    def __init__(self, *, doi: str) -> None:
        self._doi = doi
        self.attributes = {"EIdType": "doi"}

    def __str__(self) -> str:
        return self._doi


def _stub_pubmed(
    monkeypatch: pytest.MonkeyPatch, articles: list[dict[str, Any]]
) -> None:
    """Stub ``Entrez`` to return the given articles when PubMed is searched."""
    responses = iter(
        [
            {"IdList": [a["MedlineCitation"]["PMID"] for a in articles]},
            {"PubmedArticle": articles},
        ]
    )
    monkeypatch.setattr(
        Entrez, "esearch", lambda **_kwargs: MagicMock(close=MagicMock())
    )
    monkeypatch.setattr(
        Entrez, "efetch", lambda **_kwargs: MagicMock(close=MagicMock())
    )
    monkeypatch.setattr(Entrez, "read", lambda _handle: next(responses))


def _stub_pubmed_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub ``Entrez`` to return an empty IdList."""
    monkeypatch.setattr(
        Entrez, "esearch", lambda **_kwargs: MagicMock(close=MagicMock())
    )
    monkeypatch.setattr(
        Entrez, "efetch", lambda **_kwargs: MagicMock(close=MagicMock())
    )
    monkeypatch.setattr(Entrez, "read", lambda _handle: {"IdList": []})


@pytest.mark.asyncio
async def test_pubmed_handles_abstract_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    article = _make_pubmed_article(abstract="")
    article["MedlineCitation"]["Article"]["Abstract"]["AbstractText"] = [
        "Part one introduces a new approach to the problem under study and "
        "presents a detailed methodology that builds on prior work in the area.",
        "Part two presents a thorough experimental evaluation and an analysis "
        "of the results, with a discussion of limitations and future work.",
    ]
    _stub_pubmed(monkeypatch, [article])

    tool = LiteratureSearch()
    out = await tool(SearchIndexType.PUBMED, "q", limit=5)

    assert len(out) == 1
    assert out[0].abstract == (
        "Part one introduces a new approach to the problem under study and "
        "presents a detailed methodology that builds on prior work in the area. "
        "Part two presents a thorough experimental evaluation and an analysis "
        "of the results, with a discussion of limitations and future work."
    )


@pytest.mark.asyncio
async def test_pubmed_handles_collective_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_pubmed(
        monkeypatch,
        [
            _make_pubmed_article(
                authors=[{"CollectiveName": "The Cancer Genome Atlas Network"}],
            ),
        ],
    )

    tool = LiteratureSearch()
    out = await tool(SearchIndexType.PUBMED, "q", limit=5)

    assert len(out) == 1
    assert list(out[0].authors) == ["The Cancer Genome Atlas Network"]


@pytest.mark.asyncio
async def test_pubmed_handles_missing_year(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_pubmed(monkeypatch, [_make_pubmed_article(year="")])

    tool = LiteratureSearch()
    out = await tool(SearchIndexType.PUBMED, "q", limit=5)

    assert len(out) == 1
    assert out[0].publication_year is None


@pytest.mark.asyncio
async def test_pubmed_handles_non_list_abstract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    article = _make_pubmed_article(abstract="")
    article["MedlineCitation"]["Article"]["Abstract"]["AbstractText"] = (
        "A single string abstract that introduces a new approach to the "
        "problem under study, presenting a detailed methodology, a thorough "
        "experimental evaluation, and an analysis of the results that advance "
        "the state of the art and suggest several directions for future work."
    )
    _stub_pubmed(monkeypatch, [article])

    tool = LiteratureSearch()
    out = await tool(SearchIndexType.PUBMED, "q", limit=5)

    assert len(out) == 1
    assert out[0].abstract.startswith("A single string abstract that introduces")


@pytest.mark.asyncio
async def test_pubmed_coerces_empty_doi_to_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    article = _make_pubmed_article()
    article["MedlineCitation"]["Article"]["ELocationID"] = [_MockEId(doi="")]
    _stub_pubmed(monkeypatch, [article])

    tool = LiteratureSearch()
    out = await tool(SearchIndexType.PUBMED, "q", limit=5)

    assert len(out) == 1
    assert out[0].doi is None


@pytest.mark.asyncio
async def test_pubmed_handles_empty_id_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_pubmed_empty(monkeypatch)

    tool = LiteratureSearch()
    out = await tool(SearchIndexType.PUBMED, "nothing", limit=5)

    assert out == []


@pytest.mark.asyncio
async def test_pubmed_drops_records_missing_title_or_abstract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    good = _make_pubmed_article()
    bad_article = {
        "MedlineCitation": {
            "PMID": "00000",
            "Article": {
                "ArticleTitle": "",
                "Abstract": {"AbstractText": [""]},
                "AuthorList": [{"LastName": "Doe", "ForeName": "Jane"}],
                "Journal": {
                    "Title": "Journal",
                    "JournalIssue": {"PubDate": {"Year": "2020"}},
                },
                "ELocationID": [],
                "PublicationTypeList": [],
            },
        },
    }
    _stub_pubmed(monkeypatch, [bad_article, good])

    tool = LiteratureSearch()
    out = await tool(SearchIndexType.PUBMED, "cancer", limit=5)

    assert len(out) == 1
    assert "12345" in str(out[0].url)


# --- CrossRef ---


def _make_crossref_item(  # noqa: PLR0913  # test helper mirroring CrossRef items dict shape
    *,
    title: str = _TITLE,
    abstract: str = _ABSTRACT,
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
        "author": authors or [{"given": "Alice", "family": "Smith"}],
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


def _stub_crossref(
    monkeypatch: pytest.MonkeyPatch, items: list[dict[str, Any]]
) -> None:
    fake_response = {"message": {"items": items}}
    monkeypatch.setattr(Crossref, "__init__", _noop_init)
    monkeypatch.setattr(Crossref, "works", lambda _self, **_kwargs: fake_response)


@pytest.mark.asyncio
async def test_crossref_strips_jats_from_abstract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item = _make_crossref_item(abstract="")
    item["abstract"] = (
        "<jats:p>Background: <jats:italic>Important</jats:italic> findings "
        "are presented in this paper, introducing a new approach to the "
        "problem under study, presenting a detailed methodology with a "
        "thorough experimental evaluation, and an analysis of the results "
        "that advance the state of the art.</jats:p>"
    )
    _stub_crossref(monkeypatch, [item])

    tool = LiteratureSearch()
    out = await tool(SearchIndexType.CROSSREF, "q", limit=5)

    assert len(out) == 1
    assert out[0].abstract == (
        "Background: Important findings are presented in this paper, "
        "introducing a new approach to the problem under study, presenting "
        "a detailed methodology with a thorough experimental evaluation, "
        "and an analysis of the results that advance the state of the art."
    )


@pytest.mark.asyncio
async def test_crossref_falls_back_on_published_online(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item = _make_crossref_item(year=2020)
    del item["published-print"]
    item["published-online"] = {"date-parts": [[2022]]}
    _stub_crossref(monkeypatch, [item])

    tool = LiteratureSearch()
    out = await tool(SearchIndexType.CROSSREF, "q", limit=5)

    assert len(out) == 1
    assert out[0].publication_year == 2022


@pytest.mark.asyncio
async def test_crossref_falls_back_on_issued(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item = _make_crossref_item(year=2020)
    del item["published-print"]
    item["issued"] = {"date-parts": [[2019]]}
    _stub_crossref(monkeypatch, [item])

    tool = LiteratureSearch()
    out = await tool(SearchIndexType.CROSSREF, "q", limit=5)

    assert len(out) == 1
    assert out[0].publication_year == 2019


@pytest.mark.asyncio
async def test_crossref_falls_back_url_to_doi_org(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item = _make_crossref_item()
    item["URL"] = ""
    _stub_crossref(monkeypatch, [item])

    tool = LiteratureSearch()
    out = await tool(SearchIndexType.CROSSREF, "q", limit=5)

    assert len(out) == 1
    assert str(out[0].url) == "https://doi.org/10.1234/test"


@pytest.mark.asyncio
async def test_crossref_raises_when_no_url_or_doi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item = _make_crossref_item()
    item["URL"] = ""
    item["DOI"] = ""
    _stub_crossref(monkeypatch, [item])

    tool = LiteratureSearch()
    with pytest.raises(ValueError, match="CrossRef"):
        await tool(SearchIndexType.CROSSREF, "q", limit=5)


@pytest.mark.asyncio
async def test_crossref_coerces_empty_doi_to_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_crossref(monkeypatch, [_make_crossref_item(doi="")])

    tool = LiteratureSearch()
    out = await tool(SearchIndexType.CROSSREF, "q", limit=5)

    assert len(out) == 1
    assert out[0].doi is None


@pytest.mark.asyncio
async def test_crossref_raises_on_unexpected_response_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Crossref, "__init__", _noop_init)
    monkeypatch.setattr(
        Crossref, "works", lambda _self, **_kwargs: ["not", "a", "dict"]
    )

    tool = LiteratureSearch()
    with pytest.raises(TypeError, match="CrossRef returned unexpected type"):
        await tool(SearchIndexType.CROSSREF, "q", limit=5)


@pytest.mark.asyncio
async def test_crossref_drops_records_missing_title_or_abstract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bad_item = _make_crossref_item()
    bad_item["title"] = []
    bad_item["abstract"] = ""
    good = _make_crossref_item()
    _stub_crossref(monkeypatch, [bad_item, good])

    tool = LiteratureSearch()
    out = await tool(SearchIndexType.CROSSREF, "q", limit=5)

    assert len(out) == 1
    assert out[0].doi == good["DOI"]
