"""Unit tests for the unified ``LiteratureSearch`` dispatcher.

All tests go through the public ``LiteratureSearch`` class. Per-index
upstream SDKs (``arxiv``, ``Bio.Entrez``, ``habanero.Crossref``) are
stubbed at the boundary with ``monkeypatch``; the private per-index
handler classes (``_ArXivSearch``, ``_PubMedSearch``, ``_CrossRefSearch``)
are not imported here.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

import arxiv
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


def test_init_wires_three_handlers() -> None:
    tool = LiteratureSearch()

    arxiv_h = tool._handlers[SearchIndexType.ARXIV]
    pubmed = tool._handlers[SearchIndexType.PUBMED]
    crossref = tool._handlers[SearchIndexType.CROSSREF]

    assert arxiv_h is not None
    assert pubmed is not None
    assert crossref is not None


# --- arXiv ---


def _make_arxiv_result(  # noqa: PLR0913  # test helper mirroring arxiv.Result constructor
    *,
    title: str = _TITLE,
    summary: str = _ABSTRACT,
    authors: list[arxiv.Result.Author] | None = None,
    doi: str | None = None,
    categories: list[str] | None = None,
    pdf_link: str | None = None,
    entry_id: str = "http://arxiv.org/abs/2306.04338v1",
    journal_ref: str | None = None,
    comment: str | None = None,
    published: datetime | None = None,
) -> arxiv.Result:
    links: list[arxiv.Result.Link] = [
        arxiv.Result.Link(
            href=entry_id,
            title=None,
            rel="alternate",
            content_type=None,
        )
    ]
    if pdf_link:
        links.append(
            arxiv.Result.Link(
                href=pdf_link,
                title="pdf",
                rel="related",
                content_type="application/pdf",
            )
        )
    return arxiv.Result(
        entry_id=entry_id,
        title=title,
        summary=summary,
        authors=authors if authors is not None else [arxiv.Result.Author(name="Alice")],
        doi=doi or "",
        categories=categories or [],
        primary_category=(categories[0] if categories else ""),
        journal_ref=journal_ref or "",
        comment=comment or "",
        links=links,
        published=published or datetime(1, 1, 1, tzinfo=UTC),
    )


def _stub_arxiv(monkeypatch: pytest.MonkeyPatch, results: list[arxiv.Result]) -> None:
    fake_client = MagicMock()
    fake_client.results.return_value = iter(results)
    monkeypatch.setattr(arxiv, "Client", lambda **_kwargs: fake_client)
    monkeypatch.setattr(arxiv, "Search", lambda **_kwargs: object())


@pytest.mark.asyncio
async def test_arxiv_normalises_missing_doi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_arxiv(
        monkeypatch,
        [_make_arxiv_result(doi=None, authors=[arxiv.Result.Author(name="Alice")])],
    )

    tool = LiteratureSearch()
    out = await tool(SearchIndexType.ARXIV, "q", limit=5)

    assert len(out) == 1
    assert out[0].paper.source.doi is None


@pytest.mark.asyncio
async def test_arxiv_extracts_pdf_from_links_when_no_pdf_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _make_arxiv_result(authors=[arxiv.Result.Author(name="Alice")])
    result.links = [
        arxiv.Result.Link(
            href="http://arxiv.org/abs/2306.04338v1",
            title=None,
            rel="alternate",
            content_type=None,
        ),
        arxiv.Result.Link(
            href="http://arxiv.org/pdf/2306.04338v1",
            title="pdf",
            rel="related",
            content_type="application/pdf",
        ),
    ]
    _stub_arxiv(monkeypatch, [result])

    tool = LiteratureSearch()
    out = await tool(SearchIndexType.ARXIV, "q", limit=5)

    assert len(out) == 1
    assert str(out[0].paper.source.pdf_url) == "http://arxiv.org/pdf/2306.04338v1"


@pytest.mark.asyncio
async def test_arxiv_sentinel_year_is_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_arxiv(
        monkeypatch,
        [
            _make_arxiv_result(
                authors=[arxiv.Result.Author(name="Alice")],
                published=datetime(1, 1, 1, tzinfo=UTC),
            ),
        ],
    )

    tool = LiteratureSearch()
    out = await tool(SearchIndexType.ARXIV, "q", limit=5)

    assert len(out) == 1
    assert out[0].paper.publication_year is None


@pytest.mark.asyncio
async def test_arxiv_filters_none_author_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_arxiv(
        monkeypatch,
        [
            _make_arxiv_result(
                authors=[
                    arxiv.Result.Author(name="Alice"),
                    arxiv.Result.Author(name=""),
                    arxiv.Result.Author(name="Bob"),
                ],
            ),
        ],
    )

    tool = LiteratureSearch()
    out = await tool(SearchIndexType.ARXIV, "q", limit=5)

    assert len(out) == 1
    assert list(out[0].paper.authors) == ["Alice", "Bob"]


@pytest.mark.asyncio
async def test_arxiv_coerces_empty_doi_to_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_arxiv(
        monkeypatch,
        [
            _make_arxiv_result(
                doi="",
                authors=[arxiv.Result.Author(name="Alice")],
            ),
        ],
    )

    tool = LiteratureSearch()
    out = await tool(SearchIndexType.ARXIV, "q", limit=5)

    assert len(out) == 1
    assert out[0].paper.source.doi is None


@pytest.mark.asyncio
async def test_arxiv_drops_records_missing_title_or_abstract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bad = arxiv.Result(
        entry_id="http://arxiv.org/abs/bad",
        title="",
        summary="",
        authors=[arxiv.Result.Author(name="Alice")],
        doi="",
        categories=[],
        primary_category="",
        journal_ref="",
        comment="",
        links=[],
        published=datetime(2024, 1, 1, tzinfo=UTC),
    )
    good = _make_arxiv_result()
    _stub_arxiv(monkeypatch, [bad, good])

    tool = LiteratureSearch()
    out = await tool(SearchIndexType.ARXIV, "q", limit=5)

    assert len(out) == 1
    assert out[0].search_index_reference[0].id == good.entry_id


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
    assert out[0].paper.abstract == (
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
    assert list(out[0].paper.authors) == ["The Cancer Genome Atlas Network"]


@pytest.mark.asyncio
async def test_pubmed_handles_missing_year(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_pubmed(monkeypatch, [_make_pubmed_article(year="")])

    tool = LiteratureSearch()
    out = await tool(SearchIndexType.PUBMED, "q", limit=5)

    assert len(out) == 1
    assert out[0].paper.publication_year is None


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
    assert out[0].paper.abstract.startswith("A single string abstract that introduces")


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
    assert out[0].paper.source.doi is None


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
    assert out[0].search_index_reference[0].id == "12345"


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
    assert out[0].paper.abstract == (
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
    assert out[0].paper.publication_year == 2022


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
    assert out[0].paper.publication_year == 2019


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
    assert str(out[0].paper.source.url) == "https://doi.org/10.1234/test"


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
    assert out[0].paper.source.doi is None


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
    assert out[0].search_index_reference[0].id == good["DOI"]
