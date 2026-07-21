"""Unit tests for the unified ``LiteratureSearch`` dispatcher.

All tests go through the public ``LiteratureSearch`` class. Per-index
upstream SDKs (``Bio.Entrez``, ``habanero.Crossref``) are stubbed at
the boundary with ``monkeypatch``; the private per-index handler
classes (``_PubMedSearch``, ``_CrossRefSearch``) are not imported here.
"""

from __future__ import annotations

from typing import Any, Self
from unittest.mock import MagicMock

import httpx
import pytest
from Bio import Entrez
from habanero import Crossref

from research_agent.search import tools as tools_mod
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

    pubmed = tool._handlers[SearchIndexType.PUBMED]
    crossref = tool._handlers[SearchIndexType.CROSSREF]
    openalex = tool._handlers[SearchIndexType.OPENALEX]

    assert pubmed is not None
    assert crossref is not None
    assert openalex is not None
    assert set(tool._handlers) == {
        SearchIndexType.PUBMED,
        SearchIndexType.CROSSREF,
        SearchIndexType.OPENALEX,
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


# --- OpenAlex ---


def _invert_abstract(text: str) -> dict[str, list[int]]:
    inverted: dict[str, list[int]] = {}
    for index, token in enumerate(text.split()):
        inverted.setdefault(token, []).append(index)
    return inverted


def _make_openalex_work(  # noqa: PLR0913  # test helper mirroring OpenAlex work shape
    *,
    title: str = _TITLE,
    abstract: str = _ABSTRACT,
    authors: list[str] | None = None,
    doi: str = "https://doi.org/10.1234/openalex-test",
    year: int = 2021,
    citation_count: int = 42,
    landing_page_url: str = "https://example.com/paper",
    pdf_url: str = "",
    openalex_id: str = "https://openalex.org/W123",
) -> dict[str, Any]:
    author_names = authors if authors is not None else ["Alice Smith"]
    return {
        "id": openalex_id,
        "display_name": title,
        "abstract_inverted_index": _invert_abstract(abstract) if abstract else None,
        "authorships": [{"author": {"display_name": name}} for name in author_names],
        "doi": doi,
        "publication_year": year,
        "cited_by_count": citation_count,
        "primary_location": {"landing_page_url": landing_page_url, "pdf_url": None},
        "best_oa_location": {"pdf_url": pdf_url or None, "landing_page_url": None},
        "locations": [],
        "open_access": {"is_oa": bool(pdf_url)},
    }


def _stub_openalex(
    monkeypatch: pytest.MonkeyPatch,
    works: list[dict[str, Any]],
    *,
    capture: dict[str, Any] | None = None,
) -> None:
    async def fake_fetch(
        self: object, query: str, *, per_page: int
    ) -> list[dict[str, Any]]:
        if capture is not None:
            capture["query"] = query
            capture["per_page"] = per_page
            capture["api_key"] = getattr(self, "_api_key", None)
        return works

    monkeypatch.setattr(tools_mod._OpenAlexSearch, "_fetch_works", fake_fetch)


@pytest.mark.asyncio
async def test_openalex_reconstructs_inverted_abstract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    work = _make_openalex_work()
    _stub_openalex(monkeypatch, [work])

    tool = LiteratureSearch()
    out = await tool(SearchIndexType.OPENALEX, "q", limit=5)

    assert len(out) == 1
    assert out[0].abstract == _ABSTRACT
    assert out[0].title == _TITLE
    assert out[0].authors == ("Alice Smith",)
    assert out[0].doi == "10.1234/openalex-test"
    assert out[0].publication_year == 2021
    assert out[0].citation_count == 42
    assert str(out[0].url) == "https://example.com/paper"


@pytest.mark.asyncio
async def test_openalex_url_falls_back_to_doi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    work = _make_openalex_work(landing_page_url="")
    work["primary_location"] = {"landing_page_url": None, "pdf_url": None}
    _stub_openalex(monkeypatch, [work])

    tool = LiteratureSearch()
    out = await tool(SearchIndexType.OPENALEX, "q", limit=5)

    assert len(out) == 1
    assert str(out[0].url) == "https://doi.org/10.1234/openalex-test"


@pytest.mark.asyncio
async def test_openalex_url_falls_back_to_openalex_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    work = _make_openalex_work(landing_page_url="", doi="")
    work["primary_location"] = {"landing_page_url": None, "pdf_url": None}
    work["doi"] = None
    _stub_openalex(monkeypatch, [work])

    tool = LiteratureSearch()
    out = await tool(SearchIndexType.OPENALEX, "q", limit=5)

    assert len(out) == 1
    assert str(out[0].url) == "https://openalex.org/W123"


@pytest.mark.asyncio
async def test_openalex_raises_when_no_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    work = _make_openalex_work(landing_page_url="", doi="", openalex_id="")
    work["primary_location"] = {"landing_page_url": None, "pdf_url": None}
    work["doi"] = None
    work["id"] = None
    _stub_openalex(monkeypatch, [work])

    tool = LiteratureSearch()
    with pytest.raises(ValueError, match="OpenAlex"):
        await tool(SearchIndexType.OPENALEX, "q", limit=5)


@pytest.mark.asyncio
async def test_openalex_open_access_only_with_pdf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with_pdf = _make_openalex_work(pdf_url="https://example.com/paper.pdf")
    without_pdf = _make_openalex_work(
        title="Another Long Enough Paper Title Here",
        pdf_url="",
    )
    without_pdf["open_access"] = {"is_oa": True}
    _stub_openalex(monkeypatch, [with_pdf, without_pdf])

    tool = LiteratureSearch()
    out = await tool(SearchIndexType.OPENALEX, "q", limit=5)

    assert len(out) == 2
    assert out[0].open_access is True
    assert out[0].pdf_url is not None
    assert "paper.pdf" in str(out[0].pdf_url)
    assert out[1].open_access is False
    assert out[1].pdf_url is None


@pytest.mark.asyncio
async def test_openalex_drops_incomplete_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    no_title = _make_openalex_work(title="")
    no_abstract = _make_openalex_work(abstract="")
    no_authors = _make_openalex_work(authors=[])
    good = _make_openalex_work()
    _stub_openalex(monkeypatch, [no_title, no_abstract, no_authors, good])

    tool = LiteratureSearch()
    out = await tool(SearchIndexType.OPENALEX, "q", limit=5)

    assert len(out) == 1
    assert out[0].title == _TITLE


@pytest.mark.asyncio
async def test_openalex_empty_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_openalex(monkeypatch, [])

    tool = LiteratureSearch()
    out = await tool(SearchIndexType.OPENALEX, "nothing", limit=5)

    assert out == []


@pytest.mark.asyncio
async def test_openalex_clamps_limit_and_forwards_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture: dict[str, Any] = {}
    _stub_openalex(monkeypatch, [_make_openalex_work()], capture=capture)

    tool = LiteratureSearch(openalex_api_key="test-key")
    await tool(SearchIndexType.OPENALEX, "crispr", limit=500)

    assert capture["query"] == "crispr"
    assert capture["per_page"] == 100
    assert capture["api_key"] == "test-key"


@pytest.mark.asyncio
async def test_openalex_omits_api_key_when_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture: dict[str, Any] = {}
    _stub_openalex(monkeypatch, [_make_openalex_work()], capture=capture)

    tool = LiteratureSearch()
    await tool(SearchIndexType.OPENALEX, "q", limit=5)

    assert capture["api_key"] is None


@pytest.mark.asyncio
async def test_openalex_raises_on_unexpected_payload_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def bad_fetch(
        self: object, query: str, *, per_page: int
    ) -> list[dict[str, Any]]:
        del self, query, per_page
        msg = "OpenAlex returned unexpected type list"
        raise TypeError(msg)

    monkeypatch.setattr(tools_mod._OpenAlexSearch, "_fetch_works", bad_fetch)

    tool = LiteratureSearch()
    with pytest.raises(TypeError, match="OpenAlex returned unexpected type"):
        await tool(SearchIndexType.OPENALEX, "q", limit=5)


@pytest.mark.asyncio
async def test_openalex_pdf_from_locations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    work = _make_openalex_work(pdf_url="")
    work["best_oa_location"] = {"pdf_url": None}
    work["locations"] = [{"pdf_url": "https://example.com/loc.pdf"}]
    _stub_openalex(monkeypatch, [work])

    tool = LiteratureSearch()
    out = await tool(SearchIndexType.OPENALEX, "q", limit=5)

    assert len(out) == 1
    assert out[0].open_access is True
    assert "loc.pdf" in str(out[0].pdf_url)


@pytest.mark.asyncio
async def test_openalex_fetch_builds_query_and_raises_on_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise real ``_fetch_works`` via a mocked ``AsyncClient.get``."""
    work = _make_openalex_work()
    seen: dict[str, str] = {}

    class _FakeResponse:
        def __init__(self, *, status_code: int, payload: object) -> None:
            self.status_code = status_code
            self._payload = payload

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                request = httpx.Request("GET", "https://api.openalex.org/works")
                response = httpx.Response(self.status_code, request=request)
                msg = "error"
                raise httpx.HTTPStatusError(
                    msg,
                    request=request,
                    response=response,
                )

        def json(self) -> object:
            return self._payload

    class _FakeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        async def __aenter__(self) -> Self:
            return self

        async def __aexit__(self, *args: object) -> None:
            del args

        async def get(self, url: str) -> _FakeResponse:
            seen["url"] = url
            if "fail" in url:
                return _FakeResponse(status_code=429, payload={"error": "rate"})
            return _FakeResponse(status_code=200, payload={"results": [work]})

    monkeypatch.setattr(tools_mod.httpx, "AsyncClient", _FakeClient)

    tool = LiteratureSearch(openalex_api_key="secret")
    out = await tool(SearchIndexType.OPENALEX, "crispr", limit=5)
    assert len(out) == 1
    assert "api_key=secret" in seen["url"]
    assert "search=crispr" in seen["url"]
    assert "has_abstract" in seen["url"]

    with pytest.raises(httpx.HTTPStatusError):
        await tool(SearchIndexType.OPENALEX, "fail", limit=5)
