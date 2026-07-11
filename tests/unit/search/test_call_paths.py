"""Unit tests for the async ``__call__`` paths of each private index handler.

The ``_to_search_result`` normalisers are covered per-index in the
``test_arxiv.py`` / ``test_pubmed.py`` / ``test_crossref.py`` /
``test_semantic_scholar.py`` files.  This file exercises the full async
``__call__`` path by stubbing the upstream SDKs so the search runs end
to end without the network.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import arxiv
import pytest
from Bio import Entrez
from habanero import Crossref
from semanticscholar.Paper import Paper

from research_agent.search.models import SearchResult
from research_agent.search.tools import (
    _ArXivSearch,
    _CrossRefSearch,
    _PubMedSearch,
    _SemanticScholarSearch,
)

_ABSTRACT = (
    "An abstract that introduces a new approach to the problem under study, "
    "presenting a detailed methodology, a thorough experimental evaluation, "
    "and an analysis of the results. The findings advance the state of the art."
)
_TITLE = "Call Path Paper Title Long Enough"


def _noop_init(_self: object, **_kwargs: object) -> None:
    """Stand-in for ``Crossref.__init__`` that records nothing."""


@pytest.mark.asyncio
async def test_arxiv_call_runs_through_run_async(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _make_arxiv_result()
    fake_client = MagicMock()
    fake_client.results.return_value = iter([result])
    monkeypatch.setattr(arxiv, "Client", lambda **_kwargs: fake_client)
    monkeypatch.setattr(arxiv, "Search", lambda **_kwargs: object())

    tool = _ArXivSearch()
    out = await tool("q", limit=5)

    assert len(out) == 1
    assert isinstance(out[0], SearchResult)
    assert out[0].search_index_reference[0].id == result.entry_id


@pytest.mark.asyncio
async def test_pubmed_call_runs_through_run_async(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    article = _make_pubmed_article()
    responses = iter([{"IdList": ["12345"]}, {"PubmedArticle": [article]}])
    monkeypatch.setattr(
        Entrez, "esearch", lambda **_kwargs: MagicMock(close=MagicMock())
    )
    monkeypatch.setattr(
        Entrez, "efetch", lambda **_kwargs: MagicMock(close=MagicMock())
    )
    monkeypatch.setattr(Entrez, "read", lambda _handle: next(responses))

    tool = _PubMedSearch()
    out = await tool("cancer", limit=5)

    assert len(out) == 1
    assert isinstance(out[0], SearchResult)
    assert out[0].search_index_reference[0].id == "12345"


@pytest.mark.asyncio
async def test_pubmed_call_handles_empty_id_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        Entrez, "esearch", lambda **_kwargs: MagicMock(close=MagicMock())
    )
    monkeypatch.setattr(
        Entrez, "efetch", lambda **_kwargs: MagicMock(close=MagicMock())
    )
    monkeypatch.setattr(Entrez, "read", lambda _handle: {"IdList": []})

    tool = _PubMedSearch()
    out = await tool("nothing", limit=5)

    assert out == []


@pytest.mark.asyncio
async def test_crossref_call_runs_through_run_async(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_response = {"message": {"items": [_make_crossref_item()]}}
    monkeypatch.setattr(Crossref, "__init__", _noop_init)
    monkeypatch.setattr(Crossref, "works", lambda _self, **_kwargs: fake_response)

    tool = _CrossRefSearch()
    out = await tool("q", limit=5)

    assert len(out) == 1
    assert isinstance(out[0], SearchResult)
    assert out[0].search_index_reference[0].id == "10.1234/test"


@pytest.mark.asyncio
async def test_crossref_call_raises_on_unexpected_response_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Crossref, "__init__", _noop_init)
    monkeypatch.setattr(
        Crossref, "works", lambda _self, **_kwargs: ["not", "a", "dict"]
    )

    tool = _CrossRefSearch()
    with pytest.raises(TypeError, match="CrossRef returned unexpected type"):
        await tool("q", limit=5)


@pytest.mark.asyncio
async def test_semantic_scholar_call_with_async_client() -> None:
    tool = _SemanticScholarSearch()
    tool._client.search_paper = AsyncMock(  # type: ignore[method-assign]
        return_value=_StubSearchResults(
            [
                Paper(
                    {
                        "paperId": "p1",
                        "title": _TITLE,
                        "abstract": _ABSTRACT,
                        "authors": [{"name": "Alice"}],
                        "url": "http://x",
                    }
                )
            ]
        )
    )

    out = await tool("q", limit=5)

    assert len(out) == 1
    assert out[0].search_index_reference[0].id == "p1"


@pytest.mark.asyncio
async def test_semantic_scholar_call_drops_records_missing_title_or_abstract() -> None:
    tool = _SemanticScholarSearch()
    tool._client.search_paper = AsyncMock(  # type: ignore[method-assign]
        return_value=_StubSearchResults(
            [
                Paper(
                    {
                        "paperId": "missing",
                        "title": None,
                        "abstract": None,
                        "authors": [{"name": "Alice"}],
                        "url": "http://x",
                    }
                ),
                Paper(
                    {
                        "paperId": "kept",
                        "title": _TITLE,
                        "abstract": _ABSTRACT,
                        "authors": [{"name": "Alice"}],
                        "url": "http://x",
                    }
                ),
            ]
        )
    )

    out = await tool("q", limit=5)

    assert len(out) == 1
    assert out[0].search_index_reference[0].id == "kept"


@pytest.mark.asyncio
async def test_arxiv_call_drops_records_missing_title_or_abstract(
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
    fake_client = MagicMock()
    fake_client.results.return_value = iter([bad, good])
    monkeypatch.setattr(arxiv, "Client", lambda **_kwargs: fake_client)
    monkeypatch.setattr(arxiv, "Search", lambda **_kwargs: object())

    tool = _ArXivSearch()
    out = await tool("q", limit=5)

    assert len(out) == 1
    assert out[0].search_index_reference[0].id == good.entry_id


@pytest.mark.asyncio
async def test_pubmed_call_drops_records_missing_title_or_abstract(
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
    responses = iter(
        [
            {"IdList": ["00000", "12345"]},
            {"PubmedArticle": [bad_article, good]},
        ]
    )
    monkeypatch.setattr(
        Entrez, "esearch", lambda **_kwargs: MagicMock(close=MagicMock())
    )
    monkeypatch.setattr(
        Entrez, "efetch", lambda **_kwargs: MagicMock(close=MagicMock())
    )
    monkeypatch.setattr(Entrez, "read", lambda _handle: next(responses))

    tool = _PubMedSearch()
    out = await tool("cancer", limit=5)

    assert len(out) == 1
    assert out[0].search_index_reference[0].id == "12345"


@pytest.mark.asyncio
async def test_crossref_call_drops_records_missing_title_or_abstract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bad_item = _make_crossref_item()
    bad_item["title"] = []
    bad_item["abstract"] = ""
    good = _make_crossref_item()
    fake_response = {"message": {"items": [bad_item, good]}}
    monkeypatch.setattr(Crossref, "__init__", _noop_init)
    monkeypatch.setattr(Crossref, "works", lambda _self, **_kwargs: fake_response)

    tool = _CrossRefSearch()
    out = await tool("q", limit=5)

    assert len(out) == 1
    assert out[0].search_index_reference[0].id == good["DOI"]


def _make_arxiv_result() -> arxiv.Result:
    return arxiv.Result(
        entry_id="http://arxiv.org/abs/2306.04338v1",
        title=_TITLE,
        summary=_ABSTRACT,
        authors=[arxiv.Result.Author(name="Alice")],
        doi="10.1234/arxiv",
        categories=["cs.AI"],
        primary_category="cs.AI",
        journal_ref="",
        comment="",
        links=[],
        published=datetime(2024, 1, 1, tzinfo=UTC),
    )


def _make_pubmed_article() -> dict[str, Any]:
    return {
        "MedlineCitation": {
            "PMID": "12345",
            "Article": {
                "ArticleTitle": _TITLE,
                "Abstract": {"AbstractText": [_ABSTRACT]},
                "AuthorList": [{"LastName": "Doe", "ForeName": "Jane"}],
                "Journal": {
                    "Title": "Journal",
                    "JournalIssue": {"PubDate": {"Year": "2020"}},
                },
                "ELocationID": [_MockEId(doi="10.1234/pubmed")],
                "PublicationTypeList": [],
            },
        },
    }


class _MockEId:
    def __init__(self, *, doi: str) -> None:
        self._doi = doi
        self.attributes = {"EIdType": "doi"}

    def __str__(self) -> str:
        return self._doi


def _make_crossref_item() -> dict[str, Any]:
    return {
        "title": [_TITLE],
        "abstract": f"<jats:p>{_ABSTRACT}</jats:p>",
        "author": [{"given": "Alice", "family": "Smith"}],
        "DOI": "10.1234/test",
        "container-title": ["Journal"],
        "published-print": {"date-parts": [[2020]]},
        "is-referenced-by-count": 5,
        "type": "journal-article",
        "publisher": "Pub",
        "URL": "https://doi.org/10.1234/test",
        "resource": {"primary": {"URL": "https://example.com/pdf"}},
        "ISSN": [],
    }


class _StubSearchResults:
    def __init__(self, items: list[Paper]) -> None:
        self.items = items
