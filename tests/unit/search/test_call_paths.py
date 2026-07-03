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
    assert out[0].search_reference.id == result.entry_id


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
    assert out[0].search_reference.id == "12345"


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
    assert out[0].search_reference.id == "10.1234/test"


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
            [Paper({"paperId": "p1", "title": "T", "authors": [], "url": "http://x"})]
        )
    )

    out = await tool("q", limit=5)

    assert len(out) == 1
    assert out[0].search_reference.id == "p1"


def _make_arxiv_result() -> arxiv.Result:
    return arxiv.Result(
        entry_id="http://arxiv.org/abs/2306.04338v1",
        title="ArXiv Paper",
        summary="Abstract.",
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
                "ArticleTitle": "Pubmed Paper",
                "Abstract": {"AbstractText": ["Abstract."]},
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
        "title": ["Crossref Paper"],
        "abstract": "<jats:p>Abstract.</jats:p>",
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
