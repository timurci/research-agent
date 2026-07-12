"""Unit tests for the search-slice application workflows."""

from __future__ import annotations

import pytest
from pydantic import HttpUrl

from research_agent.search.models import (
    PaperInfo,
    PaperSource,
    ResearchQuery,
    SearchIndexReference,
    SearchIndexType,
    SearchResult,
)
from research_agent.search.workflows import PaperSearchWorkflow

_ABSTRACT = (
    "A sufficiently long abstract describing the research methodology, "
    "experimental setup, results, and conclusions of this work in detail "
    "to satisfy the PaperInfo min_length=200 invariant enforced by Pydantic."
)
_TITLE_A = "Paper Alpha On Quantum Computing Advances"
_TITLE_B = "Paper Beta On Quantum Computing Advances"
_TITLE_C = "Paper Gamma On Quantum Computing Advances"


def _paper(title: str) -> PaperInfo:
    return PaperInfo(
        source=PaperSource(url=HttpUrl("https://example.com/p"), open_access=False),
        title=title,
        abstract=_ABSTRACT,
        authors=("Alice",),
    )


def _result(paper: PaperInfo) -> SearchResult:
    return SearchResult(
        paper=paper,
        search_index_reference=(
            SearchIndexReference(index=SearchIndexType.ARXIV, id="1"),
        ),
    )


class _FakeSearchAgent:
    """In-memory fake of the search agent port."""

    def __init__(self, results: list[SearchResult]) -> None:
        self.calls: list[ResearchQuery] = []
        self._results = results

    async def __call__(self, data: ResearchQuery) -> list[SearchResult]:
        self.calls.append(data)
        return list(self._results)


class _FakeReranker:
    """In-memory fake of the reranker agent port."""

    def __init__(self, reordered: list[SearchResult]) -> None:
        self.calls: list[tuple[ResearchQuery, list[SearchResult]]] = []
        self._reordered = reordered

    async def __call__(
        self,
        data: tuple[ResearchQuery, list[SearchResult]],
    ) -> list[SearchResult]:
        self.calls.append(data)
        return list(self._reordered)


def _query() -> ResearchQuery:
    return ResearchQuery(text="quantum computing")


@pytest.mark.asyncio
async def test_returns_reranked_results() -> None:
    a, b, c = _paper(_TITLE_A), _paper(_TITLE_B), _paper(_TITLE_C)
    results = [_result(a), _result(b), _result(c)]
    reordered = [results[2], results[0], results[1]]
    workflow = PaperSearchWorkflow(_FakeSearchAgent(results), _FakeReranker(reordered))

    out = await workflow(_query())

    assert out == reordered


@pytest.mark.asyncio
async def test_skips_rerank_on_empty_results() -> None:
    search, rerank = _FakeSearchAgent([]), _FakeReranker([])
    workflow = PaperSearchWorkflow(search, rerank)

    out = await workflow(_query())

    assert out == []
    assert rerank.calls == []


@pytest.mark.asyncio
async def test_passes_query_to_search_agent() -> None:
    query = _query()
    search, rerank = _FakeSearchAgent([]), _FakeReranker([])
    workflow = PaperSearchWorkflow(search, rerank)

    await workflow(query)

    assert search.calls == [query]


@pytest.mark.asyncio
async def test_passes_query_and_results_to_reranker() -> None:
    query = _query()
    results = [_result(_paper(_TITLE_A))]
    search, rerank = _FakeSearchAgent(results), _FakeReranker(results)
    workflow = PaperSearchWorkflow(search, rerank)

    await workflow(query)

    assert rerank.calls == [(query, results)]
