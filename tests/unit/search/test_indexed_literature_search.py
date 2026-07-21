"""Unit tests for IndexedLiteratureSearch."""

from __future__ import annotations

import pytest
from pydantic import HttpUrl

from research_agent.search.models import PaperInfo, SearchIndexType
from research_agent.search.tools import (
    SEARCH_RESULTS_KEY,
    IndexedLiteratureSearch,
    LiteratureSearch,
)
from research_agent.shared.session import InMemorySession, InvalidSessionStateError

_ABSTRACT = (
    "A sufficiently long abstract describing the research methodology, "
    "experimental setup, results, and conclusions of this work in detail "
    "to satisfy the PaperInfo min_length=200 invariant enforced by Pydantic."
)


def _make_paper(title: str = "Alpha Paper On Quantum Computing Advances") -> PaperInfo:
    return PaperInfo(
        title=title,
        abstract=_ABSTRACT,
        authors=("Alice Smith",),
        url=HttpUrl("https://example.com/paper"),
        open_access=False,
        publication_year=2019,
    )


class _FakeLiteratureSearch(LiteratureSearch):
    """LiteratureSearch stand-in that returns fixed papers without APIs."""

    def __init__(self, batches: list[list[PaperInfo]]) -> None:
        self._batches = list(batches)
        self.calls: list[tuple[SearchIndexType, str, int]] = []

    async def __call__(
        self,
        search_index: SearchIndexType,
        query: str,
        *,
        limit: int,
    ) -> list[PaperInfo]:
        self.calls.append((search_index, query, limit))
        if not self._batches:
            return []
        batch = self._batches.pop(0)
        return batch[:limit]


@pytest.mark.asyncio
async def test_indexes_and_appends_across_calls() -> None:
    paper_a = _make_paper("Alpha Paper On Quantum Computing Advances")
    paper_b = _make_paper("Beta Paper On Neural Network Optimization")
    paper_c = _make_paper("Gamma Paper On Climate Modeling Techniques")
    session = InMemorySession()
    inner = _FakeLiteratureSearch([[paper_a, paper_b], [paper_c]])
    tool = IndexedLiteratureSearch(session, inner)

    first = await tool(SearchIndexType.CROSSREF, "q1", limit=5)
    assert [item.id for item in first] == [0, 1]
    assert first[0].paper == paper_a
    assert first[1].paper == paper_b

    second = await tool(SearchIndexType.PUBMED, "q2", limit=5)
    assert [item.id for item in second] == [2]
    assert second[0].paper == paper_c

    bag = session.get(SEARCH_RESULTS_KEY)
    assert bag == [paper_a, paper_b, paper_c]
    assert inner.calls == [
        (SearchIndexType.CROSSREF, "q1", 5),
        (SearchIndexType.PUBMED, "q2", 5),
    ]


@pytest.mark.asyncio
async def test_empty_results() -> None:
    session = InMemorySession()
    tool = IndexedLiteratureSearch(session, _FakeLiteratureSearch([[]]))
    assert await tool(SearchIndexType.CROSSREF, "q", limit=3) == []
    assert session.get(SEARCH_RESULTS_KEY) == []


@pytest.mark.asyncio
async def test_corrupt_bag_raises_invalid_session_state() -> None:
    session = InMemorySession()
    session.set(SEARCH_RESULTS_KEY, "not-a-list")
    tool = IndexedLiteratureSearch(
        session,
        _FakeLiteratureSearch([[_make_paper()]]),
    )
    with pytest.raises(InvalidSessionStateError, match="list"):
        await tool(SearchIndexType.PUBMED, "q", limit=1)
