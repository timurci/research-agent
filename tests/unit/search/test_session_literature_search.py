"""Unit tests for SessionLiteratureSearch."""

from __future__ import annotations

import pytest
from pydantic import HttpUrl

from research_agent.search.models import PaperInfo, SearchIndexType
from research_agent.search.tools import (
    SEARCH_RESULTS_KEY,
    LiteratureSearch,
    SessionLiteratureSearch,
)
from research_agent.shared.session import InMemorySession, ScopedSession

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
async def test_unions_into_set_and_returns_new_cards_only() -> None:
    paper_a = _make_paper("Alpha Paper On Quantum Computing Advances")
    paper_b = _make_paper("Beta Paper On Neural Network Optimization")
    paper_c = _make_paper("Gamma Paper On Climate Modeling Techniques")
    session = InMemorySession()
    inner = _FakeLiteratureSearch([[paper_a, paper_b], [paper_b, paper_c]])
    tool = SessionLiteratureSearch(session, inner)

    first = await tool(SearchIndexType.CROSSREF, "q1", limit=5)
    assert first == [
        {"title": paper_a.title, "abstract": paper_a.abstract},
        {"title": paper_b.title, "abstract": paper_b.abstract},
    ]
    assert all(set(card) == {"title", "abstract"} for card in first)

    second = await tool(SearchIndexType.PUBMED, "q2", limit=5)
    assert second == [{"title": paper_c.title, "abstract": paper_c.abstract}]

    bag = session.get(SEARCH_RESULTS_KEY)
    assert bag == {paper_a, paper_b, paper_c}
    assert inner.calls == [
        (SearchIndexType.CROSSREF, "q1", 5),
        (SearchIndexType.PUBMED, "q2", 5),
    ]


@pytest.mark.asyncio
async def test_empty_results_leaves_none_or_empty() -> None:
    session = InMemorySession()
    tool = SessionLiteratureSearch(session, _FakeLiteratureSearch([[]]))
    assert await tool(SearchIndexType.CROSSREF, "q", limit=3) == []
    assert session.get(SEARCH_RESULTS_KEY) == set()


@pytest.mark.asyncio
async def test_corrupt_bag_raises_type_error() -> None:
    session = InMemorySession()
    session.set(SEARCH_RESULTS_KEY, "not-a-set")
    tool = SessionLiteratureSearch(
        session,
        _FakeLiteratureSearch([[_make_paper()]]),
    )
    with pytest.raises(TypeError, match="set"):
        await tool(SearchIndexType.PUBMED, "q", limit=1)


@pytest.mark.asyncio
async def test_missing_key_treated_as_empty() -> None:
    session = InMemorySession()
    paper = _make_paper()
    tool = SessionLiteratureSearch(session, _FakeLiteratureSearch([[paper]]))
    cards = await tool(SearchIndexType.OPENALEX, "q", limit=1)
    assert cards == [{"title": paper.title, "abstract": paper.abstract}]
    assert session.get(SEARCH_RESULTS_KEY) == {paper}


@pytest.mark.asyncio
async def test_none_value_raises() -> None:
    session = InMemorySession()
    session.set(SEARCH_RESULTS_KEY, None)
    tool = SessionLiteratureSearch(
        session,
        _FakeLiteratureSearch([[_make_paper()]]),
    )
    with pytest.raises(TypeError, match="set"):
        await tool(SearchIndexType.OPENALEX, "q", limit=1)


@pytest.mark.asyncio
async def test_scoped_session_isolates_bags() -> None:
    paper_a = _make_paper("Alpha Paper On Quantum Computing Advances")
    paper_b = _make_paper("Beta Paper On Neural Network Optimization")
    base = InMemorySession()
    scoped = ScopedSession(base)
    tool = SessionLiteratureSearch(
        scoped,
        _FakeLiteratureSearch([[paper_a], [paper_b]]),
    )

    await tool(SearchIndexType.CROSSREF, "q1", limit=1)
    assert SessionLiteratureSearch.papers(base) == {paper_a}

    fresh = InMemorySession()
    with scoped.use(fresh):
        await tool(SearchIndexType.PUBMED, "q2", limit=1)
        assert SessionLiteratureSearch.papers(fresh) == {paper_b}
    assert base.get(SEARCH_RESULTS_KEY) == {paper_a}
