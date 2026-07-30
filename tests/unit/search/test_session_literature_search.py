"""Unit tests for SessionLiteratureSearch."""

from __future__ import annotations

import copy

import pytest
from pydantic import HttpUrl

from research_agent.search.models import PaperInfo, SearchIndex
from research_agent.search.tools import (
    _SEARCH_MAX_ABSTRACT_CHARS,
    SEARCH_RESULTS_KEY,
    LiteratureSearch,
    SessionLiteratureSearch,
)
from research_agent.shared.scoped_session import ScopedSession
from research_agent.shared.session import InMemorySession

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
        self.calls: list[tuple[SearchIndex, str, int]] = []

    async def __call__(
        self,
        search_index: SearchIndex,
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

    first = await tool(SearchIndex.CROSSREF, "q1", limit=5)
    assert first == [
        {"title": paper_a.title, "abstract": paper_a.abstract},
        {"title": paper_b.title, "abstract": paper_b.abstract},
    ]
    assert all(set(card) == {"title", "abstract"} for card in first)

    second = await tool(SearchIndex.PUBMED, "q2", limit=5)
    assert second == [{"title": paper_c.title, "abstract": paper_c.abstract}]

    bag = session.get(SEARCH_RESULTS_KEY)
    assert bag == {paper_a, paper_b, paper_c}
    assert inner.calls == [
        (SearchIndex.CROSSREF, "q1", 5),
        (SearchIndex.PUBMED, "q2", 5),
    ]


@pytest.mark.asyncio
async def test_empty_results_leaves_none_or_empty() -> None:
    session = InMemorySession()
    tool = SessionLiteratureSearch(session, _FakeLiteratureSearch([[]]))
    assert await tool(SearchIndex.CROSSREF, "q", limit=3) == []
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
        await tool(SearchIndex.PUBMED, "q", limit=1)


@pytest.mark.asyncio
async def test_missing_key_treated_as_empty() -> None:
    session = InMemorySession()
    paper = _make_paper()
    tool = SessionLiteratureSearch(session, _FakeLiteratureSearch([[paper]]))
    cards = await tool(SearchIndex.OPENALEX, "q", limit=1)
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
        await tool(SearchIndex.OPENALEX, "q", limit=1)


@pytest.mark.asyncio
async def test_card_abstract_truncated_but_bag_keeps_full_abstract() -> None:
    long_abstract = "A" * 4000
    paper = PaperInfo(
        title="Alpha Paper On Quantum Computing Advances",
        abstract=long_abstract,
        authors=("Alice Smith",),
        url=HttpUrl("https://example.com/paper"),
        open_access=False,
    )
    session = InMemorySession()
    tool = SessionLiteratureSearch(session, _FakeLiteratureSearch([[paper]]))

    cards = await tool(SearchIndex.OPENALEX, "q", limit=1)

    assert cards == [
        {
            "title": paper.title,
            "abstract": long_abstract[:_SEARCH_MAX_ABSTRACT_CHARS],
        }
    ]
    assert len(cards[0]["abstract"]) == _SEARCH_MAX_ABSTRACT_CHARS
    bag = SessionLiteratureSearch.papers(session)
    assert len(bag) == 1
    stored = next(iter(bag))
    assert stored.abstract == long_abstract


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

    await tool(SearchIndex.CROSSREF, "q1", limit=1)
    assert SessionLiteratureSearch.papers(base) == {paper_a}

    fresh = InMemorySession()
    with scoped.use(fresh):
        await tool(SearchIndex.PUBMED, "q2", limit=1)
        assert SessionLiteratureSearch.papers(fresh) == {paper_b}
    assert base.get(SEARCH_RESULTS_KEY) == {paper_a}


def test_deepcopy_creates_independent_instance_with_shared_dispatcher() -> None:
    session = InMemorySession()
    inner = _FakeLiteratureSearch([])
    tool = SessionLiteratureSearch(session, inner)

    copy_tool = copy.deepcopy(tool)

    assert copy_tool is not tool
    assert copy_tool._session is not tool._session
    assert copy_tool._literature_search is tool._literature_search


@pytest.mark.asyncio
async def test_deepcopy_isolates_bag() -> None:
    paper_a = _make_paper("Alpha Paper On Quantum Computing Advances")
    paper_b = _make_paper("Beta Paper On Neural Network Optimization")
    session = InMemorySession()
    inner = _FakeLiteratureSearch([[paper_a], [paper_b]])
    tool = SessionLiteratureSearch(session, inner)

    copy_tool = copy.deepcopy(tool)

    await tool(SearchIndex.CROSSREF, "q1", limit=1)
    assert SessionLiteratureSearch.papers(session) == {paper_a}

    copy_session = copy_tool._session
    await copy_tool(SearchIndex.PUBMED, "q2", limit=1)
    assert SessionLiteratureSearch.papers(copy_session) == {paper_b}
    assert SessionLiteratureSearch.papers(session) == {paper_a}
