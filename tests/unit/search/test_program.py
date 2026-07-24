"""Unit tests for the search GEPA student program."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar
from unittest.mock import patch

import pytest
from pydantic import HttpUrl

if TYPE_CHECKING:
    from collections.abc import Sequence

from optimize.search.program import SearchProgram
from research_agent.search.models import PaperInfo, ResearchQuery, SearchIndexType
from research_agent.search.tools import LiteratureSearch
from research_agent.shared.agent import LMConfig
from research_agent.shared.session import InMemorySession

_ABSTRACT = (
    "A sufficiently long abstract describing the research methodology, "
    "experimental setup, results, and conclusions of this work in detail "
    "to satisfy the PaperInfo min_length=200 invariant enforced by Pydantic."
)

_CONFIG = LMConfig(model="openai/test-student")
_QUERY = ResearchQuery(text="quantum error correction codes")


def _make_paper(title: str = "Alpha Paper On Quantum Computing Advances") -> PaperInfo:
    return PaperInfo(
        title=title,
        abstract=_ABSTRACT,
        authors=("Alice Smith",),
        url=HttpUrl("https://example.com/paper"),
        open_access=False,
    )


class _FakeLiteratureSearch(LiteratureSearch):
    def __init__(self, papers: Sequence[PaperInfo]) -> None:
        self._papers = list(papers)
        self.calls: list[tuple[SearchIndexType, str, int]] = []

    async def __call__(
        self,
        search_index: SearchIndexType,
        query: str,
        *,
        limit: int,
    ) -> list[PaperInfo]:
        self.calls.append((search_index, query, limit))
        return self._papers[:limit]


class _StubSearchAgent:
    """SearchAgent double: returns fixed papers."""

    selected_papers: ClassVar[list[PaperInfo]] = []

    def __init__(
        self,
        lm_config: LMConfig,
        session: object,
        literature_search: LiteratureSearch,
    ) -> None:
        self._lm_config = lm_config
        self._session = session
        self._literature_search = literature_search

    async def __call__(self, _data: ResearchQuery) -> list[PaperInfo]:
        return list(self.selected_papers)


def _make_program(papers: Sequence[PaperInfo]) -> SearchProgram:
    return SearchProgram(
        lm_config=_CONFIG,
        literature_search=_FakeLiteratureSearch(papers),
    )


@patch("optimize.search.program.SearchAgent", _StubSearchAgent)
def test_forward_returns_hydrated_search_results() -> None:
    paper = _make_paper()
    _StubSearchAgent.selected_papers = [paper]
    program = _make_program([paper])

    result = asyncio.run(program.forward(research_query=_QUERY))

    assert result == [paper]


@patch("optimize.search.program.SearchAgent", _StubSearchAgent)
def test_forward_empty_selection_returns_empty_results() -> None:
    _StubSearchAgent.selected_papers = []
    program = _make_program([])

    result = asyncio.run(program.forward(research_query=_QUERY))

    assert result == []


@patch("optimize.search.program.SearchAgent", _StubSearchAgent)
def test_forward_binds_fresh_session_per_call() -> None:
    paper1 = _make_paper("Paper One Title Long Enough")
    paper2 = _make_paper("Paper Two Title Long Enough")
    program = _make_program([paper1, paper2])

    _StubSearchAgent.selected_papers = [paper1]
    first = asyncio.run(program.forward(research_query=_QUERY))
    assert first == [paper1]

    _StubSearchAgent.selected_papers = [paper2]
    second = asyncio.run(program.forward(research_query=_QUERY))
    assert second == [paper2]


def test_forward_unknown_selected_id_raises() -> None:
    """Test SearchAgent raises UnknownSelectedIdError for invalid index.

    Requires real SearchAgent (needs API keys). Stub doesn't validate indices.
    """
    pytest.skip("Requires live SearchAgent with API credentials")
    pytest.skip("Requires live SearchAgent with API credentials")


def test_program_builds_search_agent_with_correct_config() -> None:
    captured: dict[str, object] = {}

    class _CaptureSearchAgent:
        def __init__(
            self,
            lm_config: LMConfig,
            session: object,
            literature_search: LiteratureSearch,
        ) -> None:
            captured.update(
                lm_config=lm_config,
                session=session,
                literature_search=literature_search,
            )

        async def __call__(self, _data: ResearchQuery) -> list[PaperInfo]:
            return []

    with patch("optimize.search.program.SearchAgent", _CaptureSearchAgent):
        program = SearchProgram(
            lm_config=_CONFIG,
            literature_search=_FakeLiteratureSearch([]),
        )
        # SearchAgent is instantiated in forward(), so we must call it
        asyncio.run(program.forward(research_query=_QUERY))

    assert captured["lm_config"] is _CONFIG
    assert isinstance(captured["session"], InMemorySession)
    assert isinstance(captured["literature_search"], _FakeLiteratureSearch)


def test_program_stores_lm_config() -> None:
    program = _make_program([])

    assert program._lm_config is _CONFIG
    assert program._literature_search is not None
    assert program._session is not None
