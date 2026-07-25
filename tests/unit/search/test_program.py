"""Unit tests for the search GEPA student program."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import dspy
from pydantic import HttpUrl

from optimize.search.agents import SearchProgram
from research_agent.search.models import PaperInfo, ResearchQuery, SearchIndexType
from research_agent.search.tools import SEARCH_RESULTS_KEY, LiteratureSearch
from research_agent.shared.agent import LMConfig
from research_agent.shared.session import InMemorySession

if TYPE_CHECKING:
    from collections.abc import Sequence

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


def _make_program(papers: Sequence[PaperInfo]) -> SearchProgram:
    return SearchProgram(
        lm_config=_CONFIG,
        literature_search=_FakeLiteratureSearch(papers),
    )


def test_search_program_exposes_react_predictors() -> None:
    program = _make_program([])
    names = [name for name, _ in program.named_predictors()]
    assert names
    assert any(name.startswith("react") for name in names)


def test_forward_returns_prediction_with_search_results() -> None:
    paper = _make_paper()
    program = _make_program([paper])

    def _fake_react(*, research_query: ResearchQuery) -> dspy.Prediction:
        assert research_query == _QUERY
        program._session.set(SEARCH_RESULTS_KEY, [paper])
        return dspy.Prediction(selected_ids=[0])

    program.react = MagicMock(side_effect=_fake_react)

    result = program(research_query=_QUERY)

    assert isinstance(result, dspy.Prediction)
    assert result.search_results == [paper]


def test_forward_empty_selection_returns_empty_results() -> None:
    program = _make_program([])

    def _fake_react(*, research_query: ResearchQuery) -> dspy.Prediction:
        del research_query
        return dspy.Prediction(selected_ids=[])

    program.react = MagicMock(side_effect=_fake_react)

    result = program(research_query=_QUERY)

    assert result.search_results == []


def test_forward_isolates_session_across_calls() -> None:
    paper1 = _make_paper("Paper One Title Long Enough")
    paper2 = _make_paper("Paper Two Title Long Enough")
    program = _make_program([paper1, paper2])
    selections = [[0], [1]]
    call_index = 0

    def _fake_react(*, research_query: ResearchQuery) -> dspy.Prediction:
        nonlocal call_index
        del research_query
        assert program._session.get(SEARCH_RESULTS_KEY) == []
        papers = [paper1, paper2]
        program._session.set(SEARCH_RESULTS_KEY, papers)
        selected = selections[call_index]
        call_index += 1
        return dspy.Prediction(selected_ids=selected)

    program.react = MagicMock(side_effect=_fake_react)

    first = program(research_query=_QUERY)
    second = program(research_query=_QUERY)

    assert first.search_results == [paper1]
    assert second.search_results == [paper2]


def test_program_builds_react_with_shared_builder() -> None:
    fake_react = object()
    with patch("optimize.search.agents.build_search_react") as build:
        build.return_value = fake_react
        program = SearchProgram(
            lm_config=_CONFIG,
            literature_search=_FakeLiteratureSearch([]),
        )

    build.assert_called_once()
    assert build.call_args.args[0] is program._session
    assert isinstance(build.call_args.args[0], InMemorySession)
    assert isinstance(build.call_args.args[1], _FakeLiteratureSearch)
    assert program.react is fake_react


def test_program_stores_lm_config() -> None:
    program = _make_program([])

    assert program._lm_config is _CONFIG
    assert program._literature_search is not None
    assert program._session is not None
    assert program.react is not None
