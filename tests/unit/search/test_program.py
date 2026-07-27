"""Unit tests for the search GEPA student program."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import dspy
from pydantic import HttpUrl

from optimize.search.agents import SearchProgram
from research_agent.search.agents import SearchOutcome
from research_agent.search.models import (
    PaperInfo,
    ResearchQuery,
    SearchIndex,
)
from research_agent.search.tools import (
    LiteratureSearch,
    SessionLiteratureSearch,
    load_search_results,
)
from research_agent.shared.config.models import LMConfig

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
        self.calls: list[tuple[SearchIndex, str, int]] = []

    async def __call__(
        self,
        search_index: SearchIndex,
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


def test_forward_returns_react_pred_with_isolated_bag() -> None:
    paper_a = _make_paper("Alpha Paper On Quantum Computing Advances")
    paper_b = _make_paper("Beta Paper On Neural Network Optimization")
    program = _make_program([paper_a, paper_b])
    trajectory = {"thought_0": "search", "tool_name_0": "LiteratureSearch"}
    react_preds: list[dspy.Prediction] = []

    def _fake_react(*, research_query: ResearchQuery) -> dspy.Prediction:
        assert research_query == _QUERY
        bag = load_search_results(program.session)
        bag.add(paper_a)
        bag.add(paper_b)
        pred = dspy.Prediction(
            trajectory=trajectory,
            status=SearchOutcome.COMPLETE,
        )
        react_preds.append(pred)
        return pred

    program.react = MagicMock(side_effect=_fake_react)

    result = program(research_query=_QUERY)

    assert result is react_preds[0]
    assert result.trajectory == trajectory
    assert set(result.search_results) == {paper_a, paper_b}
    assert result.status == SearchOutcome.COMPLETE


def test_forward_empty_bag_returns_empty_results() -> None:
    program = _make_program([])

    def _fake_react(*, research_query: ResearchQuery) -> dspy.Prediction:
        del research_query
        return dspy.Prediction(
            trajectory={},
            status=SearchOutcome.MISSING_RESULTS,
        )

    program.react = MagicMock(side_effect=_fake_react)

    result = program(research_query=_QUERY)

    assert result.search_results == []
    assert result.status == SearchOutcome.MISSING_RESULTS
    assert result.trajectory == {}


def test_forward_isolates_bags_across_sequential_calls() -> None:
    paper1 = _make_paper("Paper One Title Long Enough")
    paper2 = _make_paper("Paper Two Title Long Enough")
    program = _make_program([paper1, paper2])
    bags = [{paper1}, {paper2}]
    call_index = 0

    def _fake_react(*, research_query: ResearchQuery) -> dspy.Prediction:
        nonlocal call_index
        del research_query
        bag = load_search_results(program.session)
        assert bag == set()
        bag.update(bags[call_index])
        call_index += 1
        return dspy.Prediction(status=SearchOutcome.COMPLETE)

    program.react = MagicMock(side_effect=_fake_react)

    first = program(research_query=_QUERY)
    second = program(research_query=_QUERY)

    assert set(first.search_results) == {paper1}
    assert set(second.search_results) == {paper2}


def test_forward_isolates_bags_across_threads() -> None:
    paper_a = _make_paper("Alpha Paper On Quantum Computing Advances")
    paper_b = _make_paper("Beta Paper On Neural Network Optimization")
    program = _make_program([])
    query_a = ResearchQuery(text="alpha query about quantum error")
    query_b = ResearchQuery(text="beta query about neural nets")

    def _fake_react(*, research_query: ResearchQuery) -> dspy.Prediction:
        bag = load_search_results(program.session)
        assert bag == set()
        if research_query == query_a:
            bag.add(paper_a)
        elif research_query == query_b:
            bag.add(paper_b)
        else:
            msg = f"unexpected query {research_query!r}"
            raise AssertionError(msg)
        return dspy.Prediction(status=SearchOutcome.COMPLETE)

    program.react = MagicMock(side_effect=_fake_react)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(program, research_query=query_a),
            pool.submit(program, research_query=query_b),
            pool.submit(program, research_query=query_a),
            pool.submit(program, research_query=query_b),
        ]
        results = [future.result() for future in futures]

    assert set(results[0].search_results) == {paper_a}
    assert set(results[1].search_results) == {paper_b}
    assert set(results[2].search_results) == {paper_a}
    assert set(results[3].search_results) == {paper_b}


def test_program_builds_react_with_shared_builder() -> None:
    fake_react = object()
    with patch("optimize.search.agents.build_search_react") as build:
        build.return_value = fake_react
        program = SearchProgram(
            lm_config=_CONFIG,
            literature_search=_FakeLiteratureSearch([]),
        )

    build.assert_called_once()
    assert build.call_args.args[0] is program._session_search
    assert isinstance(build.call_args.args[0], SessionLiteratureSearch)
    assert program.react is fake_react


def test_program_stores_lm_config() -> None:
    program = _make_program([])

    assert program._lm_config is _CONFIG
    assert program._literature_search is not None
    assert program.session is not None
    assert program.react is not None
