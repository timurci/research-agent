"""Unit tests for the combined search GEPA metric."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback
from pydantic import HttpUrl

from optimize.search.metrics import (
    MetricShapeError,
    _relevance_metrics_from_ranking,
    _research_query_from_example,
    _search_results_from_pred,
    search_query_metric,
    search_suggest_metric,
)
from research_agent.search.metrics import (
    SUGGESTION_MIN_WORDS,
    RelevanceMetric,
    search_result_count,
)
from research_agent.search.models import PaperInfo, ResearchQuery
from research_agent.shared.judge import JudgeVerdict

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

_ABSTRACT = (
    "A sufficiently long abstract describing the research methodology, "
    "experimental setup, results, and conclusions of this work in detail "
    "to satisfy the PaperInfo min_length=200 invariant enforced by Pydantic."
)

_TITLE_A = "Alpha Paper On Quantum Computing Advances"
_TITLE_B = "Beta Paper On Neural Network Optimization"


def _make_paper(title: str = _TITLE_A) -> PaperInfo:
    return PaperInfo(
        title=title,
        abstract=_ABSTRACT,
        authors=("Alice Smith",),
        url=HttpUrl("https://example.com/paper"),
        open_access=False,
    )


class _FakeReranker:
    """Test double for the relevance labeler protocol."""

    def __init__(self, scores: Sequence[float]) -> None:
        self._scores = list(scores)
        self.calls: list[tuple[ResearchQuery, list[PaperInfo]]] = []

    async def relevance(
        self,
        data: tuple[ResearchQuery, list[PaperInfo]],
    ) -> list[dict[str, object]]:
        query, papers = data
        self.calls.append((query, papers))
        if len(self._scores) != len(papers):
            msg = "fake score count must match paper count"
            raise AssertionError(msg)
        return [
            {"index": index, "relevance_score": score}
            for index, score in enumerate(self._scores)
        ]


def test_research_query_from_example_attribute() -> None:
    query = ResearchQuery(text="quantum error correction codes")
    assert _research_query_from_example(SimpleNamespace(research_query=query)) == query


def test_research_query_from_example_dict() -> None:
    query = ResearchQuery(text="quantum error correction codes")
    assert _research_query_from_example({"research_query": query}) == query


def test_research_query_from_example_rejects_missing() -> None:
    with pytest.raises(MetricShapeError, match="research_query"):
        _research_query_from_example(SimpleNamespace())


def test_search_results_from_pred_rejects_missing() -> None:
    with pytest.raises(MetricShapeError, match="search_results"):
        _search_results_from_pred(SimpleNamespace())


def test_relevance_metrics_from_ranking_aligns_by_index() -> None:
    papers = [_make_paper(_TITLE_A), _make_paper(_TITLE_B)]
    ranking: Sequence[Mapping[str, object]] = [
        {"index": 1, "relevance_score": 0.2},
        {"index": 0, "relevance_score": 0.9},
    ]
    metrics = _relevance_metrics_from_ranking(papers, ranking)
    assert metrics == [
        RelevanceMetric(value=0.9),
        RelevanceMetric(value=0.2),
    ]


def test_relevance_metrics_from_ranking_clamps_out_of_range_scores() -> None:
    papers = [_make_paper(_TITLE_A), _make_paper(_TITLE_B)]
    ranking: Sequence[Mapping[str, object]] = [
        {"index": 0, "relevance_score": 1.5},
        {"index": 1, "relevance_score": -0.2},
    ]
    metrics = _relevance_metrics_from_ranking(papers, ranking)
    assert metrics == [
        RelevanceMetric(value=1.0),
        RelevanceMetric(value=0.0),
    ]


def test_search_query_metric_combines_count_and_relevance() -> None:
    papers = [_make_paper(_TITLE_A), _make_paper(_TITLE_B)]
    fake = _FakeReranker([1.0, 1.0])
    metric = search_query_metric(labeler=fake)
    query = ResearchQuery(text="quantum computing survey methods")
    gold = SimpleNamespace(research_query=query)
    pred = SimpleNamespace(search_results=papers)

    result = metric(gold, pred)

    count = search_result_count(papers)
    expected = (count.score + 1.0) / 2
    assert isinstance(result, ScoreWithFeedback)
    assert result.score == pytest.approx(expected)
    assert "search_result_count" in result.feedback
    assert "search_result_relevance" in result.feedback
    assert "Returned 2 results" in result.feedback
    assert len(fake.calls) == 1
    assert fake.calls[0][0] == query
    assert fake.calls[0][1] == papers


def test_search_query_metric_empty_results_scores_zero() -> None:
    fake = _FakeReranker([])
    metric = search_query_metric(labeler=fake)
    result = metric(
        SimpleNamespace(research_query=ResearchQuery(text="neural network methods")),
        SimpleNamespace(search_results=[]),
    )

    assert result.score == 0.0
    assert "Empty input" in result.feedback
    assert fake.calls == []


def test_search_query_metric_accepts_dict_paper_payloads() -> None:
    paper = _make_paper(_TITLE_A)
    fake = _FakeReranker([1.0])
    metric = search_query_metric(labeler=fake)
    result = metric(
        SimpleNamespace(research_query=ResearchQuery(text="quantum computing survey")),
        SimpleNamespace(search_results=[paper.model_dump(mode="json")]),
    )
    assert result.score > 0.0
    assert fake.calls


class _FakeQualityJudge:
    """Test double for the suggestion quality judge protocol."""

    def __init__(self, verdict: JudgeVerdict) -> None:
        self._verdict = verdict
        self.calls: list[dict[str, str]] = []

    async def judge(
        self,
        *,
        task_input: str,
        task_output: str,
        task_context: str = "",
    ) -> JudgeVerdict:
        self.calls.append(
            {
                "task_input": task_input,
                "task_output": task_output,
                "task_context": task_context,
            },
        )
        return self._verdict


def test_search_suggest_metric_combines_length_and_quality() -> None:
    papers = [_make_paper(_TITLE_A)]
    suggestion = " ".join(f"w{i}" for i in range(SUGGESTION_MIN_WORDS))
    fake = _FakeQualityJudge(
        JudgeVerdict(score=1.0, failing=False, reason="Strong direction."),
    )
    metric = search_suggest_metric(quality_judge=fake)
    query = ResearchQuery(text="quantum computing survey methods")
    gold = SimpleNamespace(research_query=query, papers=papers)
    pred = SimpleNamespace(suggestion=suggestion)

    result = metric(gold, pred)

    assert isinstance(result, ScoreWithFeedback)
    assert result.score == pytest.approx(1.0)
    assert "suggestion_length" in result.feedback
    assert "suggestion_quality" in result.feedback
    assert len(fake.calls) == 1
    assert fake.calls[0]["task_output"] == suggestion
    assert _TITLE_A in fake.calls[0]["task_context"]


def test_search_suggest_metric_rejects_missing_suggestion() -> None:
    metric = search_suggest_metric(
        quality_judge=_FakeQualityJudge(
            JudgeVerdict(score=0.0, failing=True, reason="x"),
        ),
    )
    with pytest.raises(MetricShapeError, match="suggestion"):
        metric(
            SimpleNamespace(
                research_query=ResearchQuery(text="quantum computing survey"),
                papers=[_make_paper()],
            ),
            SimpleNamespace(),
        )


def test_search_suggest_metric_rejects_missing_papers() -> None:
    metric = search_suggest_metric(
        quality_judge=_FakeQualityJudge(
            JudgeVerdict(score=1.0, failing=False, reason="ok"),
        ),
    )
    with pytest.raises(MetricShapeError, match="papers"):
        metric(
            SimpleNamespace(research_query=ResearchQuery(text="quantum computing")),
            SimpleNamespace(suggestion="short"),
        )
