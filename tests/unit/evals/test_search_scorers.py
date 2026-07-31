"""Unit tests for search Opik scoring metric adapters.

Exercises agent-aligned scoring and reranker-backed relevance. No
network, no ``opik.evaluate`` run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from opik.evaluation.metrics.score_result import ScoreResult
from pydantic import HttpUrl

from evals.search.scorers import (
    ScorerShapeError,
    SearchResultCountMetric,
    SearchResultRelevanceMetric,
    SuggestionLengthMetric,
    SuggestionQualityMetric,
    relevance_metrics_from_ranking,
)
from research_agent.search.metrics import (
    SUGGESTION_MIN_WORDS,
    RelevanceMetric,
)
from research_agent.search.models import PaperInfo, ResearchQuery
from research_agent.search.rubrics import (
    format_suggestion_judge_context,
    format_suggestion_judge_input,
)
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


def _make_paper(
    title: str = _TITLE_A,
    *,
    doi: str | None = None,
) -> PaperInfo:
    return PaperInfo(
        title=title,
        abstract=_ABSTRACT,
        authors=("Alice Smith",),
        url=HttpUrl("https://example.com/paper"),
        open_access=False,
        doi=doi,
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


def test_count_metric_below_min() -> None:
    papers = [_make_paper(_TITLE_A), _make_paper(_TITLE_B)]
    metric = SearchResultCountMetric()
    result = metric.score(papers=papers)  # ty: ignore[missing-argument]  # false positive: union of bound/unbound score()

    assert isinstance(result, ScoreResult)
    assert result.name == "search_result_count"
    assert result.value == 0.0
    assert result.metadata == {"passing": "False"}
    assert "Returned 2 results" in (result.reason or "")
    assert "need at least 10" in (result.reason or "")


def test_count_metric_pass_with_partial_score() -> None:
    papers = [_make_paper(f"Paper Number {i:03d}") for i in range(20)]
    metric = SearchResultCountMetric()
    result = metric.score(papers=papers)  # ty: ignore[missing-argument]  # false positive: union of bound/unbound score()

    assert isinstance(result, ScoreResult)
    assert result.value == 0.5
    assert result.metadata == {"passing": "True"}
    assert "Returned 20 results" in (result.reason or "")


def test_count_metric_meets_peak() -> None:
    papers = [_make_paper(f"Paper Number {i:03d}") for i in range(30)]
    metric = SearchResultCountMetric()
    result = metric.score(papers=papers)  # ty: ignore[missing-argument]  # false positive: union of bound/unbound score()

    assert isinstance(result, ScoreResult)
    assert result.value == 1.0
    assert result.metadata == {"passing": "True"}
    assert "peak 30 met" in (result.reason or "")


def test_relevance_metrics_from_ranking_aligns_by_index() -> None:
    papers = [_make_paper(_TITLE_A), _make_paper(_TITLE_B)]
    ranking: Sequence[Mapping[str, object]] = [
        {"index": 1, "relevance_score": 0.2},
        {"index": 0, "relevance_score": 0.9},
    ]
    metrics = relevance_metrics_from_ranking(papers, ranking)
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
    metrics = relevance_metrics_from_ranking(papers, ranking)
    assert metrics == [
        RelevanceMetric(value=1.0),
        RelevanceMetric(value=0.0),
    ]


def test_relevance_metrics_from_ranking_rejects_length_mismatch() -> None:
    with pytest.raises(ScorerShapeError, match="lengths must match"):
        relevance_metrics_from_ranking(
            [_make_paper()],
            [
                {"index": 0, "relevance_score": 0.5},
                {"index": 1, "relevance_score": 0.1},
            ],
        )


def test_require_paper_list_rejects_non_list() -> None:
    metric = SearchResultCountMetric()
    with pytest.raises(ScorerShapeError, match="list\\[PaperInfo\\]"):
        metric.score(papers="not-a-list")  # ty: ignore[missing-argument]  # false positive: union of bound/unbound score()


def test_relevance_metric_uses_reranker() -> None:
    papers = [_make_paper(f"Paper Number {i:03d} Title") for i in range(4)]
    fake = _FakeReranker([0.95, 0.92, 0.9, 1.0])
    metric = SearchResultRelevanceMetric(labeler=fake)
    query = ResearchQuery(text="neural network optimization methods")

    result = metric.score(query=query, papers=papers)  # ty: ignore[missing-argument]  # false positive: union of bound/unbound score()

    assert isinstance(result, ScoreResult)
    assert result.name == "search_result_relevance"
    assert result.value == 1.0
    assert result.metadata == {"passing": "True"}
    assert len(fake.calls) == 1
    assert fake.calls[0][0] == query
    assert fake.calls[0][1] == papers


def test_relevance_metric_empty_outputs_skips_reranker() -> None:
    fake = _FakeReranker([])
    metric = SearchResultRelevanceMetric(labeler=fake)
    query = ResearchQuery(text="neural network optimization methods")

    result = metric.score(query=query, papers=[])  # ty: ignore[missing-argument]  # false positive: union of bound/unbound score()

    assert isinstance(result, ScoreResult)
    assert result.name == "search_result_relevance"
    assert result.value == 0.0
    assert result.metadata == {"passing": "False"}
    assert fake.calls == []


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


def test_suggestion_length_metric_uses_domain_metric() -> None:
    suggestion = " ".join(f"w{i}" for i in range(SUGGESTION_MIN_WORDS))
    metric = SuggestionLengthMetric()
    result = metric.score(suggestion=suggestion)  # ty: ignore[missing-argument]  # false positive: union of bound/unbound score()

    assert isinstance(result, ScoreResult)
    assert result.name == "suggestion_length"
    assert result.value == 1.0
    assert result.metadata == {"passing": "True"}


def test_suggestion_length_metric_rejects_non_str() -> None:
    metric = SuggestionLengthMetric()
    with pytest.raises(ScorerShapeError, match="suggestion must be str"):
        metric.score(suggestion=123)  # ty: ignore[missing-argument]  # false positive: union of bound/unbound score()


def test_suggestion_quality_metric_uses_judge() -> None:
    papers = [_make_paper(_TITLE_A)]
    fake = _FakeQualityJudge(
        JudgeVerdict(score=1.0, failing=False, reason="Strong direction."),
    )
    metric = SuggestionQualityMetric(judge=fake)
    query = ResearchQuery(text="quantum error correction codes", domains=("physics",))
    suggestion = "Start with the survey paper and replicate the main experiment."

    result = metric.score(  # ty: ignore[missing-argument]  # false positive: union of bound/unbound score()
        query=query,
        papers=papers,
        suggestion=suggestion,
    )

    assert isinstance(result, ScoreResult)
    assert result.name == "suggestion_quality"
    assert result.value == 1.0
    assert result.metadata == {"passing": "True"}
    assert result.reason == "Strong direction."
    assert len(fake.calls) == 1
    assert "quantum error correction codes" in fake.calls[0]["task_input"]
    assert "physics" in fake.calls[0]["task_input"]
    assert fake.calls[0]["task_output"] == suggestion
    assert _TITLE_A in fake.calls[0]["task_context"]


def test_suggestion_quality_metric_maps_failing_verdict() -> None:
    fake = _FakeQualityJudge(
        JudgeVerdict(score=0.0, failing=True, reason="Fabricated finding."),
    )
    metric = SuggestionQualityMetric(judge=fake)
    result = metric.score(  # ty: ignore[missing-argument]  # false positive: union of bound/unbound score()
        query=ResearchQuery(text="quantum error correction codes"),
        papers=[_make_paper(_TITLE_A)],
        suggestion="invented next step",
    )
    assert isinstance(result, ScoreResult)
    assert result.value == 0.0
    assert result.metadata == {"passing": "False"}


def test_format_suggestion_judge_input_and_context() -> None:
    query = ResearchQuery(text="quantum error correction codes", domains=("physics",))
    papers = [_make_paper(_TITLE_A)]
    assert "Domains: physics" in format_suggestion_judge_input(query)
    assert _TITLE_A in format_suggestion_judge_context(papers)
    assert format_suggestion_judge_context([]) == "(no papers provided)"


def test_relevance_metric_low_relevance() -> None:
    papers = [_make_paper(f"Paper Number {i:03d} Title") for i in range(4)]
    metric = SearchResultRelevanceMetric(
        labeler=_FakeReranker([0.0, 0.1, 0.2, 0.29]),
    )
    result = metric.score(  # ty: ignore[missing-argument]  # false positive: union of bound/unbound score()
        query=ResearchQuery(text="climate modeling techniques today"),
        papers=papers,
    )
    assert isinstance(result, ScoreResult)
    assert result.value == 0.0
    assert result.metadata == {"passing": "False"}
    assert "Too many low relevance" in (result.reason or "")


def test_count_metric_accepts_dict_paper_payloads() -> None:
    papers = [
        _make_paper(f"Paper Number {i:03d}").model_dump(mode="json") for i in range(20)
    ]
    metric = SearchResultCountMetric()
    result = metric.score(papers=papers)  # ty: ignore[missing-argument]  # false positive: union of bound/unbound score()
    assert isinstance(result, ScoreResult)
    assert result.name == "search_result_count"
    assert result.value == 0.5
    assert result.metadata == {"passing": "True"}
