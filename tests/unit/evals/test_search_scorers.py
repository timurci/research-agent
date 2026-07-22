"""Unit tests for search MLflow scorer adapters.

Exercises agent-aligned scoring and reranker-backed relevance. No
network, no ``mlflow.genai.evaluate`` run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from mlflow.entities import Feedback
from pydantic import HttpUrl

from evals.search.scorers import (
    ScorerShapeError,
    make_search_result_relevance_scorer,
    relevance_metrics_from_ranking,
    research_query_from_inputs,
    search_result_count_scorer,
    search_result_non_duplicate_scorer,
)
from research_agent.search.metrics import RelevanceMetric
from research_agent.search.models import PaperInfo, ResearchQuery

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


def test_research_query_from_inputs_accepts_model() -> None:
    query = ResearchQuery(text="quantum error correction")
    assert research_query_from_inputs(query) == query


def test_research_query_from_inputs_accepts_query_key() -> None:
    query = ResearchQuery(text="quantum error correction")
    assert research_query_from_inputs({"query": query}) == query


def test_research_query_from_inputs_accepts_dict_payload() -> None:
    result = research_query_from_inputs({"query": {"text": "quantum error correction"}})
    assert result == ResearchQuery(text="quantum error correction")


def test_research_query_from_inputs_rejects_bad_shape() -> None:
    with pytest.raises(ScorerShapeError, match="ResearchQuery or dict"):
        research_query_from_inputs(42)


def test_research_query_from_inputs_requires_query_key() -> None:
    with pytest.raises(ScorerShapeError, match="query"):
        research_query_from_inputs({"text": "quantum error correction codes"})


def test_count_scorer_below_pass_floor() -> None:
    outputs = [_make_paper(_TITLE_A), _make_paper(_TITLE_B)]
    feedback = search_result_count_scorer(outputs=outputs)

    assert isinstance(feedback, Feedback)
    assert feedback.name == "search_result_count"
    assert feedback.value is False
    assert feedback.metadata == {"score": str(0.95 * 2 / 25)}
    assert "Returned 2 results" in (feedback.rationale or "")
    assert feedback.source is not None
    assert feedback.source.source_id == "research_agent.search.metrics"


def test_count_scorer_pass_with_partial_score() -> None:
    outputs = [_make_paper(f"Paper Number {i:03d}") for i in range(10)]
    feedback = search_result_count_scorer(outputs=outputs)

    assert isinstance(feedback, Feedback)
    assert feedback.value is True
    assert feedback.metadata == {"score": str(0.95 * 10 / 25)}
    assert "Returned 10 results" in (feedback.rationale or "")


def test_count_scorer_meets_target() -> None:
    outputs = [_make_paper(f"Paper Number {i:03d}") for i in range(25)]
    feedback = search_result_count_scorer(outputs=outputs)

    assert isinstance(feedback, Feedback)
    assert feedback.value is True
    assert feedback.metadata == {"score": "0.95"}
    assert "target 25 met" in (feedback.rationale or "")


def test_non_duplicate_scorer_passes_unique_titles() -> None:
    outputs = [_make_paper(_TITLE_A), _make_paper(_TITLE_B)]
    feedback = search_result_non_duplicate_scorer(outputs=outputs)

    assert isinstance(feedback, Feedback)
    assert feedback.name == "search_result_non_duplicate"
    assert feedback.value is True
    assert feedback.metadata == {"score": "1.0"}
    assert "No duplicate papers found." in (feedback.rationale or "")
    assert feedback.source is not None
    assert feedback.source.source_id == "research_agent.search.metrics"


def test_non_duplicate_scorer_fails_on_duplicate_titles() -> None:
    outputs = [_make_paper(_TITLE_A), _make_paper(_TITLE_B), _make_paper(_TITLE_A)]
    feedback = search_result_non_duplicate_scorer(outputs=outputs)

    assert isinstance(feedback, Feedback)
    assert feedback.value is False
    assert feedback.metadata == {"score": "0.0"}
    assert _TITLE_A in (feedback.rationale or "")


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


def test_require_paper_list_rejects_non_list_with_outputs_error() -> None:
    with pytest.raises(ScorerShapeError, match="list\\[PaperInfo\\]"):
        search_result_non_duplicate_scorer(outputs="not-a-list")


def test_search_result_relevance_scorer_uses_reranker() -> None:
    papers = [_make_paper(f"Paper Number {i:03d} Title") for i in range(4)]
    fake = _FakeReranker([0.7, 0.8, 0.9, 1.0])
    scorer = make_search_result_relevance_scorer(fake)
    query = ResearchQuery(text="neural network optimization methods")

    feedback = scorer(inputs={"query": query}, outputs=papers)

    assert isinstance(feedback, Feedback)
    assert feedback.name == "search_result_relevance"
    assert feedback.value is True
    assert feedback.metadata == {"score": "1.0"}
    assert len(fake.calls) == 1
    assert fake.calls[0][0] == query
    assert fake.calls[0][1] == papers


def test_search_result_relevance_scorer_empty_outputs_skips_reranker() -> None:
    fake = _FakeReranker([])
    scorer = make_search_result_relevance_scorer(fake)
    query = ResearchQuery(text="neural network optimization methods")

    feedback = scorer(inputs={"query": query}, outputs=[])

    assert isinstance(feedback, Feedback)
    assert feedback.name == "search_result_relevance"
    assert feedback.value is False
    assert feedback.metadata == {"score": "0.0"}
    assert "Empty input" in (feedback.rationale or "")
    assert fake.calls == []


def test_search_result_relevance_scorer_low_relevance() -> None:
    papers = [_make_paper(f"Paper Number {i:03d} Title") for i in range(4)]
    scorer = make_search_result_relevance_scorer(
        _FakeReranker([0.0, 0.1, 0.2, 0.29]),
    )
    feedback = scorer(
        inputs={"query": ResearchQuery(text="climate modeling techniques today")},
        outputs=papers,
    )
    assert isinstance(feedback, Feedback)
    assert feedback.value is False
    assert feedback.metadata == {"score": "0.0"}
    assert "Too many low relevance" in (feedback.rationale or "")


def test_mlflow_scorer_wrappers_are_callable() -> None:
    assert callable(search_result_count_scorer)
    assert search_result_count_scorer.name == "search_result_count"
    assert callable(search_result_non_duplicate_scorer)
    assert search_result_non_duplicate_scorer.name == "search_result_non_duplicate"


def test_non_duplicate_scorer_accepts_dict_paper_payloads() -> None:
    paper = _make_paper(_TITLE_A)
    result = search_result_non_duplicate_scorer(outputs=[paper.model_dump(mode="json")])
    assert isinstance(result, Feedback)
    assert result.name == "search_result_non_duplicate"
    assert result.value is True
