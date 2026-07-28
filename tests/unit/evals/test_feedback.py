"""Unit tests for domain-to-Opik ScoreResult mapping."""

from __future__ import annotations

from opik.evaluation.metrics.score_result import ScoreResult

from evals.feedback import evaluation_score_to_score_result
from research_agent.shared.metric import EvaluationScore


def test_evaluation_score_to_score_result_maps_passing() -> None:
    score = EvaluationScore(
        passing=True,
        reason="Results are mostly high relevance.",
        score=0.875,
    )
    result = evaluation_score_to_score_result(score, name="search_result_relevance")

    assert isinstance(result, ScoreResult)
    assert result.name == "search_result_relevance"
    assert result.value == 0.875
    assert result.reason == "Results are mostly high relevance."
    assert result.metadata == {"passing": "True"}


def test_evaluation_score_to_score_result_maps_failing() -> None:
    score = EvaluationScore(passing=False, reason="Empty input", score=0.0)
    result = evaluation_score_to_score_result(score, name="search_result_count")

    assert isinstance(result, ScoreResult)
    assert result.name == "search_result_count"
    assert result.value == 0.0
    assert result.reason == "Empty input"
    assert result.metadata == {"passing": "False"}


def test_evaluation_score_to_score_result_preserves_float_score() -> None:
    passing = evaluation_score_to_score_result(
        EvaluationScore(passing=True, reason="ok", score=0.5),
        name="test",
    )
    failing = evaluation_score_to_score_result(
        EvaluationScore(passing=False, reason="bad", score=0.9),
        name="test",
    )

    assert passing.value == 0.5
    assert passing.metadata == {"passing": "True"}
    assert failing.value == 0.9
    assert failing.metadata == {"passing": "False"}
