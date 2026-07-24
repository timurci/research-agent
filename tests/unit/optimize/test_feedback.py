"""Unit tests for domain-to-GEPA feedback mapping."""

from __future__ import annotations

from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback

from optimize.feedback import evaluation_score_to_score_with_feedback
from research_agent.shared.metric import EvaluationScore


def test_evaluation_score_to_score_with_feedback_maps_fields() -> None:
    score = EvaluationScore(
        passing=True,
        reason="Results are mostly high relevance.",
        score=0.875,
    )
    result = evaluation_score_to_score_with_feedback(
        score,
        name="search_result_relevance",
    )

    assert isinstance(result, ScoreWithFeedback)
    assert result.score == 0.875
    assert result.feedback == (
        "search_result_relevance (pass=True): Results are mostly high relevance."
    )


def test_evaluation_score_to_score_with_feedback_default_name() -> None:
    score = EvaluationScore(passing=False, reason="Empty input", score=0.0)
    result = evaluation_score_to_score_with_feedback(score)

    assert result.score == 0.0
    assert result.feedback == "metric (pass=False): Empty input"
