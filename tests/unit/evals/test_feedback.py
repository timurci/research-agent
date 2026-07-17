"""Unit tests for domain-to-MLflow feedback mapping."""

from __future__ import annotations

from mlflow.entities import AssessmentSourceType
from mlflow.genai.evaluation.entities import _assertion_outcome

from evals.feedback import code_assessment_source, evaluation_score_to_feedback
from research_agent.shared.metric import EvaluationScore


def test_code_assessment_source_uses_module_path() -> None:
    source = code_assessment_source("research_agent.search.metrics")

    assert source.source_type == AssessmentSourceType.CODE
    assert source.source_id == "research_agent.search.metrics"


def test_evaluation_score_to_feedback_maps_fields() -> None:
    source = code_assessment_source("research_agent.search.metrics")
    score = EvaluationScore(
        passing=True,
        reason="Results are mostly high relevance.",
        score=0.875,
    )
    feedback = evaluation_score_to_feedback(
        score,
        source=source,
        name="search_result_relevance",
    )

    assert feedback.name == "search_result_relevance"
    assert feedback.value is True
    assert feedback.rationale == "Results are mostly high relevance."
    assert feedback.metadata == {"score": "0.875"}
    assert feedback.source is source


def test_evaluation_score_to_feedback_records_failing() -> None:
    source = code_assessment_source("research_agent.search.metrics")
    score = EvaluationScore(passing=False, reason="Empty input", score=0.0)
    feedback = evaluation_score_to_feedback(score, source=source)

    assert feedback.name == "feedback"
    assert feedback.value is False
    assert feedback.metadata == {"score": "0.0"}
    assert feedback.source is not None
    assert feedback.source.source_id == "research_agent.search.metrics"


def test_evaluation_score_bool_value_matches_mlflow_default_assertion() -> None:
    """Bool value is accepted by MLflow's default assertion path without pass_if."""
    source = code_assessment_source("research_agent.search.metrics")
    passing = evaluation_score_to_feedback(
        EvaluationScore(passing=True, reason="ok", score=0.5),
        source=source,
        name="search_result_relevance",
    )
    failing = evaluation_score_to_feedback(
        EvaluationScore(passing=False, reason="bad", score=0.9),
        source=source,
        name="search_result_relevance",
    )

    assert _assertion_outcome(passing.value, None, passing.rationale, None)[0] is True
    assert _assertion_outcome(failing.value, None, failing.rationale, None)[0] is False
