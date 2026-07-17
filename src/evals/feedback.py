"""Map domain evaluation scores to MLflow Feedback.

Layer: Infrastructure (evaluation harness).
"""

from typing import TYPE_CHECKING

from mlflow.entities import AssessmentSource, AssessmentSourceType, Feedback

if TYPE_CHECKING:
    from research_agent.shared.metric import EvaluationScore


def code_assessment_source(source_id: str) -> AssessmentSource:
    """Build a CODE ``AssessmentSource`` for a domain metrics module.

    ``source_id`` should identify the module that defines the metric
    (e.g. ``research_agent.search.metrics``), not a generic domain label.

    Args:
        source_id: Dotted module path of the metric implementation.

    Returns:
        MLflow assessment source with type CODE and the given id.
    """
    return AssessmentSource(
        source_type=AssessmentSourceType.CODE,
        source_id=source_id,
    )


def evaluation_score_to_feedback(
    score: EvaluationScore,
    *,
    source: AssessmentSource,
    name: str | None = None,
) -> Feedback:
    """Convert a domain ``EvaluationScore`` to an MLflow ``Feedback``.

    Domain pass/fail is the feedback value so MLflow's default assertion
    path (``EvaluationResult.passed``) treats the row correctly without
    ``pass_if``. The continuous domain score is stored in metadata under
    ``score`` for drill-down; suite means therefore reflect pass rate.

    Args:
        score: Domain evaluation score.
        source: Assessment source for the metric module that produced
            *score* (use ``code_assessment_source``).
        name: Optional metric name for multi-feedback scorers.

    Returns:
        MLflow feedback with bool value, rationale, source, and score
        metadata.
    """
    return Feedback(
        name=name or "feedback",
        value=score.passing,
        rationale=score.reason,
        source=source,
        metadata={"score": str(score.score)},
    )
