"""Map domain evaluation scores to Opik ScoreResult.

Layer: Infrastructure (evaluation harness).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from opik.evaluation.metrics.score_result import ScoreResult

if TYPE_CHECKING:
    from research_agent.shared.metric import EvaluationScore


def evaluation_score_to_score_result(
    score: EvaluationScore,
    *,
    name: str,
) -> ScoreResult:
    """Convert a domain ``EvaluationScore`` to an Opik ``ScoreResult``.

    The continuous domain score becomes the ``value`` so Opik aggregates
    mean score. Domain pass/fail is preserved in metadata for drill-down.

    Args:
        score: Domain evaluation score.
        name: Metric name for the score result.

    Returns:
        Opik score result with float value, reason, and passing metadata.
    """
    return ScoreResult(
        name=name,
        value=score.score,
        reason=score.reason,
        metadata={"passing": str(score.passing)},
    )
