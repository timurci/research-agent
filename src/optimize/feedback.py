"""Map domain evaluation scores to GEPA score-with-feedback.

Layer: Infrastructure (optimization harness).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback

if TYPE_CHECKING:
    from research_agent.shared.metric import EvaluationScore


def evaluation_score_to_score_with_feedback(
    score: EvaluationScore,
    *,
    name: str | None = None,
) -> ScoreWithFeedback:
    """Convert a domain ``EvaluationScore`` to GEPA ``ScoreWithFeedback``.

    The continuous domain score is the GEPA float objective. Pass/fail and
    rationale become the reflection feedback string (optionally named).

    Args:
        score: Domain evaluation score.
        name: Optional metric name prefix in the feedback text.

    Returns:
        GEPA prediction with ``score`` and ``feedback`` fields.
    """
    label = name or "metric"
    feedback = f"{label} (pass={score.passing}): {score.reason}"
    return ScoreWithFeedback(score=score.score, feedback=feedback)
