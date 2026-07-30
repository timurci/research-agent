"""Runtime Opik observability helpers.

Layer: Infrastructure.

Tracing and user-feedback logging for the composition root.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import dspy
import opik
from opik.integrations.dspy import OpikCallback

if TYPE_CHECKING:
    from opik.types import BatchFeedbackScoreDict

__all__ = [
    "USER_USEFUL_SCORE_NAME",
    "configure_dspy_opik_callback",
    "flush_opik_client",
    "record_user_feedback",
    "user_useful_feedback_score",
]

USER_USEFUL_SCORE_NAME: str = "user_useful"


def configure_dspy_opik_callback(*, project_name: str | None = None) -> None:
    """Register Opik as the process-wide DSPy callback for nested spans.

    Replaces the entire DSPy callbacks list (does not merge with any
    callbacks already configured). Intended once at process start.
    Callers that already configure DSPy should set
    ``configure_observability=False`` on the composition root and
    register callbacks themselves.

    Args:
        project_name: Optional Opik project override for DSPy spans.
    """
    dspy.configure(callbacks=[OpikCallback(project_name=project_name)])


def user_useful_feedback_score(
    trace_id: str,
    *,
    useful: bool,
    comment: str | None = None,
    project_name: str | None = None,
) -> BatchFeedbackScoreDict:
    """Build an Opik batch feedback score for thumbs-up/down feedback.

    Args:
        trace_id: Opik trace id from a completed search run.
        useful: Whether the user found the results useful.
        comment: Optional free-text reason.
        project_name: Optional Opik project for the score batch entry.

    Returns:
        Score dict suitable for ``Opik.log_traces_feedback_scores``.
    """
    score: BatchFeedbackScoreDict = {
        "id": trace_id,
        "name": USER_USEFUL_SCORE_NAME,
        "value": 1.0 if useful else 0.0,
    }
    if comment is not None:
        stripped = comment.strip()
        if stripped:
            score["reason"] = stripped
    if project_name is not None:
        score["project_name"] = project_name
    return score


def flush_opik_client(client: opik.Opik | None = None) -> None:
    """Flush the Opik client streamer so queued messages are sent.

    Args:
        client: Opik client; uses the process global client when omitted.
    """
    active = client if client is not None else opik.get_global_client()
    active.flush()


def record_user_feedback(
    trace_id: str,
    *,
    useful: bool,
    comment: str | None = None,
    client: opik.Opik | None = None,
    project_name: str | None = None,
) -> None:
    """Log thumbs-up/down user feedback against an Opik trace.

    Flushes the client after enqueueing scores so short-lived processes
    deliver feedback before exit.

    Args:
        trace_id: Opik trace id from a completed search run.
        useful: Whether the user found the results useful.
        comment: Optional free-text reason.
        client: Opik client; uses the process global client when omitted.
        project_name: Optional Opik project for the score batch entry.
    """
    active = client if client is not None else opik.get_global_client()
    score = user_useful_feedback_score(
        trace_id,
        useful=useful,
        comment=comment,
        project_name=project_name,
    )
    active.log_traces_feedback_scores(scores=[score])
    active.flush()
