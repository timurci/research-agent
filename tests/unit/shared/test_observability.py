"""Unit tests for runtime Opik observability helpers."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from research_agent.shared.observability import (
    USER_USEFUL_SCORE_NAME,
    flush_opik_client,
    record_user_feedback,
    user_useful_feedback_score,
)


def test_user_useful_feedback_score_true_with_comment() -> None:
    score = user_useful_feedback_score(
        "trace-1",
        useful=True,
        comment="good set",
    )
    assert score == {
        "id": "trace-1",
        "name": USER_USEFUL_SCORE_NAME,
        "value": 1.0,
        "reason": "good set",
    }


def test_user_useful_feedback_score_false_without_comment() -> None:
    score = user_useful_feedback_score("trace-2", useful=False)
    assert score == {
        "id": "trace-2",
        "name": USER_USEFUL_SCORE_NAME,
        "value": 0.0,
    }
    assert "reason" not in score


def test_user_useful_feedback_score_omits_blank_comment() -> None:
    score = user_useful_feedback_score("trace-blank", useful=True, comment="  ")
    assert "reason" not in score


def test_user_useful_feedback_score_strips_comment() -> None:
    score = user_useful_feedback_score(
        "trace-strip",
        useful=True,
        comment="  good set  ",
    )
    assert score["reason"] == "good set"


def test_user_useful_feedback_score_includes_project_name() -> None:
    score = user_useful_feedback_score(
        "trace-3",
        useful=True,
        project_name="research-agent",
    )
    assert score["project_name"] == "research-agent"


def test_record_user_feedback_logs_score_via_client() -> None:
    client = MagicMock()
    record_user_feedback(
        "trace-9",
        useful=True,
        comment="nice",
        client=client,
        project_name="proj",
    )
    client.log_traces_feedback_scores.assert_called_once()
    scores: list[dict[str, Any]] = client.log_traces_feedback_scores.call_args.kwargs[
        "scores"
    ]
    assert scores == [
        {
            "id": "trace-9",
            "name": USER_USEFUL_SCORE_NAME,
            "value": 1.0,
            "reason": "nice",
            "project_name": "proj",
        }
    ]
    client.flush.assert_called_once_with()


def test_record_user_feedback_uses_global_client_when_omitted() -> None:
    client = MagicMock()
    with patch(
        "research_agent.shared.observability.opik.get_global_client",
        return_value=client,
    ) as get_client:
        record_user_feedback("trace-x", useful=False)
    get_client.assert_called_once_with()
    client.log_traces_feedback_scores.assert_called_once()
    client.flush.assert_called_once_with()


def test_flush_opik_client_uses_given_client() -> None:
    client = MagicMock()
    flush_opik_client(client)
    client.flush.assert_called_once_with()


def test_flush_opik_client_uses_global_when_omitted() -> None:
    client = MagicMock()
    with patch(
        "research_agent.shared.observability.opik.get_global_client",
        return_value=client,
    ) as get_client:
        flush_opik_client()
    get_client.assert_called_once_with()
    client.flush.assert_called_once_with()
