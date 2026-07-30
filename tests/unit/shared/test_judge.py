"""Unit tests for the shared rubric LLM judge."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from research_agent.search.rubrics import SUGGESTION_QUALITY_RUBRIC
from research_agent.shared.config.models import LMConfig
from research_agent.shared.judge import JudgeVerdict, RubricJudge


def test_judge_verdict_snaps_score_to_nearest_band() -> None:
    assert JudgeVerdict(score=0.2, failing=False, reason="low").score == 0.0
    assert JudgeVerdict(score=0.4, failing=False, reason="mid").score == 0.5
    assert JudgeVerdict(score=0.8, failing=False, reason="high").score == 1.0


@pytest.mark.asyncio
async def test_rubric_judge_maps_prediction_to_verdict() -> None:
    judge = RubricJudge(
        LMConfig(model="openai/test-judge"),
        SUGGESTION_QUALITY_RUBRIC,
    )
    fake_pred = SimpleNamespace(score=0.5, failing=False, reason="Mixed quality.")
    with patch.object(
        judge._predict,  # test doubles the DSPy predictor
        "aforward",
        new=AsyncMock(return_value=fake_pred),
    ) as aforward:
        verdict = await judge.judge(
            task_input="query text",
            task_output="suggestion text",
            task_context="paper context",
        )

    aforward.assert_awaited_once()
    assert aforward.await_args is not None
    call_kwargs = aforward.await_args.kwargs
    assert call_kwargs["task_input"] == "query text"
    assert call_kwargs["task_output"] == "suggestion text"
    assert call_kwargs["task_context"] == "paper context"
    assert "1. " in call_kwargs["rubric_criteria"]
    assert call_kwargs["rubric_scoring"]
    assert "1. " in call_kwargs["rubric_fail_conditions"]
    assert verdict == JudgeVerdict(
        score=0.5,
        failing=False,
        reason="Mixed quality.",
    )
