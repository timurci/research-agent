"""LLM rubric judge for subjective evaluation metrics.

Layer: Infrastructure.

Applies a domain ``Rubric`` via a held-out language model and maps the
verdict onto a continuous score, binary pass/fail, and reason. Callers
own which rubric and LM role to use (for example ``llm-judge``) and how
to format task input/output/context text.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import dspy
from pydantic import BaseModel, ConfigDict, Field, field_validator

from research_agent.shared.dspy import dspy_lm

if TYPE_CHECKING:
    from research_agent.shared.config.models import LMConfig
    from research_agent.shared.rubric import Rubric

_ALLOWED_SCORES: frozenset[float] = frozenset({0.0, 0.5, 1.0})


class JudgeVerdict(BaseModel):
    """Structured output of a single rubric-judge call."""

    model_config = ConfigDict(frozen=True)

    score: float = Field(..., ge=0.0, le=1.0)
    failing: bool
    reason: str = Field(..., min_length=1)

    @field_validator("score")
    @classmethod
    def _score_is_band(cls, value: float) -> float:
        return min(_ALLOWED_SCORES, key=lambda band: abs(band - value))


class _RubricJudgeSignature(dspy.Signature):
    """Score task output against the given rubric.

    Read the criteria, scoring guide, and fail conditions carefully.
    Assign score as exactly one of 0.0, 0.5, or 1.0 per the scoring guide.
    Set failing to true when any fail condition applies, independent of
    the continuous score. Explain the verdict briefly in reason.
    """

    task_input: str = dspy.InputField(desc="Primary task input text")
    task_output: str = dspy.InputField(desc="Model or agent output to judge")
    task_context: str = dspy.InputField(
        desc="Supporting context (papers, evidence, constraints)",
    )
    rubric_criteria: str = dspy.InputField(desc="Numbered quality criteria")
    rubric_scoring: str = dspy.InputField(desc="Scoring guide for continuous score")
    rubric_fail_conditions: str = dspy.InputField(
        desc="Numbered hard fail conditions",
    )
    score: float = dspy.OutputField(desc="Exactly 0.0, 0.5, or 1.0")
    failing: bool = dspy.OutputField(
        desc="True when any fail condition applies",
    )
    reason: str = dspy.OutputField(desc="Brief rationale for score and pass/fail")


class RubricJudge:
    """Held-out LM judge that scores free-text tasks with a ``Rubric``."""

    def __init__(self, lm_config: LMConfig, rubric: Rubric) -> None:
        """Build a judge bound to one LM role and one rubric.

        Args:
            lm_config: Language model settings (typically ``llm-judge``).
            rubric: Domain rubric defining criteria, scoring, and fails.
        """
        self._lm = dspy_lm(lm_config)
        self._rubric = rubric
        self._predict = dspy.Predict(_RubricJudgeSignature)

    async def judge(
        self,
        *,
        task_input: str,
        task_output: str,
        task_context: str = "",
    ) -> JudgeVerdict:
        """Judge *task_output* against the bound rubric.

        Args:
            task_input: Primary task input text (e.g. research query).
            task_output: Candidate output to score.
            task_context: Supporting evidence (e.g. paper titles/abstracts).

        Returns:
            Structured verdict with snapped discrete score, fail flag, and
            reason text.
        """
        with dspy.settings.context(lm=self._lm):
            prediction = await self._predict.aforward(
                task_input=task_input,
                task_output=task_output,
                task_context=task_context,
                rubric_criteria=self._rubric.checklist(),
                rubric_scoring=self._rubric.scoring,
                rubric_fail_conditions=self._rubric.fail_checklist(),
            )
        return JudgeVerdict(
            score=float(prediction.score),
            failing=bool(prediction.failing),
            reason=str(prediction.reason),
        )
