"""Shared rubric model for subjective LLM-judge metrics.

Layer: Domain.

LLM judges cover qualities that code metrics cannot score. A rubric
supplies: a quality checklist, language-guided scoring, and explicit
fail conditions. The judge result maps to ``EvaluationScore``
(``passing``, ``reason``, ``score``).

Pass/fail is decided only by ``fail_conditions``, not by comparing
``score`` to a numeric threshold. ``score`` is a separate continuous
signal guided by natural-language bands in ``scoring``.

Good quality criteria are specific. Bad: "Was the response helpful?"
Good: several concrete checks (grounded claims, specific next step,
professional tone).

Optional ``examples`` are labeled cases for few-shot calibration. Task
payloads are plain text so any slice can format domain objects into the
judge prompt. An empty ``examples`` tuple is valid when no labeled set
exists yet.
"""

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RubricExample(BaseModel):
    """One human-labeled case for few-shot LLM judging.

    Mirrors the judge metric result: continuous ``score``, binary
    ``passing``, and ``reason`` as feedback. Text fields stay
    task-agnostic; a harness formats domain objects into ``input`` /
    ``output`` / ``context``.
    """

    model_config = ConfigDict(frozen=True)

    input: str = Field(..., min_length=1)
    output: str = Field(..., min_length=1)
    context: str = ""
    score: float = Field(..., ge=0.0, le=1.0)
    passing: bool
    reason: str = Field(..., min_length=1)


class Rubric(BaseModel):
    """Criteria, scoring guide, and fail modes for one LLM-judge metric.

    ``criteria`` — quality checklist the judge assesses.

    ``scoring`` — natural-language instructions for assigning
    ``EvaluationScore.score`` (prefer discrete language bands such as
    low / mixed / high mapped to 0.0 / 0.5 / 1.0; do not ask the model
    to invent its own arithmetic).

    ``fail_conditions`` — hard failure modes. If any apply, ``passing``
    is false, independent of the continuous score.

    ``examples`` may be empty; when present they label score, pass/fail,
    and reason for the same rubric.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., min_length=5)
    criteria: tuple[str, ...] = Field(..., min_length=1)
    scoring: str = Field(..., min_length=10)
    fail_conditions: tuple[str, ...] = Field(..., min_length=1)
    examples: tuple[RubricExample, ...] = ()

    @field_validator("criteria", "fail_conditions")
    @classmethod
    def _items_non_empty(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item.strip() for item in value):
            msg = "each item must be non-empty"
            raise ValueError(msg)
        return value

    @field_validator("scoring")
    @classmethod
    def _scoring_non_blank(cls, value: str) -> str:
        if not value.strip():
            msg = "scoring guide must be non-empty"
            raise ValueError(msg)
        return value

    def checklist(self) -> str:
        """Numbered quality criteria text suitable for a judge prompt."""
        return "\n".join(
            f"{index}. {item}" for index, item in enumerate(self.criteria, start=1)
        )

    def fail_checklist(self) -> str:
        """Numbered fail-condition text suitable for a judge prompt."""
        return "\n".join(
            f"{index}. {item}"
            for index, item in enumerate(self.fail_conditions, start=1)
        )
