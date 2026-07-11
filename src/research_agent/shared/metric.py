"""Shared AI evaluation models.

Layer: Domain.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EvaluationScore(BaseModel):
    """Evaluation score."""

    model_config = ConfigDict(frozen=True)

    passing: bool
    reason: str
    score: float = Field(..., ge=0.0, le=1.0)


class ToolCallObservation[OutputT](BaseModel):
    """Result of a tool call."""

    model_config = ConfigDict(frozen=True)

    tool_name: str
    call_args: dict[str, Any]
    observation: OutputT | None
