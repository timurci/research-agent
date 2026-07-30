"""HTTP request and response models for the API.

Layer: Presentation.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from research_agent.search.models import (  # noqa: TC001  # Pydantic field type used at model build
    PaperInfo,
)


class SearchResponse(BaseModel):
    """HTTP body returned by ``POST /search``."""

    model_config = ConfigDict(frozen=True)

    papers: list[PaperInfo]
    suggestion: str
    trace_id: str


class FeedbackBody(BaseModel):
    """HTTP body for ``POST /feedback``."""

    model_config = ConfigDict(frozen=True)

    trace_id: str = Field(..., min_length=1)
    useful: bool
    comment: str | None = None


class HealthResponse(BaseModel):
    """HTTP body returned by ``GET /health``."""

    model_config = ConfigDict(frozen=True)

    status: str
