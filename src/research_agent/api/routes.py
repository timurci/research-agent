"""HTTP route handlers.

Layer: Presentation.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, status

from research_agent.api.deps import PaperSearchFacade, get_paper_search_app
from research_agent.api.schemas import FeedbackBody, HealthResponse, SearchResponse
from research_agent.search.models import (  # noqa: TC001  # FastAPI resolves body model at runtime
    ResearchQuery,
)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness probe for the HTTP service."""
    return HealthResponse(status="ok")


@router.post("/search", response_model=SearchResponse)
async def search(
    query: ResearchQuery,
    app: Annotated[PaperSearchFacade, Depends(get_paper_search_app)],
) -> SearchResponse:
    """Run paper search and return papers, suggestion, and trace id."""
    run = await app.search(query)
    return SearchResponse(
        papers=run.papers,
        suggestion=run.suggestion,
        trace_id=run.trace_id,
    )


@router.post("/feedback", status_code=status.HTTP_204_NO_CONTENT)
async def feedback(
    body: FeedbackBody,
    app: Annotated[PaperSearchFacade, Depends(get_paper_search_app)],
) -> None:
    """Attach thumbs feedback to a completed search trace."""
    await app.record_feedback(
        body.trace_id,
        useful=body.useful,
        comment=body.comment,
    )
