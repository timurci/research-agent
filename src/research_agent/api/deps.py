"""FastAPI dependencies for the presentation layer.

Layer: Presentation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from fastapi import Request  # noqa: TC002  # FastAPI injects Request at runtime

if TYPE_CHECKING:
    from research_agent.app import SearchRun
    from research_agent.search.models import ResearchQuery


class PaperSearchFacade(Protocol):
    """Structural shape of the runtime facade used by HTTP handlers."""

    async def search(self, query: ResearchQuery) -> SearchRun:
        """Run paper search under an Opik root trace."""
        ...

    async def record_feedback(
        self,
        trace_id: str,
        *,
        useful: bool,
        comment: str | None = None,
    ) -> None:
        """Attach thumbs feedback to a completed search trace."""
        ...


def get_paper_search_app(request: Request) -> PaperSearchFacade:
    """Return the process-scoped paper-search facade from app state."""
    return request.app.state.paper_search_app
