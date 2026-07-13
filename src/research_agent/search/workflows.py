"""Application workflows for the search slice.

Layer: Application.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from research_agent.search.models import PaperInfo, ResearchQuery
    from research_agent.shared.agent import Agent


class PaperSearchWorkflow:
    """Search papers in the literature for a research query."""

    def __init__(
        self,
        search_agent: Agent[ResearchQuery, list[PaperInfo]],
        reranker: Agent[tuple[ResearchQuery, list[PaperInfo]], list[PaperInfo]],
    ) -> None:
        """Initialize the workflow with the two agent ports.

        Args:
            search_agent: Port that turns a query into a list of papers.
            reranker: Port that reorders a (query, papers) pair into a
                new list ordered by relevance.
        """
        self._search = search_agent
        self._rerank = reranker

    async def __call__(self, query: ResearchQuery) -> list[PaperInfo]:
        """Run search and reranking for the given query.

        Args:
            query: The research question to search for.

        Returns:
            The papers for *query*, reordered by relevance.
            Empty when the search agent returns no results.
        """
        results = await self._search(query)
        if not results:
            return results
        return await self._rerank((query, results))
