"""Application workflows for the search slice.

Layer: Application.

Workflows are thin orchestrators that compose ``Agent`` ports. They
hold no business logic, never import DSPy or LiteLLM, and never depend
on a specific adapter — production wires concrete adapters at the
composition root, tests wire in-memory fakes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from research_agent.search.models import ResearchQuery, SearchResult
    from research_agent.shared.agent import Agent


class PaperSearchWorkflow:
    """Search papers in the literature for a research query."""

    def __init__(
        self,
        search_agent: Agent[ResearchQuery, list[SearchResult]],
        reranker: Agent[tuple[ResearchQuery, list[SearchResult]], list[SearchResult]],
    ) -> None:
        """Initialize the workflow with the two agent ports.

        Args:
            search_agent: Port that turns a query into a list of search results.
            reranker: Port that reorders a (query, results) pair into a
                new list ordered by relevance.
        """
        self._search = search_agent
        self._rerank = reranker

    async def __call__(self, query: ResearchQuery) -> list[SearchResult]:
        """Search for papers in the literature for the given query.

        Args:
            query: The research question to search for.

        Returns:
            The search results for *query*, reordered by relevance.
            Empty when the search agent returns no results.
        """
        results = await self._search(query)
        if not results:
            return results
        return await self._rerank((query, results))
