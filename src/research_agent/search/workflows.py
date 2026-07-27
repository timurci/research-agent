"""Application workflows for the search slice.

Layer: Application.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from research_agent.search.models import PaperInfo, ResearchQuery
    from research_agent.shared.agent import Agent

SUGGESTION_TOP_N: int = 10


class PaperSearchWorkflow:
    """Search papers and suggest a manual research direction."""

    def __init__(
        self,
        search_agent: Agent[ResearchQuery, list[PaperInfo]],
        reranker: Agent[tuple[ResearchQuery, list[PaperInfo]], list[PaperInfo]],
        suggestion_generator: Agent[tuple[ResearchQuery, list[PaperInfo]], str],
    ) -> None:
        """Initialize the workflow with the three agent ports.

        Args:
            search_agent: Port that turns a query into a list of papers.
            reranker: Port that reorders a (query, papers) pair into a
                new list ordered by relevance.
            suggestion_generator: Port that turns a (query, top papers)
                pair into a free-text research-direction suggestion.
        """
        self._search = search_agent
        self._rerank = reranker
        self._suggest = suggestion_generator

    async def __call__(
        self,
        query: ResearchQuery,
    ) -> tuple[list[PaperInfo], str]:
        """Run search, reranking, and suggestion for the given query.

        Args:
            query: The research question to search for.

        Returns:
            A tuple of the reranked papers and a free-text suggestion for
            manual research. The suggestion is empty when the search agent
            returns no results.
        """
        results = await self._search(query)
        if not results:
            return results, ""
        reranked = await self._rerank((query, results))
        top_papers = reranked[:SUGGESTION_TOP_N]
        suggestion = await self._suggest((query, top_papers))
        return reranked, suggestion
