"""DSPy signature for search.

Layer: Infrastructure.
"""

from typing import TYPE_CHECKING, TypedDict

import dspy
import litellm

from research_agent.search.tools import LiteratureSearch
from research_agent.shared.agent import Agent

from .models import ResearchQuery, SearchResult

if TYPE_CHECKING:
    from research_agent.shared.agent import LMConfig


class _RelevanceScore(TypedDict):
    index: int
    relevance_score: float


class SearchAgentSignature(dspy.Signature):
    """Perform literature search for a given research query."""

    research_query: ResearchQuery = dspy.InputField()
    search_results: list[SearchResult] = dspy.OutputField()


def get_search_agent(
    *, pubmed_api_key: str | None = None
) -> Agent[ResearchQuery, list[SearchResult]]:
    """Create a search agent instance.

    Args:
        pubmed_api_key: PubMed API key.
    """
    return dspy.ReAct(
        SearchAgentSignature,
        tools=[LiteratureSearch(pubmed_api_key=pubmed_api_key)],
    )


class SearchAgent(Agent[ResearchQuery, list[SearchResult]]):
    """Search agent that uses a DSPy ReAct program to search for research results."""

    def __init__(
        self,
        lm_config: LMConfig,
        pubmed_api_key: str | None = None,
    ) -> None:
        """Initialize the search agent.

        Args:
            lm_config: The language model to use.
            pubmed_api_key: PubMed API key.
        """
        self._lm = dspy.LM(
            model=lm_config.model,
            api_key=lm_config.api_key,
            api_base=str(lm_config.base_url) if lm_config.base_url else None,
        )
        self._program = dspy.ReAct(
            SearchAgentSignature,
            tools=[LiteratureSearch(pubmed_api_key=pubmed_api_key)],
        )

    async def __call__(self, data: ResearchQuery) -> list[SearchResult]:
        """Search for research results based on the given query."""
        with dspy.settings.context(lm=self._lm):
            results: list[SearchResult] = await self._program.aforward(
                research_query=data
            )
        return results


class Reranker(Agent[tuple[ResearchQuery, list[SearchResult]], list[SearchResult]]):
    """Reranker agent that re-ranks search results."""

    def __init__(self, reranker_config: LMConfig) -> None:
        """Initialize the reranker agent.

        Args:
            reranker_config: The language model to use.
        """
        self._model = reranker_config.model
        self._api_key = reranker_config.api_key
        self._api_base = (
            str(reranker_config.base_url) if reranker_config.base_url else None
        )

    async def __call__(
        self, data: tuple[ResearchQuery, list[SearchResult]]
    ) -> list[SearchResult]:
        """Rerank search results based on relevance scores."""
        _, results = data
        relevance_scores = await self.relevance(data)
        return [results[item["index"]] for item in relevance_scores]

    async def relevance(
        self, data: tuple[ResearchQuery, list[SearchResult]]
    ) -> list[_RelevanceScore]:
        """Compute relevance scores for search results."""
        query, results = data
        if query.domains:
            query = f"Query: {query.text}; Domains: " + ", ".join(query.domains)
        else:
            query = f"Query: {query.text}"

        docs = [
            f"Title: {r.paper.title}; Abstract: {r.paper.abstract}" for r in results
        ]
        ranking = await litellm.arerank(
            model=self._model,
            api_key=self._api_key,
            api_base=self._api_base,
            query=query,
            documents=docs,
        )
        return ranking.results
