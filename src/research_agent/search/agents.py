"""DSPy search agent and reranker adapters.

Layer: Infrastructure.

The search agent uses ReAct with a session-backed literature tool. Tool
calls union full ``PaperInfo`` records into session ``search_results``
(a ``set[PaperInfo]`` when present; missing key means empty) and return
slim title/abstract cards for new hits. After the tool loop, the LM reports
a ``SearchStatus``; the agent returns the index search results as a list.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import dspy
import litellm

from research_agent.search.tools import (
    LiteratureSearch,
    SessionLiteratureSearch,
)
from research_agent.shared.agent import Agent

from .models import PaperInfo, ResearchQuery, SearchStatus

if TYPE_CHECKING:
    from research_agent.shared.agent import LMConfig
    from research_agent.shared.session import Session

_MAX_REACT_ITERS: int = 10
_LITERATURE_SEARCH_TOOL_NAME: str = "LiteratureSearch"
_RERANK_MAX_ABSTRACT_CHARS: int = 2048


class _RelevanceScore(TypedDict):
    index: int
    relevance_score: float


class SearchAgentSignature(dspy.Signature):
    """Search literature for a research query.

    Use the LiteratureSearch tool to query PubMed, CrossRef, or OpenAlex.
    Gather papers by running one or more searches. Set status to the
    best-matching outcome.
    """

    research_query: ResearchQuery = dspy.InputField()
    status: SearchStatus = dspy.OutputField(
        desc=("Final search outcome"),
    )


def build_search_react(session_search: SessionLiteratureSearch) -> dspy.ReAct:
    """Build the search ReAct program bound to a session-backed tool.

    Args:
        session_search: Session-backed literature tool wrapping a pure
            index client. Owned by the caller so its ``_session`` can be
            swapped between runs for bag isolation.

    Returns:
        A ``dspy.ReAct`` over ``SearchAgentSignature`` with the session
        literature tool. Attach the return value as a ``dspy.Module``
        attribute when GEPA must discover its predictors.
    """
    tool = dspy.Tool(session_search, name=_LITERATURE_SEARCH_TOOL_NAME)
    return dspy.ReAct(
        SearchAgentSignature,
        tools=[tool],
        max_iters=_MAX_REACT_ITERS,
    )


class SearchAgent(Agent[ResearchQuery, list[PaperInfo]]):
    """Search agent: ReAct tool use and session-bag return.

    Session ``search_results`` is a deduplicating ``set[PaperInfo]`` when
    present. The agent returns ``list(bag)`` without LM selection.
    """

    def __init__(
        self,
        lm_config: LMConfig,
        session: Session,
        literature_search: LiteratureSearch,
    ) -> None:
        """Initialize the search agent for a session.

        Args:
            lm_config: The language model to use.
            session: Session holding ``search_results`` as
                ``set[PaperInfo]`` when present.
            literature_search: Pure literature index client.
        """
        self._lm = dspy.LM(
            model=lm_config.model,
            api_key=lm_config.api_key,
            api_base=str(lm_config.base_url) if lm_config.base_url else None,
            extra_body=lm_config.provider_config,
        )
        self._session = session
        self._session_search = SessionLiteratureSearch(session, literature_search)
        self._program = build_search_react(self._session_search)

    async def __call__(self, data: ResearchQuery) -> list[PaperInfo]:
        """Search for papers and return the session bag as a list."""
        with dspy.settings.context(lm=self._lm):
            await self._program.aforward(research_query=data)
        return list(SessionLiteratureSearch.papers(self._session))


class Reranker(Agent[tuple[ResearchQuery, list[PaperInfo]], list[PaperInfo]]):
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
        self._provider_config = reranker_config.provider_config

    async def __call__(
        self, data: tuple[ResearchQuery, list[PaperInfo]]
    ) -> list[PaperInfo]:
        """Rerank search results based on relevance scores."""
        _, results = data
        relevance_scores = await self.relevance(data)
        return [results[item["index"]] for item in relevance_scores]

    async def relevance(
        self, data: tuple[ResearchQuery, list[PaperInfo]]
    ) -> list[_RelevanceScore]:
        """Compute relevance scores for search results."""
        query, results = data
        if query.domains:
            query_text = f"Query: {query.text}; Domains: " + ", ".join(query.domains)
        else:
            query_text = f"Query: {query.text}"

        docs = [
            f"Title: {r.title}; Abstract: {r.abstract[:_RERANK_MAX_ABSTRACT_CHARS]}"
            for r in results
        ]
        ranking = await litellm.arerank(
            model=self._model,
            api_key=self._api_key,
            api_base=self._api_base,
            query=query_text,
            documents=docs,
            extra_body=self._provider_config,
        )
        return ranking.results
