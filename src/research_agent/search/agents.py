"""DSPy search agent and reranker adapters.

Layer: Infrastructure.

The search agent uses ReAct with a session-backed literature tool. Tool
calls union full ``PaperInfo`` records into session ``search_results``
(a ``set[PaperInfo]`` when present; missing key means empty) and return
slim title/abstract cards for new hits. After the tool loop, the LM reports
a ``SearchOutcome``; the agent returns the index search results as a list.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, TypedDict

import dspy
import litellm

from research_agent.search.tools import (
    LiteratureSearch,
    SessionLiteratureSearch,
)
from research_agent.shared.agent import Agent

from .models import PaperInfo, ResearchQuery

if TYPE_CHECKING:
    from pathlib import Path

    from research_agent.shared.config.models import LMConfig
    from research_agent.shared.session import Session

_MAX_REACT_ITERS: int = 10
_LITERATURE_SEARCH_TOOL_NAME: str = "LiteratureSearch"
_RERANK_MAX_ABSTRACT_CHARS: int = 2048


class _RelevanceScore(TypedDict):
    index: int
    relevance_score: float


class SearchOutcome(StrEnum):
    """Terminal status of a literature search ReAct episode.

    Reported by the search agent after tool use; not used to filter the
    session paper bag. Values:

    * ``complete`` — enough useful papers gathered.
    * ``insufficient_search`` — stopped early or under-explored.
    * ``irrelevant_results`` — hits found but do not answer the query.
    * ``missing_results`` — little or no hits despite reasonable queries.
    * ``tool_error`` — tool or API failures dominated the trajectory.
    """

    COMPLETE = "complete"
    INSUFFICIENT_SEARCH = "insufficient_search"
    IRRELEVANT_RESULTS = "irrelevant_results"
    MISSING_RESULTS = "missing_results"
    TOOL_ERROR = "tool_error"


class SearchAgentSignature(dspy.Signature):
    """Search literature for a research query.

    Use the LiteratureSearch tool to query PubMed, CrossRef, or OpenAlex.
    Gather papers by running one or more searches. Set status to the
    best-matching outcome.
    """

    research_query: ResearchQuery = dspy.InputField()
    status: SearchOutcome = dspy.OutputField(
        desc=("Final search outcome"),
    )


class SuggestionGeneratorSignature(dspy.Signature):
    """Generate a practical manual research direction from the top papers.

    Given a research query and the most relevant papers found, write a
    concise, actionable suggestion for how a human researcher should
    proceed. Focus on practical next steps, not a full literature review.
    """

    research_query: ResearchQuery = dspy.InputField()
    papers: list[PaperInfo] = dspy.InputField()
    suggestion: str = dspy.OutputField(
        desc="Concise, practical research direction based on the papers",
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


class _SearchProgram(dspy.Module):
    """Runtime DSPy module wrapping the search ReAct.

    Mirrors the shape of the GEPA-optimized ``SearchProgram`` so that
    saved program JSON can be loaded onto the runtime agent's ReAct
    predictors via ``dspy.Module.load``.
    """

    def __init__(self, session_search: SessionLiteratureSearch) -> None:
        """Build the program with an owned ReAct module.

        Args:
            session_search: Session-backed literature tool.
        """
        super().__init__()
        self.react = build_search_react(session_search)

    async def aforward(self, research_query: ResearchQuery) -> SearchOutcome:
        prediction = await self.react.aforward(research_query=research_query)
        return SearchOutcome(prediction.status)


class _SuggestionGeneratorProgram(dspy.Module):
    """Runtime DSPy module wrapping the suggestion generator predictor.

    Mirrors the shape of a future GEPA-optimized suggestion program so
    saved program JSON can be loaded onto the runtime agent's predictor
    via ``dspy.Module.load``.
    """

    def __init__(self) -> None:
        """Build the program with an owned predictor."""
        super().__init__()
        self.predict = dspy.Predict(SuggestionGeneratorSignature)

    async def aforward(
        self,
        research_query: ResearchQuery,
        papers: list[PaperInfo],
    ) -> str:
        prediction = await self.predict.aforward(
            research_query=research_query,
            papers=papers,
        )
        return prediction.suggestion


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
        *,
        instructions_path: Path | None = None,
    ) -> None:
        """Initialize the search agent for a session.

        Args:
            lm_config: The language model to use.
            session: Session holding ``search_results`` as
                ``set[PaperInfo]`` when present.
            literature_search: Pure literature index client.
            instructions_path: Optional path to a saved DSPy program whose
                optimized instructions are loaded onto the ReAct.
        """
        self._lm = dspy.LM(
            model=lm_config.model,
            api_key=lm_config.api_key,
            api_base=str(lm_config.base_url) if lm_config.base_url else None,
            extra_body=lm_config.provider_config,
        )
        self._session = session
        self._session_search = SessionLiteratureSearch(session, literature_search)
        self._program = _SearchProgram(self._session_search)
        if instructions_path is not None:
            self._program.load(str(instructions_path))

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


class SuggestionGenerator(
    Agent[tuple[ResearchQuery, list[PaperInfo]], str],
):
    """Suggestion generator agent that produces a manual research direction."""

    def __init__(
        self,
        lm_config: LMConfig,
        *,
        instructions_path: Path | None = None,
    ) -> None:
        """Initialize the suggestion generator agent.

        Args:
            lm_config: The language model to use.
            instructions_path: Optional path to a saved DSPy program whose
                optimized instructions are loaded onto the predictor.
        """
        self._lm = dspy.LM(
            model=lm_config.model,
            api_key=lm_config.api_key,
            api_base=str(lm_config.base_url) if lm_config.base_url else None,
            extra_body=lm_config.provider_config,
        )
        self._program = _SuggestionGeneratorProgram()
        if instructions_path is not None:
            self._program.load(str(instructions_path))

    async def __call__(
        self,
        data: tuple[ResearchQuery, list[PaperInfo]],
    ) -> str:
        """Generate a research-direction suggestion from the top papers."""
        query, papers = data
        with dspy.settings.context(lm=self._lm):
            return await self._program.aforward(
                research_query=query,
                papers=papers,
            )
