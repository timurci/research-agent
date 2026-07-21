"""DSPy search agent and reranker adapters.

Layer: Infrastructure.

The search agent uses ReAct with an indexed literature tool. The LM
selects absolute indices into session ``search_results``; full
``PaperInfo`` records are resolved from a ``Session``.

``search_results`` is append-only for the lifetime of the session.
Indices remain valid across ReAct tool calls within one turn and across
turns that share the same session. Callers that need per-query isolation
must construct a fresh ``Session`` (and agent) per query. Construct one
session (and agent) per conversation session when multi-turn reuse of
hits is desired.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import dspy
import litellm

from research_agent.search.tools import (
    SEARCH_RESULTS_KEY,
    IndexedLiteratureSearch,
    LiteratureSearch,
)
from research_agent.shared.agent import Agent
from research_agent.shared.session import InvalidSessionStateError

from .models import PaperInfo, ResearchQuery

if TYPE_CHECKING:
    from research_agent.shared.agent import LMConfig
    from research_agent.shared.session import Session

_MAX_REACT_ITERS: int = 3
_LITERATURE_SEARCH_TOOL_NAME: str = "LiteratureSearch"


class UnknownSelectedIdError(Exception):
    """Raised when a selected id is not a valid index into search_results."""


class _RelevanceScore(TypedDict):
    index: int
    relevance_score: float


class SearchAgentSignature(dspy.Signature):
    """Search literature for a research query.

    Use the LiteratureSearch tool to query PubMed or CrossRef.
    After gathering results, set selected_ids to the integer ids from tool
    observations that best answer the research query (most relevant first).
    Do not invent ids; only use ids returned by LiteratureSearch.
    """

    research_query: ResearchQuery = dspy.InputField()
    selected_ids: list[int] = dspy.OutputField(
        desc="Integer ids from LiteratureSearch results, most relevant first",
    )


class SearchAgent(Agent[ResearchQuery, list[PaperInfo]]):
    """Search agent: ReAct tool use, index selection, session hydration.

    Session ``search_results`` is append-only across calls that share this
    agent. Selected ids may resolve papers recorded on earlier turns.
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
            session: Session holding append-only ``search_results``.
            literature_search: Pure literature index client.
        """
        self._lm = dspy.LM(
            model=lm_config.model,
            api_key=lm_config.api_key,
            api_base=str(lm_config.base_url) if lm_config.base_url else None,
            extra_body=lm_config.provider_config,
        )
        self._session = session
        indexed_search = IndexedLiteratureSearch(session, literature_search)
        tool = dspy.Tool(indexed_search, name=_LITERATURE_SEARCH_TOOL_NAME)
        self._program = dspy.ReAct(
            SearchAgentSignature,
            tools=[tool],
            max_iters=_MAX_REACT_ITERS,
        )

    async def __call__(self, data: ResearchQuery) -> list[PaperInfo]:
        """Search for papers and return full records for selected ids.

        Does not clear session ``search_results``; ids remain absolute for
        the session lifetime. Use a fresh session per query for isolation.
        """
        with dspy.settings.context(lm=self._lm):
            prediction = await self._program.aforward(research_query=data)
        selected_ids: list[int] = prediction.selected_ids
        if not selected_ids:
            return []
        return _papers_for_ids(self._session, selected_ids)


def _papers_for_ids(session: Session, selected_ids: list[int]) -> list[PaperInfo]:
    """Resolve selected indices against session ``search_results``.

    Raises:
        InvalidSessionStateError: If the bag is not a ``list[PaperInfo]``.
        UnknownSelectedIdError: If an id is not a non-negative index in range.
    """
    raw = session.get(SEARCH_RESULTS_KEY)
    if not isinstance(raw, list):
        msg = f"{SEARCH_RESULTS_KEY!r} must be a list[PaperInfo]"
        raise InvalidSessionStateError(msg)
    results: list[PaperInfo] = []
    for selected_id in selected_ids:
        if not isinstance(selected_id, int) or isinstance(selected_id, bool):
            msg = f"selected id must be a non-bool int, got {selected_id!r}"
            raise UnknownSelectedIdError(msg)
        if selected_id < 0 or selected_id >= len(raw):
            msg = (
                f"selected id {selected_id} out of range "
                f"for search_results of length {len(raw)}"
            )
            raise UnknownSelectedIdError(msg)
        paper = raw[selected_id]
        if not isinstance(paper, PaperInfo):
            msg = f"search_results[{selected_id}] is not a PaperInfo"
            raise InvalidSessionStateError(msg)
        results.append(paper)
    return results


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

        docs = [f"Title: {r.title}; Abstract: {r.abstract}" for r in results]
        ranking = await litellm.arerank(
            model=self._model,
            api_key=self._api_key,
            api_base=self._api_base,
            query=query_text,
            documents=docs,
            extra_body=self._provider_config,
        )
        return ranking.results
