"""GEPA student program for search-agent optimization.

Layer: Infrastructure (optimization-only, not imported by runtime).

``SearchProgram`` wraps the runtime ``SearchAgent`` as a ``dspy.Module``
with an async ``forward`` method. Each forward call resets the session
``search_results`` to ensure training-example isolation. GEPA compiles
this program by optimizing the predictor instructions of its internal
``dspy.ReAct``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import dspy

from research_agent.search.agents import SearchAgent
from research_agent.search.tools import LiteratureSearch
from research_agent.shared.session import InMemorySession

if TYPE_CHECKING:
    from research_agent.search.models import PaperInfo, ResearchQuery
    from research_agent.shared.agent import LMConfig


class SearchProgram(dspy.Module):
    """Single-query search student: ReAct search + session hydration."""

    def __init__(
        self,
        lm_config: LMConfig,
        *,
        literature_search: LiteratureSearch | None = None,
    ) -> None:
        """Build the student with a persistent ReAct program.

        Args:
            lm_config: Student LM settings (``search-search`` role).
            literature_search: Pure literature index client. Defaults to
                the live index dispatcher; inject a fake in tests.
        """
        super().__init__()
        self._lm_config = lm_config
        self._literature_search = literature_search or LiteratureSearch()
        self._session = InMemorySession()

    async def forward(self, research_query: ResearchQuery) -> list[PaperInfo]:
        """Search for papers and return hydrated ``search_results``.

        Resets the session ``search_results`` before each query so that
        training examples cannot share accumulated hits.
        """
        # Clear accumulated search results for isolation
        self._session.set("search_results", [])

        agent = SearchAgent(self._lm_config, self._session, self._literature_search)
        return await agent(research_query)
