"""Concrete agents for search optimization.

Optimization infrastructure: wires DSPy/LiteLLM adapters.

**Student under optimization:** ``SearchProgram`` — a ``dspy.Module`` that
owns a persistent ``dspy.ReAct`` (``self.react``) so GEPA can discover and
rewrite predictor instructions. Each sync ``forward`` activates a fresh
``InMemorySession`` on the program's ``ScopedSession`` so concurrent
GEPA eval threads get a private paper bag, not a shared session race.

GEPA evaluates students via sync ``Module.__call__`` / ``forward`` (not
``aforward``), so this student is intentionally synchronous.

GEPA does not optimize the reranker or the search→rerank e2e workflow;
those are multi-step / separate capabilities and are out of scope here.

**Reranker** is built only as a relevance *labeler* for metrics (same
role as in evals scorers), not as a student program.

Default ``LMConfig`` values load from ``config/lm.yaml`` when not injected:

* ``search-search`` — search student agent
* ``search-rerank`` — relevance labeler only

Builders accept an optional ``lm_config`` to inject a custom config
(tests, CLI composition root, alternate endpoints) without reading the
file. Prefer injection over ambient defaults.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import dspy

from research_agent.search.agents import (
    Reranker,
    build_search_react,
)
from research_agent.search.tools import LiteratureSearch, SessionLiteratureSearch
from research_agent.shared.config.lm import ROLE_SEARCH_RERANK, ROLE_SEARCH_SEARCH
from research_agent.shared.config.lm import lm_config as load_lm_config
from research_agent.shared.scoped_session import ScopedSession
from research_agent.shared.session import InMemorySession

if TYPE_CHECKING:
    from research_agent.search.models import PaperInfo, ResearchQuery
    from research_agent.shared.agent import Agent
    from research_agent.shared.config.models import LMConfig
    from research_agent.shared.session import Session


_LITERATURE_SEARCH = LiteratureSearch()

__all__ = [
    "SearchProgram",
    "relevance_labeler",
    "search_agent",
]


class SearchProgram(dspy.Module):
    """GEPA student: session-isolated search with an owned ReAct module.

    Attaches ``self.react`` so ``named_predictors()`` is non-empty and GEPA
    can optimize instruction text. Uses sync ``forward`` so GEPA's
    evaluator can call the module. Runtime ``SearchAgent`` stays a plain
    async ``Agent``.
    """

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
        self._lm = dspy.LM(
            model=lm_config.model,
            api_key=lm_config.api_key,
            api_base=str(lm_config.base_url) if lm_config.base_url else None,
            extra_body=lm_config.provider_config,
        )
        self._literature_search = (
            literature_search if literature_search is not None else LiteratureSearch()
        )
        self._scoped = ScopedSession(InMemorySession())
        self._session_search = SessionLiteratureSearch(
            self._scoped,
            self._literature_search,
        )
        self.react = build_search_react(self._session_search)

    @property
    def session(self) -> Session:
        """Active session for the current thread (override or base)."""
        return self._scoped.active_session

    def forward(self, research_query: ResearchQuery) -> dspy.Prediction:
        """Run ReAct over a fresh session; attach its bag as ``search_results``.

        Activates a new ``InMemorySession`` on the program's
        ``ScopedSession`` for the duration of the call so each
        concurrent GEPA thread evaluates against its own bag without
        rebuilding the owned ``dspy.ReAct`` (GEPA-compiled predictors
        stay intact). Enables DSPy async-tool conversion so ReAct's sync
        path can run async index tools.

        Returns the ReAct prediction (trajectory, status, …) with
        ``search_results`` set from the fresh session bag for metrics.
        """
        fresh = InMemorySession()
        with self._scoped.use(fresh):
            with dspy.settings.context(
                lm=self._lm,
                allow_tool_async_sync_conversion=True,
            ):
                pred = self.react(research_query=research_query)
            pred.search_results = list(SessionLiteratureSearch.papers(fresh))
        return pred


def search_agent(
    *,
    lm_config: LMConfig | None = None,
    literature_search: LiteratureSearch | None = None,
) -> Agent[ResearchQuery, list[PaperInfo]]:
    """Search agent student for optimize: one program, isolated bag per task.

    Returns a long-lived callable over a single ``SearchProgram``. Each call
    runs sync ``forward`` with a private paper bag so tasks cannot share
    hits.

    Args:
        lm_config: LM settings for constructed agents. Defaults to the
            ``search-search`` role from ``config/lm.yaml``.
        literature_search: Pure literature index client. Defaults to a
            module-level singleton for connection pooling.
    """
    config = lm_config if lm_config is not None else load_lm_config(ROLE_SEARCH_SEARCH)
    lit_search = (
        literature_search if literature_search is not None else _LITERATURE_SEARCH
    )
    program = SearchProgram(lm_config=config, literature_search=lit_search)

    async def run(data: ResearchQuery) -> list[PaperInfo]:
        prediction = program(research_query=data)
        return list(prediction.search_results)

    return run


def relevance_labeler(*, lm_config: LMConfig | None = None) -> Reranker:
    """Reranker used only to label relevance for metrics (not optimized).

    Args:
        lm_config: Labeler LM settings. Defaults to the
            ``search-rerank`` role from ``config/lm.yaml``.
    """
    config = lm_config if lm_config is not None else load_lm_config(ROLE_SEARCH_RERANK)
    return Reranker(config)
