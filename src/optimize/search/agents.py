"""Concrete agents for search optimization.

Optimization infrastructure: wires DSPy/LiteLLM adapters.

**Student under optimization:** ``SearchProgram`` — a ``dspy.Module`` that
owns a persistent ``dspy.ReAct`` (``self.react``) so GEPA can discover and
rewrite predictor instructions. Each sync ``forward`` clears session
``search_results`` so train examples do not share hits.

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
    _papers_for_ids,
    build_search_react,
)
from research_agent.search.tools import SEARCH_RESULTS_KEY, LiteratureSearch
from research_agent.shared.lm_config import ROLE_SEARCH_RERANK, ROLE_SEARCH_SEARCH
from research_agent.shared.lm_config import lm_config as load_lm_config
from research_agent.shared.session import InMemorySession

if TYPE_CHECKING:
    from research_agent.search.models import PaperInfo, ResearchQuery
    from research_agent.shared.agent import Agent, LMConfig


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
        self._session = InMemorySession()
        self.react = build_search_react(self._session, self._literature_search)

    def forward(self, research_query: ResearchQuery) -> dspy.Prediction:
        """Search and return a prediction with hydrated ``search_results``.

        Resets session ``search_results`` before each query so training
        examples cannot share accumulated hits. Enables DSPy async-tool
        conversion so ReAct's sync path can run async index tools.
        """
        self._session.set(SEARCH_RESULTS_KEY, [])
        with dspy.settings.context(
            lm=self._lm,
            allow_tool_async_sync_conversion=True,
        ):
            raw = self.react(research_query=research_query)
        selected_ids: list[int] = raw.selected_ids
        if not selected_ids:
            papers: list[PaperInfo] = []
        else:
            papers = _papers_for_ids(self._session, selected_ids)
        return dspy.Prediction(search_results=papers)


def search_agent(
    *,
    lm_config: LMConfig | None = None,
    literature_search: LiteratureSearch | None = None,
) -> Agent[ResearchQuery, list[PaperInfo]]:
    """Search agent student for optimize: one program, cleared session per task.

    Returns a long-lived callable over a single ``SearchProgram``. Each call
    runs sync ``forward`` (clears ``search_results``) so tasks cannot share
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
