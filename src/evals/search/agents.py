"""Concrete agents for search evaluation.

Evals infrastructure: wires DSPy/LiteLLM adapters. Default ``LMConfig``
values load from ``config/lm.yaml`` (``DEFAULT_LM_CONFIG_PATH``) when not
injected:

* ``search-search`` — search agent
* ``search-rerank`` — reranker / relevance labeler

Builders accept an optional ``lm_config`` to inject a custom config
(tests, CLI composition root, alternate endpoints) without reading the
file. Prefer injection over ambient defaults.

MLflow scorers import these agents; they do not import DSPy themselves.

``SearchAgent`` binds a session at construction and stores
``search_results`` as ``set[PaperInfo]`` when present.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from research_agent.search.agents import Reranker, SearchAgent
from research_agent.search.tools import LiteratureSearch
from research_agent.search.workflows import PaperSearchWorkflow
from research_agent.shared.config.lm import ROLE_SEARCH_RERANK, ROLE_SEARCH_SEARCH
from research_agent.shared.config.lm import lm_config as load_lm_config
from research_agent.shared.session import InMemorySession

if TYPE_CHECKING:
    from pathlib import Path

    from research_agent.search.models import PaperInfo, ResearchQuery
    from research_agent.shared.agent import Agent
    from research_agent.shared.config.models import LMConfig


def search_agent(
    *,
    lm_config: LMConfig | None = None,
    instructions_path: Path | None = None,
) -> Agent[ResearchQuery, list[PaperInfo]]:
    """Search agent for eval: new session per task.

    Returns a long-lived callable that builds a fresh ``SearchAgent`` with
    its own ``InMemorySession`` on every call so tasks cannot share
    ``search_results``.

    Args:
        lm_config: LM settings for constructed agents. Defaults to the
            ``search-search`` role from ``config/lm.yaml``.
        instructions_path: Optional path to a saved DSPy program whose
            optimized instructions are loaded onto each fresh search agent.

    Parameter name on the returned callable is ``data`` to match the
    ``Agent`` protocol. Prefer ``paper_search_workflow()`` as the MLflow
    ``predict_fn`` — eval rows use the ``query`` key, which matches the
    workflow's ``__call__``.
    """
    config = lm_config if lm_config is not None else load_lm_config(ROLE_SEARCH_SEARCH)
    literature_search = LiteratureSearch()

    async def run(data: ResearchQuery) -> list[PaperInfo]:
        agent = SearchAgent(
            config,
            InMemorySession(),
            literature_search,
            instructions_path=instructions_path,
        )
        return await agent(data)

    return run


def reranker(*, lm_config: LMConfig | None = None) -> Reranker:
    """Reranker for relevance scoring and the search workflow.

    Args:
        lm_config: Reranker LM settings. Defaults to the
            ``search-rerank`` role from ``config/lm.yaml``.
    """
    config = lm_config if lm_config is not None else load_lm_config(ROLE_SEARCH_RERANK)
    return Reranker(config)


def paper_search_workflow(
    *,
    search_lm_config: LMConfig | None = None,
    rerank_lm_config: LMConfig | None = None,
    search_instructions_path: Path | None = None,
) -> PaperSearchWorkflow:
    """Search → rerank workflow for ``predict_fn`` (query-only evalset).

    Use this as MLflow ``predict_fn``: eval inputs are
    ``{"query": ResearchQuery}``, matching ``PaperSearchWorkflow.__call__``.

    Args:
        search_lm_config: Forwarded to ``search_agent``.
        rerank_lm_config: Forwarded to ``reranker``.
        search_instructions_path: Forwarded to ``search_agent``.
    """
    return PaperSearchWorkflow(
        search_agent(
            lm_config=search_lm_config,
            instructions_path=search_instructions_path,
        ),
        reranker(lm_config=rerank_lm_config),
    )
