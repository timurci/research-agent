"""Concrete agents for search evaluation.

Evals infrastructure: wires DSPy/LiteLLM adapters. Default ``LMConfig``
values are hard-coded (same endpoints as live tests) and overridable at
import time via suite-prefixed environment variables (no dotenv):

* ``SEARCH_MODEL``, ``SEARCH_API_KEY``, ``SEARCH_BASE_URL``
* ``SEARCH_RERANK_MODEL``, ``SEARCH_RERANK_API_KEY``, ``SEARCH_RERANK_BASE_URL``

Builders accept an optional ``lm_config`` to inject a custom config
(tests, alternate endpoints) without depending on process env.

MLflow scorers import these agents; they do not import DSPy themselves.

``SearchAgent`` binds a session at construction and treats
``search_results`` as append-only. Eval tasks are independent queries, so
``search_agent()`` returns a callable that builds a new
``InMemorySession`` (and ``SearchAgent``) on every call.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from pydantic import HttpUrl

from research_agent.search.agents import Reranker, SearchAgent
from research_agent.search.tools import LiteratureSearch
from research_agent.search.workflows import PaperSearchWorkflow
from research_agent.shared.agent import LMConfig
from research_agent.shared.session import InMemorySession

if TYPE_CHECKING:
    from research_agent.search.models import PaperInfo, ResearchQuery
    from research_agent.shared.agent import Agent

SEARCH_LM_CONFIG = LMConfig(
    model=os.environ.get("SEARCH_MODEL", "openai/Qwen3.5-4B"),
    api_key=os.environ.get("SEARCH_API_KEY", "ignored-auth-key"),
    base_url=HttpUrl(
        os.environ.get("SEARCH_BASE_URL", "http://localhost:8080/v1"),
    ),
)

RERANK_LM_CONFIG = LMConfig(
    model=os.environ.get("SEARCH_RERANK_MODEL", "infinity/LFM2.5-ColBERT-350M"),
    api_key=os.environ.get("SEARCH_RERANK_API_KEY", "ignored-auth-key"),
    base_url=HttpUrl(
        os.environ.get("SEARCH_RERANK_BASE_URL", "http://localhost:8080/v1"),
    ),
)


def search_agent(
    *,
    lm_config: LMConfig | None = None,
) -> Agent[ResearchQuery, list[PaperInfo]]:
    """Search agent for eval: new session per task.

    Returns a long-lived callable that builds a fresh ``SearchAgent`` with
    its own ``InMemorySession`` on every call so tasks cannot share
    ``search_results``.

    Args:
        lm_config: LM settings for constructed agents. Defaults to
            ``SEARCH_LM_CONFIG`` (env-overridable defaults at import time).

    Parameter name on the returned callable is ``data`` to match the
    ``Agent`` protocol. Prefer ``paper_search_workflow()`` as the MLflow
    ``predict_fn`` — eval rows use the ``query`` key, which matches the
    workflow's ``__call__``.
    """
    config = SEARCH_LM_CONFIG if lm_config is None else lm_config
    literature_search = LiteratureSearch()

    async def run(data: ResearchQuery) -> list[PaperInfo]:
        agent = SearchAgent(config, InMemorySession(), literature_search)
        return await agent(data)

    return run


def reranker(*, lm_config: LMConfig | None = None) -> Reranker:
    """Reranker for relevance scoring and the search workflow.

    Args:
        lm_config: Reranker LM settings. Defaults to ``RERANK_LM_CONFIG``.
    """
    return Reranker(RERANK_LM_CONFIG if lm_config is None else lm_config)


def paper_search_workflow(
    *,
    search_lm_config: LMConfig | None = None,
    rerank_lm_config: LMConfig | None = None,
) -> PaperSearchWorkflow:
    """Search → rerank workflow for ``predict_fn`` (query-only evalset).

    Use this as MLflow ``predict_fn``: eval inputs are
    ``{"query": ResearchQuery}``, matching ``PaperSearchWorkflow.__call__``.

    Args:
        search_lm_config: Forwarded to ``search_agent``.
        rerank_lm_config: Forwarded to ``reranker``.
    """
    return PaperSearchWorkflow(
        search_agent(lm_config=search_lm_config),
        reranker(lm_config=rerank_lm_config),
    )
