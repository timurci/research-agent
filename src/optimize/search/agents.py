"""Concrete agents for search optimization.

Optimization infrastructure: wires DSPy/LiteLLM adapters.

**Student under optimization:** the search agent only (one step).
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

``SearchAgent`` binds a session at construction and treats
``search_results`` as append-only. Optimize tasks are independent
queries, so ``search_agent()`` returns a callable that builds a new
``InMemorySession`` (and ``SearchAgent``) on every call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from research_agent.search.agents import Reranker, SearchAgent
from research_agent.search.tools import LiteratureSearch
from research_agent.shared.lm_config import ROLE_SEARCH_RERANK, ROLE_SEARCH_SEARCH
from research_agent.shared.lm_config import lm_config as load_lm_config
from research_agent.shared.session import InMemorySession

if TYPE_CHECKING:
    from research_agent.search.models import PaperInfo, ResearchQuery
    from research_agent.shared.agent import Agent, LMConfig


_LITERATURE_SEARCH = LiteratureSearch()


def search_agent(
    *,
    lm_config: LMConfig | None = None,
    literature_search: LiteratureSearch | None = None,
) -> Agent[ResearchQuery, list[PaperInfo]]:
    """Search agent student for optimize: new session per task.

    Returns a long-lived callable that builds a fresh ``SearchAgent`` with
    its own ``InMemorySession`` on every call so tasks cannot share
    ``search_results``.

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

    async def run(data: ResearchQuery) -> list[PaperInfo]:
        agent = SearchAgent(config, InMemorySession(), lit_search)
        return await agent(data)

    return run


def relevance_labeler(*, lm_config: LMConfig | None = None) -> Reranker:
    """Reranker used only to label relevance for metrics (not optimized).

    Args:
        lm_config: Labeler LM settings. Defaults to the
            ``search-rerank`` role from ``config/lm.yaml``.
    """
    config = lm_config if lm_config is not None else load_lm_config(ROLE_SEARCH_RERANK)
    return Reranker(config)
