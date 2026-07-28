"""Search evaluation module registry for ``opik.evaluate``.

Query-only modules load HF research queries with no gold paper lists.
``search-search`` runs the search agent alone and returns a list of papers.
The loaded dataset is capped at ``SEARCH_SAMPLE_LIMIT`` rows via seeded
subsampling (see ``evals.harness.sample_rows``). Default relevance scorers
label with the ``search-rerank`` config at score time. This is a bootstrap
signal, not an independent ranking judge. Prefer a held-out labeler via
injected ``LMConfig`` for ``search-rerank`` or ``reranker(lm_config=...)``.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from evals.harness import EvalModule
from evals.search.agents import search_agent
from evals.search.dataset import load_search_eval_data
from evals.search.scorers import search_query_scorers
from research_agent.search.models import PaperInfo, ResearchQuery

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping, Sequence
    from pathlib import Path

    from opik.evaluation.metrics import BaseMetric

    from research_agent.shared.config.models import LMConfig

MODULE_NAMES: frozenset[str] = frozenset({"search-search"})

SEARCH_SAMPLE_LIMIT: int = 30


def as_query_task(
    agent: Callable[[ResearchQuery], Awaitable[list[PaperInfo]]],
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Wrap a query->papers callable as an Opik task function."""

    def task(dataset_item: dict[str, Any]) -> dict[str, Any]:
        query = ResearchQuery.model_validate(dataset_item["query"])
        papers: list[PaperInfo] = _run_agent(agent(query))
        return {"papers": papers}

    return task


def _run_agent(coro: Awaitable[list[PaperInfo]]) -> list[PaperInfo]:
    """Run an async agent call from the synchronous Opik task path."""
    return asyncio.run(coro)


def query_module(
    name: str,
    predict_factory: Callable[
        [],
        Callable[[ResearchQuery], Awaitable[list[PaperInfo]]],
    ],
    *,
    load_data: Callable[[], list[dict[str, Any]]] = load_search_eval_data,
    scorers: Callable[[], Sequence[BaseMetric]] = search_query_scorers,
    sample_limit: int | None = None,
) -> EvalModule:
    """Build a query-shaped search ``EvalModule`` (one registry entry)."""
    return EvalModule(
        name=name,
        load_data=load_data,
        build_task=lambda: as_query_task(predict_factory()),
        build_scorers=scorers,
        sample_limit=sample_limit,
    )


def build_modules(
    *,
    search_lm_config: LMConfig,
    rerank_lm_config: LMConfig,
    instructions: Mapping[str, Path] | None = None,
) -> dict[str, EvalModule]:
    """Build search eval modules with injected LM configs.

    Args:
        search_lm_config: ``search-search`` settings for the search agent.
        rerank_lm_config: ``search-rerank`` settings for workflow rerank
            and relevance scorers.
        instructions: Optional module name -> optimized program path.
            Used to load GEPA-optimized instructions for the search agent.
    """
    search_instructions_path = (
        instructions.get("search-search") if instructions is not None else None
    )
    return {
        "search-search": query_module(
            "search-search",
            lambda: search_agent(
                lm_config=search_lm_config,
                instructions_path=search_instructions_path,
            ),
            scorers=lambda: search_query_scorers(lm_config=rerank_lm_config),
            sample_limit=SEARCH_SAMPLE_LIMIT,
        ),
    }
