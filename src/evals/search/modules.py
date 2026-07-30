"""Search evaluation module registry for ``opik.evaluate``.

* ``search-search`` — search agent alone; query-only HF test rows;
  relevance labeled with ``search-rerank`` at score time (bootstrap).
* ``search-suggest`` — suggestion generator alone; local Opik
  search-search I/O export with fixed query+papers inputs; length +
  ``llm-judge`` quality scorers.

Loaded datasets are capped via seeded subsampling (see
``evals.harness.sample_rows``).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from evals.harness import EvalModule
from evals.search.agents import search_agent, suggestion_generator
from evals.search.dataset import load_search_eval_data, load_suggest_eval_data
from evals.search.scorers import search_query_scorers, search_suggest_scorers
from research_agent.search.models import PaperInfo, ResearchQuery

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping, Sequence
    from pathlib import Path

    from opik.evaluation.metrics import BaseMetric

    from research_agent.shared.config.models import LMConfig

MODULE_NAMES: frozenset[str] = frozenset({"search-search", "search-suggest"})

SEARCH_SAMPLE_LIMIT: int = 30
SUGGEST_SAMPLE_LIMIT: int = 30


def as_query_task(
    agent: Callable[[ResearchQuery], Awaitable[list[PaperInfo]]],
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Wrap a query->papers callable as an Opik task function."""

    def task(dataset_item: dict[str, Any]) -> dict[str, Any]:
        query = ResearchQuery.model_validate(dataset_item["query"])
        papers: list[PaperInfo] = _run_agent(agent(query))
        return {"papers": papers}

    return task


def as_suggest_task(
    agent: Callable[[tuple[ResearchQuery, list[PaperInfo]]], Awaitable[str]],
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Wrap a (query, papers)->suggestion callable as an Opik task."""

    def task(dataset_item: dict[str, Any]) -> dict[str, Any]:
        query = ResearchQuery.model_validate(dataset_item["query"])
        papers = [PaperInfo.model_validate(paper) for paper in dataset_item["papers"]]
        suggestion = _run_suggest_agent(agent((query, papers)))
        return {
            "query": query.model_dump(mode="json"),
            "papers": [paper.model_dump(mode="json") for paper in papers],
            "suggestion": suggestion,
        }

    return task


def _run_agent(coro: Awaitable[list[PaperInfo]]) -> list[PaperInfo]:
    """Run an async agent call from the synchronous Opik task path."""
    return asyncio.run(coro)


def _run_suggest_agent(coro: Awaitable[str]) -> str:
    """Run an async suggestion agent from the synchronous Opik task path."""
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


def suggest_module(
    name: str,
    predict_factory: Callable[
        [],
        Callable[[tuple[ResearchQuery, list[PaperInfo]]], Awaitable[str]],
    ],
    *,
    load_data: Callable[[], list[dict[str, Any]]] = load_suggest_eval_data,
    scorers: Callable[[], Sequence[BaseMetric]] = search_suggest_scorers,
    sample_limit: int | None = None,
) -> EvalModule:
    """Build a suggestion-shaped ``EvalModule`` (one registry entry)."""
    return EvalModule(
        name=name,
        load_data=load_data,
        build_task=lambda: as_suggest_task(predict_factory()),
        build_scorers=scorers,
        sample_limit=sample_limit,
    )


def build_modules(
    *,
    search_lm_config: LMConfig,
    rerank_lm_config: LMConfig,
    suggest_lm_config: LMConfig,
    judge_lm_config: LMConfig,
    instructions: Mapping[str, Path] | None = None,
) -> dict[str, EvalModule]:
    """Build search eval modules with injected LM configs.

    Args:
        search_lm_config: ``search-search`` settings for the search agent.
        rerank_lm_config: ``search-rerank`` settings for relevance scorers.
        suggest_lm_config: ``search-suggest`` settings for the suggestion
            generator.
        judge_lm_config: ``llm-judge`` settings for suggestion quality.
        instructions: Optional module name -> optimized program path.
            Used to load GEPA-optimized instructions for search and
            suggest agents.
    """
    search_instructions_path = (
        instructions.get("search-search") if instructions is not None else None
    )
    suggest_instructions_path = (
        instructions.get("search-suggest") if instructions is not None else None
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
        "search-suggest": suggest_module(
            "search-suggest",
            lambda: suggestion_generator(
                lm_config=suggest_lm_config,
                instructions_path=suggest_instructions_path,
            ),
            scorers=lambda: search_suggest_scorers(lm_config=judge_lm_config),
            sample_limit=SUGGEST_SAMPLE_LIMIT,
        ),
    }
