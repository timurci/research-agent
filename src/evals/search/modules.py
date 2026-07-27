"""Search evaluation module registry for ``mlflow.genai.evaluate``.

Query-only modules load HF research queries with no gold paper lists.
``search-search`` runs the search agent alone; ``search-e2e`` runs search then
rerank. Both cap the loaded dataset at ``*_SAMPLE_LIMIT`` rows via
seeded subsampling (see ``evals.harness.sample_rows``). Default
relevance scorers label with the same default reranker config family as
the e2e workflow, so e2e relevance is largely self-labeled — useful as
a bootstrap, not as an independent ranking judge. Prefer a held-out
labeler via injected ``LMConfig`` for ``search-rerank`` or
``reranker(lm_config=...)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import mlflow

from evals.harness import EvalModule
from evals.search.agents import paper_search_workflow, search_agent
from evals.search.dataset import load_search_eval_data
from evals.search.scorers import search_query_scorers
from research_agent.search.models import PaperInfo, ResearchQuery

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping, Sequence
    from pathlib import Path

    from mlflow.genai.scorers import Scorer

    from research_agent.shared.agent import LMConfig

MODULE_NAMES: frozenset[str] = frozenset({"search-search", "search-e2e"})

SEARCH_SAMPLE_LIMIT: int = 30
SEARCH_E2E_SAMPLE_LIMIT: int = 30


def as_query_predict_fn(
    name: str,
    agent: Callable[[ResearchQuery], Awaitable[list[PaperInfo]]],
) -> Callable[..., Any]:
    """Wrap a query→papers callable as a traced MLflow ``predict_fn``."""

    @mlflow.trace(name=name)
    async def predict_fn(
        query: ResearchQuery | Mapping[str, Any],
    ) -> list[PaperInfo]:
        return await agent(ResearchQuery.model_validate(query))

    return predict_fn


def query_module(
    name: str,
    predict_factory: Callable[
        [],
        Callable[[ResearchQuery], Awaitable[list[PaperInfo]]],
    ],
    *,
    load_data: Callable[[], list[dict[str, Any]]] = load_search_eval_data,
    scorers: Callable[[], Sequence[Scorer]] = search_query_scorers,
    sample_limit: int | None = None,
) -> EvalModule:
    """Build a query-shaped search ``EvalModule`` (one registry entry)."""
    return EvalModule(
        name=name,
        load_data=load_data,
        build_predict_fn=lambda: as_query_predict_fn(name, predict_factory()),
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
        instructions: Optional module name → optimized program path.
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
        "search-e2e": query_module(
            "search-e2e",
            lambda: paper_search_workflow(
                search_lm_config=search_lm_config,
                rerank_lm_config=rerank_lm_config,
                search_instructions_path=search_instructions_path,
            ),
            scorers=lambda: search_query_scorers(lm_config=rerank_lm_config),
            sample_limit=SEARCH_E2E_SAMPLE_LIMIT,
        ),
    }
