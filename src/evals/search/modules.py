"""Search evaluation module registry for ``mlflow.genai.evaluate``.

Query-only modules load HF research queries with no gold paper lists.
``search`` runs the search agent alone; ``search-e2e`` runs search then
rerank. Default relevance scorers label with the same default reranker
config family as the e2e workflow, so e2e relevance is largely
self-labeled — useful as a bootstrap, not as an independent ranking
judge. Prefer a held-out labeler via ``SEARCH_RERANK_*`` /
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

    from mlflow.genai.scorers import Scorer


def as_query_predict_fn(
    name: str,
    agent: Callable[[ResearchQuery], Awaitable[list[PaperInfo]]],
) -> Callable[..., Any]:
    """Wrap a query→papers callable as a traced MLflow ``predict_fn``."""

    @mlflow.trace(name=name)
    async def predict_fn(
        query: ResearchQuery | Mapping[str, Any],
    ) -> list[PaperInfo]:
        if isinstance(query, ResearchQuery):
            return await agent(query)
        return await agent(ResearchQuery.model_validate(dict(query)))

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
) -> EvalModule:
    """Build a query-shaped search ``EvalModule`` (one registry entry)."""
    return EvalModule(
        name=name,
        load_data=load_data,
        build_predict_fn=lambda: as_query_predict_fn(name, predict_factory()),
        build_scorers=scorers,
    )


MODULES: dict[str, EvalModule] = {
    "search-e2e": query_module("search-e2e", paper_search_workflow),
    "search": query_module("search", search_agent),
}
