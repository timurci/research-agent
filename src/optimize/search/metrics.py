"""DSPy/GEPA metric adapter for search-agent optimization.

Layer: Infrastructure (optimization harness).

GEPA accepts a single metric. ``search_query_metric`` is that entrypoint:
it runs the three domain search quality functions (count, non-duplicate,
relevance), averages their continuous scores, and concatenates feedback
for reflection.

Domain metric logic is not reimplemented here. Adapters map domain
``EvaluationScore`` to GEPA ``ScoreWithFeedback``. Relevance labels come
from a held-out labeler (``search-rerank``), not from the student.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Protocol

from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback
from pydantic import TypeAdapter, ValidationError

from optimize.feedback import evaluation_score_to_score_with_feedback
from optimize.search.agents import relevance_labeler
from research_agent.search.metrics import (
    RelevanceMetric,
    search_result_count,
    search_result_non_duplicate,
    search_result_relevance,
)
from research_agent.search.models import PaperInfo, ResearchQuery

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine, Mapping, Sequence

    from research_agent.shared.agent import LMConfig

__all__ = ["MetricShapeError", "search_query_metric"]


class MetricShapeError(Exception):
    """Raised when metric I/O cannot be coerced to domain types."""


class _RelevanceLabeler(Protocol):
    """Async provider of per-document relevance scores."""

    async def relevance(
        self,
        data: tuple[ResearchQuery, list[PaperInfo]],
    ) -> Sequence[Mapping[str, object]]:
        """Score papers for a query; items include index and score."""
        ...


_PAPER_LIST_ADAPTER: TypeAdapter[list[PaperInfo]] = TypeAdapter(list[PaperInfo])


def _require_paper_list(raw: object) -> list[PaperInfo]:
    """Narrow prediction payloads to ``list[PaperInfo]``."""
    try:
        return _PAPER_LIST_ADAPTER.validate_python(raw)
    except ValidationError as exc:
        msg = f"search_results must be list[PaperInfo]: {exc}"
        raise MetricShapeError(msg) from exc


def _research_query_from_example(example: object) -> ResearchQuery:
    """Extract a ``ResearchQuery`` from a DSPy gold example."""
    payload: object
    match example:
        case {"research_query": query}:
            payload = query
        case _ if hasattr(example, "research_query"):
            payload = example.research_query
        case _:
            msg = "example must carry a 'research_query' attribute or mapping key"
            raise MetricShapeError(msg)
    try:
        return ResearchQuery.model_validate(payload)
    except ValidationError as exc:
        msg = f"example.research_query must be a ResearchQuery: {exc}"
        raise MetricShapeError(msg) from exc


def _search_results_from_pred(pred: object) -> list[PaperInfo]:
    """Extract agent-level papers from a DSPy prediction."""
    if not hasattr(pred, "search_results"):
        msg = "prediction must expose a 'search_results' attribute"
        raise MetricShapeError(msg)
    return _require_paper_list(pred.search_results)


def _relevance_metrics_from_ranking(
    papers: list[PaperInfo],
    ranking: Sequence[Mapping[str, object]],
) -> list[RelevanceMetric]:
    """Align reranker hits to paper order; clamp scores to ``[0, 1]``."""
    if len(ranking) != len(papers):
        msg = (
            f"reranker returned {len(ranking)} scores for "
            f"{len(papers)} papers; lengths must match"
        )
        raise MetricShapeError(msg)

    by_index: dict[int, float] = {}
    for item in ranking:
        index = item["index"]
        score = item["relevance_score"]
        if not isinstance(index, int) or isinstance(index, bool):
            msg = f"reranker index must be int, got {type(index).__name__}"
            raise MetricShapeError(msg)
        if not isinstance(score, int | float) or isinstance(score, bool):
            msg = f"reranker relevance_score must be float, got {type(score).__name__}"
            raise MetricShapeError(msg)
        if index < 0 or index >= len(papers):
            msg = f"reranker index {index} out of range for {len(papers)} papers"
            raise MetricShapeError(msg)
        if index in by_index:
            msg = f"duplicate reranker index {index}"
            raise MetricShapeError(msg)
        by_index[index] = min(1.0, max(0.0, float(score)))

    expected = set(range(len(papers)))
    if set(by_index) != expected:
        msg = f"reranker indices {sorted(by_index)} != expected {sorted(expected)}"
        raise MetricShapeError(msg)

    return [RelevanceMetric(value=by_index[i]) for i in range(len(papers))]


def _run_coroutine[T](coro: Coroutine[object, object, T]) -> T:
    """Run an async coroutine from a synchronous metric context."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


def search_query_metric(
    *,
    lm_config: LMConfig | None = None,
    labeler: _RelevanceLabeler | None = None,
) -> Callable[..., ScoreWithFeedback]:
    """GEPA metric for search-agent optimization.

    Averages continuous domain scores for count, non-duplicate, and
    relevance, and concatenates feedback strings for reflection.

    Args:
        lm_config: Labeler LM settings when *labeler* is not provided.
            Defaults to the ``search-rerank`` role from ``config/lm.yaml``.
        labeler: Optional relevance provider; overrides *lm_config*.
            Not the optimization student.
    """
    active_labeler = (
        labeler if labeler is not None else relevance_labeler(lm_config=lm_config)
    )

    def metric(
        gold: Any,  # noqa: ANN401  # DSPy gold example is opaque
        pred: Any,  # noqa: ANN401  # DSPy prediction is opaque
        trace: Any | None = None,  # noqa: ANN401, ARG001  # GEPA signature; unused
        pred_name: str | None = None,  # noqa: ARG001  # GEPA signature; unused
        pred_trace: Any | None = None,  # noqa: ANN401, ARG001  # GEPA signature; unused
    ) -> ScoreWithFeedback:
        papers = _search_results_from_pred(pred)

        count = evaluation_score_to_score_with_feedback(
            search_result_count(papers),
            name="search_result_count",
        )
        non_dup = evaluation_score_to_score_with_feedback(
            search_result_non_duplicate(papers),
            name="search_result_non_duplicate",
        )

        if not papers:
            domain_relevance = search_result_relevance([], [])
        else:
            query = _research_query_from_example(gold)
            ranking = _run_coroutine(active_labeler.relevance((query, papers)))
            domain_relevance = search_result_relevance(
                papers,
                _relevance_metrics_from_ranking(papers, ranking),
            )
        rel = evaluation_score_to_score_with_feedback(
            domain_relevance,
            name="search_result_relevance",
        )

        combined_score = (count.score + non_dup.score + rel.score) / 3
        feedback = f"{count.feedback}\n{non_dup.feedback}\n{rel.feedback}"
        return ScoreWithFeedback(score=combined_score, feedback=feedback)

    return metric
