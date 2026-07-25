"""MLflow scorers adapting search domain metrics.

Layer: Infrastructure (evaluation harness).

Search eval datasets are query-only: rows carry a ``ResearchQuery`` and
no gold paper lists. ``predict_fn`` is the search ``Agent`` /
``PaperSearchWorkflow`` (``ResearchQuery`` → ``list[PaperInfo]``).

Relevance scores are produced at score time by a reranker agent, not
loaded from expectations. Default query scorers use the same
``search-rerank`` YAML role as the e2e workflow reranker, so e2e
relevance is largely self-labeled (bootstrap only). Point the labeler
at a held-out model via ``config/lm.yaml`` or
``reranker(lm_config=...)``.

Domain metric logic is not reimplemented here.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Protocol

from mlflow.genai.scorers import scorer
from pydantic import TypeAdapter, ValidationError

from evals.feedback import code_assessment_source, evaluation_score_to_feedback
from evals.search.agents import reranker
from research_agent.search.metrics import (
    RelevanceMetric,
    search_result_count,
    search_result_relevance,
)
from research_agent.search.models import PaperInfo, ResearchQuery

if TYPE_CHECKING:
    from collections.abc import Coroutine, Mapping, Sequence

    from mlflow.entities import Feedback
    from mlflow.genai.scorers import Scorer

    from research_agent.shared.agent import LMConfig

SEARCH_METRICS_SOURCE = code_assessment_source("research_agent.search.metrics")


class ScorerShapeError(Exception):
    """Raised when scorer I/O cannot be coerced to domain types."""


class _RelevanceLabeler(Protocol):
    """Async provider of per-document relevance scores."""

    async def relevance(
        self,
        data: tuple[ResearchQuery, list[PaperInfo]],
    ) -> Sequence[Mapping[str, object]]:
        """Score papers for a query; items include index and score."""
        ...


_PAPER_LIST_ADAPTER: TypeAdapter[list[PaperInfo]] = TypeAdapter(list[PaperInfo])


def _require_paper_list(outputs: object) -> list[PaperInfo]:
    """Narrow scorer outputs to ``list[PaperInfo]``."""
    try:
        return _PAPER_LIST_ADAPTER.validate_python(outputs)
    except ValidationError as exc:
        msg = f"outputs must be list[PaperInfo]: {exc}"
        raise ScorerShapeError(msg) from exc


def research_query_from_inputs(inputs: object) -> ResearchQuery:
    """Extract a ``ResearchQuery`` from MLflow scorer ``inputs``.

    Expects a ``ResearchQuery`` or a kwargs dict with ``query`` (model or
    field dict), matching ``load_search_eval_data`` / ``as_query_predict_fn``.

    Args:
        inputs: Value passed as the scorer ``inputs`` argument.

    Returns:
        Domain research query.

    Raises:
        ScorerShapeError: If ``inputs`` cannot be interpreted as a query.
    """
    match inputs:
        case {"query": query}:
            payload = query
        case dict():
            msg = "inputs must include 'query'"
            raise ScorerShapeError(msg)
        case _:
            payload = inputs
    try:
        return ResearchQuery.model_validate(payload)
    except ValidationError as exc:
        msg = f"inputs must be ResearchQuery or dict with 'query': {exc}"
        raise ScorerShapeError(msg) from exc


def relevance_metrics_from_ranking(
    papers: list[PaperInfo],
    ranking: Sequence[Mapping[str, object]],
) -> list[RelevanceMetric]:
    """Align reranker hits to the original paper order.

    LiteLLM/ColBERT scores are not guaranteed to lie in ``[0, 1]``. Values
    outside that range are clamped at this adapter boundary so domain
    ``RelevanceMetric`` construction succeeds without reimplementing
    metric logic.

    Args:
        papers: Search results in the order returned by the agent.
        ranking: Reranker hits with ``index`` into *papers* and
            ``relevance_score``.

    Returns:
        ``RelevanceMetric`` list parallel to *papers*.

    Raises:
        ScorerShapeError: If indices or scores are incomplete or mistyped.
    """
    if len(ranking) != len(papers):
        msg = (
            f"reranker returned {len(ranking)} scores for "
            f"{len(papers)} papers; lengths must match"
        )
        raise ScorerShapeError(msg)

    by_index: dict[int, float] = {}
    for item in ranking:
        index = item["index"]
        score = item["relevance_score"]
        if not isinstance(index, int) or isinstance(index, bool):
            msg = f"reranker index must be int, got {type(index).__name__}"
            raise ScorerShapeError(msg)
        if not isinstance(score, int | float) or isinstance(score, bool):
            msg = f"reranker relevance_score must be float, got {type(score).__name__}"
            raise ScorerShapeError(msg)
        if index < 0 or index >= len(papers):
            msg = f"reranker index {index} out of range for {len(papers)} papers"
            raise ScorerShapeError(msg)
        if index in by_index:
            msg = f"duplicate reranker index {index}"
            raise ScorerShapeError(msg)
        by_index[index] = min(1.0, max(0.0, float(score)))

    expected = set(range(len(papers)))
    if set(by_index) != expected:
        msg = f"reranker indices {sorted(by_index)} != expected {sorted(expected)}"
        raise ScorerShapeError(msg)

    return [RelevanceMetric(value=by_index[i]) for i in range(len(papers))]


def _run_coroutine[T](coro: Coroutine[object, object, T]) -> T:
    """Run an async coroutine from a synchronous scorer via ``asyncio.run``.

    MLflow's scorer path is synchronous, so no event loop is running in
    the calling thread.
    """
    return asyncio.run(coro)


def _relevance_feedback(
    papers: list[PaperInfo],
    relevance_scores: list[RelevanceMetric],
) -> Feedback:
    """Map domain relevance metric output to MLflow Feedback."""
    score = search_result_relevance(papers, relevance_scores)
    return evaluation_score_to_feedback(
        score,
        source=SEARCH_METRICS_SOURCE,
        name="search_result_relevance",
    )


def search_query_scorers(
    *,
    lm_config: LMConfig | None = None,
) -> Sequence[Scorer]:
    """Default scorers for query-only search modules.

    Relevance uses the eval reranker (same family as the e2e workflow).
    Self-labeled relevance is a bootstrap signal only.

    Args:
        lm_config: Reranker LM settings for the relevance labeler.
            Defaults to the ``search-rerank`` role from ``config/lm.yaml``.
    """
    return (
        search_result_count_scorer,
        make_search_result_relevance_scorer(reranker(lm_config=lm_config)),
    )


def make_search_result_relevance_scorer(labeler: _RelevanceLabeler) -> Scorer:
    """Build an MLflow scorer that labels relevance via a reranker agent.

    Search eval rows have no gold relevance labels. The returned scorer
    calls *labeler* with ``(ResearchQuery, list[PaperInfo])`` and feeds
    the scores into the domain ``search_result_relevance`` metric.

    Construct per evalset, e.g.::

        make_search_result_relevance_scorer(reranker())

    Args:
        labeler: Async relevance provider (typically ``Reranker``).

    Returns:
        MLflow code-based scorer for ``mlflow.genai.evaluate``.
    """

    @scorer(name="search_result_relevance")
    def search_result_relevance_scorer(
        *,
        inputs: object = None,
        outputs: object = None,
    ) -> Feedback:
        """Score agent outputs using reranker-produced relevance."""
        papers = _require_paper_list(outputs)
        if not papers:
            return _relevance_feedback([], [])
        query = research_query_from_inputs(inputs)
        ranking = _run_coroutine(labeler.relevance((query, papers)))
        relevance_scores = relevance_metrics_from_ranking(papers, ranking)
        return _relevance_feedback(papers, relevance_scores)

    return search_result_relevance_scorer


@scorer(name="search_result_count")
def search_result_count_scorer(*, outputs: object = None) -> Feedback:
    """MLflow scorer for search result volume (integer count vs target)."""
    score = search_result_count(_require_paper_list(outputs))
    return evaluation_score_to_feedback(
        score,
        source=SEARCH_METRICS_SOURCE,
        name="search_result_count",
    )
