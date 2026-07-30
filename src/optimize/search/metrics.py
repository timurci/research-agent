"""DSPy/GEPA metric adapters for search-slice optimization.

Layer: Infrastructure (optimization harness).

GEPA accepts a single metric per student:

* ``search_query_metric`` — count + relevance (held-out ``search-rerank``)
* ``search_suggest_metric`` — length + quality (held-out ``llm-judge``)

Domain metric logic is not reimplemented here. Adapters map domain
``EvaluationScore`` to GEPA ``ScoreWithFeedback``.
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
    search_result_relevance,
    suggestion_length,
    suggestion_quality,
)
from research_agent.search.models import PaperInfo, ResearchQuery
from research_agent.search.rubrics import SUGGESTION_QUALITY_RUBRIC
from research_agent.shared.config.lm import ROLE_LLM_JUDGE
from research_agent.shared.config.lm import lm_config as load_lm_config
from research_agent.shared.judge import RubricJudge

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine, Mapping, Sequence

    from research_agent.shared.config.models import LMConfig
    from research_agent.shared.judge import JudgeVerdict

__all__ = ["MetricShapeError", "search_query_metric", "search_suggest_metric"]


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


class _QualityJudge(Protocol):
    """Async provider of suggestion-quality verdicts."""

    async def judge(
        self,
        *,
        task_input: str,
        task_output: str,
        task_context: str = "",
    ) -> JudgeVerdict:
        """Score free-text output against a bound rubric."""
        ...


_PAPER_LIST_ADAPTER: TypeAdapter[list[PaperInfo]] = TypeAdapter(list[PaperInfo])
_JUDGE_ABSTRACT_CHARS: int = 500


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

    Averages continuous domain scores for count and relevance, and
    concatenates feedback strings for reflection.

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

        combined_score = (count.score + rel.score) / 2
        feedback = f"{count.feedback}\n{rel.feedback}"
        return ScoreWithFeedback(score=combined_score, feedback=feedback)

    return metric


def _format_suggestion_judge_input(query: ResearchQuery) -> str:
    """Format a research query as judge task input text."""
    if query.domains:
        domains = ", ".join(query.domains)
        return f"Query: {query.text}\nDomains: {domains}"
    return f"Query: {query.text}"


def _format_suggestion_judge_context(papers: list[PaperInfo]) -> str:
    """Format paper titles and abstract snippets as judge context."""
    if not papers:
        return "(no papers provided)"
    blocks: list[str] = []
    for index, paper in enumerate(papers, start=1):
        abstract = paper.abstract[:_JUDGE_ABSTRACT_CHARS]
        blocks.append(
            f"[{index}] Title: {paper.title}\nAbstract: {abstract}",
        )
    return "\n\n".join(blocks)


def _papers_from_example(example: object) -> list[PaperInfo]:
    """Extract paper inputs from a DSPy gold example."""
    payload: object
    match example:
        case {"papers": papers}:
            payload = papers
        case _ if hasattr(example, "papers"):
            payload = example.papers
        case _:
            msg = "example must carry a 'papers' attribute or mapping key"
            raise MetricShapeError(msg)
    try:
        return _PAPER_LIST_ADAPTER.validate_python(payload)
    except ValidationError as exc:
        msg = f"example.papers must be list[PaperInfo]: {exc}"
        raise MetricShapeError(msg) from exc


def _suggestion_from_pred(pred: object) -> str:
    """Extract suggestion text from a DSPy prediction."""
    if not hasattr(pred, "suggestion"):
        msg = "prediction must expose a 'suggestion' attribute"
        raise MetricShapeError(msg)
    suggestion = pred.suggestion
    if not isinstance(suggestion, str):
        msg = f"prediction.suggestion must be str, got {type(suggestion).__name__}"
        raise MetricShapeError(msg)
    return suggestion


def search_suggest_metric(
    *,
    lm_config: LMConfig | None = None,
    quality_judge: _QualityJudge | None = None,
) -> Callable[..., ScoreWithFeedback]:
    """GEPA metric for suggestion-generator optimization.

    Averages continuous domain scores for length and quality, and
    concatenates feedback strings for reflection. Quality uses a
    held-out rubric judge (``llm-judge``), not the student.

    Args:
        lm_config: Judge LM settings when *quality_judge* is not provided.
            Defaults to the ``llm-judge`` role from ``config/lm.yaml``.
        quality_judge: Optional judge; overrides *lm_config*. Not the
            optimization student.
    """
    active_judge: _QualityJudge
    if quality_judge is not None:
        active_judge = quality_judge
    else:
        config = lm_config if lm_config is not None else load_lm_config(ROLE_LLM_JUDGE)
        active_judge = RubricJudge(config, SUGGESTION_QUALITY_RUBRIC)

    def metric(
        gold: Any,  # noqa: ANN401  # DSPy gold example is opaque
        pred: Any,  # noqa: ANN401  # DSPy prediction is opaque
        trace: Any | None = None,  # noqa: ANN401, ARG001  # GEPA signature; unused
        pred_name: str | None = None,  # noqa: ARG001  # GEPA signature; unused
        pred_trace: Any | None = None,  # noqa: ANN401, ARG001  # GEPA signature; unused
    ) -> ScoreWithFeedback:
        suggestion = _suggestion_from_pred(pred)
        length = evaluation_score_to_score_with_feedback(
            suggestion_length(suggestion),
            name="suggestion_length",
        )

        query = _research_query_from_example(gold)
        papers = _papers_from_example(gold)
        verdict = _run_coroutine(
            active_judge.judge(
                task_input=_format_suggestion_judge_input(query),
                task_output=suggestion,
                task_context=_format_suggestion_judge_context(papers),
            ),
        )
        quality = evaluation_score_to_score_with_feedback(
            suggestion_quality(
                score=verdict.score,
                passing=not verdict.failing,
                reason=verdict.reason,
            ),
            name="suggestion_quality",
        )

        combined_score = (length.score + quality.score) / 2
        feedback = f"{length.feedback}\n{quality.feedback}"
        return ScoreWithFeedback(score=combined_score, feedback=feedback)

    return metric
