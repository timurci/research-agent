"""Opik scoring metrics adapting search domain metrics.

Layer: Infrastructure (evaluation harness).

* **search-search** — query-only rows; relevance labeled at score time by
  the ``search-rerank`` role (bootstrap signal unless held out).
* **search-suggest** — query+papers rows; length is a code metric;
  quality is labeled by a held-out ``llm-judge`` rubric judge.

Domain metric logic is not reimplemented here.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Protocol

from opik.evaluation.metrics import BaseMetric
from pydantic import TypeAdapter, ValidationError

from evals.feedback import evaluation_score_to_score_result
from evals.search.agents import reranker
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
    from collections.abc import Coroutine, Mapping, Sequence

    from opik.evaluation.metrics.score_result import ScoreResult

    from research_agent.shared.config.models import LMConfig
    from research_agent.shared.judge import JudgeVerdict


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


def _require_paper_list(outputs: object) -> list[PaperInfo]:
    """Narrow ``papers`` value to ``list[PaperInfo]``."""
    try:
        return _PAPER_LIST_ADAPTER.validate_python(outputs)
    except ValidationError as exc:
        msg = f"papers must be list[PaperInfo]: {exc}"
        raise ScorerShapeError(msg) from exc


def _require_suggestion(outputs: object) -> str:
    """Narrow scorer ``suggestion`` value to ``str``."""
    if not isinstance(outputs, str):
        msg = f"suggestion must be str, got {type(outputs).__name__}"
        raise ScorerShapeError(msg)
    return outputs


def format_suggestion_judge_input(query: ResearchQuery) -> str:
    """Format a research query as judge task input text."""
    if query.domains:
        domains = ", ".join(query.domains)
        return f"Query: {query.text}\nDomains: {domains}"
    return f"Query: {query.text}"


def format_suggestion_judge_context(papers: list[PaperInfo]) -> str:
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


def suggestion_quality_judge(
    *,
    lm_config: LMConfig | None = None,
) -> RubricJudge:
    """Build the held-out suggestion-quality rubric judge.

    Args:
        lm_config: Judge LM settings. Defaults to the ``llm-judge`` role
            from ``config/lm.yaml``.
    """
    config = lm_config if lm_config is not None else load_lm_config(ROLE_LLM_JUDGE)
    return RubricJudge(config, SUGGESTION_QUALITY_RUBRIC)


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

    Opik's scorer path is synchronous, so no event loop is running in
    the calling thread.
    """
    return asyncio.run(coro)


def search_query_scorers(
    *,
    lm_config: LMConfig | None = None,
) -> Sequence[BaseMetric]:
    """Default scorers for query-only search modules.

    Relevance uses the eval reranker (same family as the e2e workflow).
    Self-labeled relevance is a bootstrap signal only.

    Args:
        lm_config: Reranker LM settings for the relevance labeler.
            Defaults to the ``search-rerank`` role from ``config/lm.yaml``.
    """
    return (
        SearchResultCountMetric(),
        SearchResultRelevanceMetric(labeler=reranker(lm_config=lm_config)),
    )


def search_suggest_scorers(
    *,
    lm_config: LMConfig | None = None,
    quality_judge: _QualityJudge | None = None,
) -> Sequence[BaseMetric]:
    """Default scorers for the suggestion module.

    Combines the code length metric with a held-out LLM quality judge
    over ``SUGGESTION_QUALITY_RUBRIC``.

    Args:
        lm_config: Judge LM settings when *quality_judge* is omitted.
            Defaults to the ``llm-judge`` role from ``config/lm.yaml``.
        quality_judge: Optional pre-built judge; overrides *lm_config*.
    """
    judge = (
        quality_judge
        if quality_judge is not None
        else suggestion_quality_judge(lm_config=lm_config)
    )
    return (
        SuggestionLengthMetric(),
        SuggestionQualityMetric(judge=judge),
    )


class SearchResultCountMetric(BaseMetric):
    """Opik metric for search result volume (count band with peak)."""

    def __init__(self) -> None:
        """Initialize with metric name."""
        super().__init__(name="search_result_count")

    def score(
        self,
        *,
        papers: object,
        **kwargs: object,  # noqa: ARG002  # absorbs extra merged keys from Opik
    ) -> ScoreResult:
        """Score agent outputs using the domain count metric."""
        score = search_result_count(_require_paper_list(papers))
        return evaluation_score_to_score_result(score, name=self.name)


class SearchResultRelevanceMetric(BaseMetric):
    """Opik metric that labels relevance via a reranker agent.

    Search eval rows have no gold relevance labels. The metric calls
    the labeler with ``(ResearchQuery, list[PaperInfo])`` and feeds the
    scores into the domain ``search_result_relevance`` metric.
    """

    def __init__(self, *, labeler: _RelevanceLabeler) -> None:
        """Initialize with a relevance labeler."""
        super().__init__(name="search_result_relevance")
        self._labeler = labeler

    def score(
        self,
        *,
        query: object,
        papers: object,
        **kwargs: object,  # noqa: ARG002  # absorbs extra merged keys from Opik
    ) -> ScoreResult:
        """Score agent outputs using reranker-produced relevance."""
        papers_list = _require_paper_list(papers)
        if not papers_list:
            empty_score = search_result_relevance([], [])
            return evaluation_score_to_score_result(empty_score, name=self.name)
        research_query = ResearchQuery.model_validate(query)
        ranking = _run_coroutine(self._labeler.relevance((research_query, papers_list)))
        relevance_scores = relevance_metrics_from_ranking(papers_list, ranking)
        score = search_result_relevance(papers_list, relevance_scores)
        return evaluation_score_to_score_result(score, name=self.name)


class SuggestionLengthMetric(BaseMetric):
    """Opik metric for suggestion word-count band compliance."""

    def __init__(self) -> None:
        """Initialize with metric name."""
        super().__init__(name="suggestion_length")

    def score(
        self,
        *,
        suggestion: object,
        **kwargs: object,  # noqa: ARG002  # absorbs extra merged keys from Opik
    ) -> ScoreResult:
        """Score suggestion length using the domain metric."""
        score = suggestion_length(_require_suggestion(suggestion))
        return evaluation_score_to_score_result(score, name=self.name)


class SuggestionQualityMetric(BaseMetric):
    """Opik metric that labels suggestion quality via an LLM rubric judge.

    Rows supply ``query`` and ``papers`` as context; the task output is
    ``suggestion``. Pass/fail follows rubric fail conditions; the
    continuous score follows the discrete quality bands.
    """

    def __init__(self, *, judge: _QualityJudge) -> None:
        """Initialize with a quality judge."""
        super().__init__(name="suggestion_quality")
        self._judge = judge

    def score(
        self,
        *,
        query: object,
        papers: object,
        suggestion: object,
        **kwargs: object,  # noqa: ARG002  # absorbs extra merged keys from Opik
    ) -> ScoreResult:
        """Score suggestion quality using the held-out rubric judge."""
        research_query = ResearchQuery.model_validate(query)
        papers_list = _require_paper_list(papers)
        suggestion_text = _require_suggestion(suggestion)
        verdict = _run_coroutine(
            self._judge.judge(
                task_input=format_suggestion_judge_input(research_query),
                task_output=suggestion_text,
                task_context=format_suggestion_judge_context(papers_list),
            ),
        )
        score = suggestion_quality(
            score=verdict.score,
            passing=not verdict.failing,
            reason=verdict.reason,
        )
        return evaluation_score_to_score_result(score, name=self.name)
