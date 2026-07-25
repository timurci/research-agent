"""AI Evaluation metrics for search.

Layer: Domain.
"""

from collections import Counter
from math import log2
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from research_agent.shared.metric import EvaluationScore

if TYPE_CHECKING:
    from research_agent.search.models import PaperInfo

HIGH_RELEVANCE_THRESHOLD = 0.7
MEDIUM_RELEVANCE_THRESHOLD = 0.3
HIGH_RELEVANCE_SCORE = 1.0
MEDIUM_RELEVANCE_SCORE = 0.5
NDCG_AT_K = 10
TARGET_RESULT_COUNT = 25
MIN_PASS_RESULT_COUNT = 5
SCORE_AT_TARGET = 0.95
OVERSHOOT_HALF_CREDIT = 25
MIN_RELEVANCE_GRADE = 0
MAX_RELEVANCE_GRADE = 3


class RelevanceMetric(BaseModel):
    """Represents a relevance score."""

    model_config = ConfigDict(frozen=True)

    value: float = Field(..., ge=0.0, le=1.0)


def search_result_relevance(
    search_results: list[PaperInfo], relevance_scores: list[RelevanceMetric]
) -> EvaluationScore:
    """Determines if a paper is relevant to the research topic.

    Args:
        search_results: A list of search results.
        relevance_scores: Relevance scores determined by an embedding or BERT model.
    """
    if len(search_results) != len(relevance_scores):
        error_msg = (
            f"search_results ({len(search_results)})"
            f"and relevance_scores ({len(relevance_scores)}) must have the same length"
        )
        raise ValueError(error_msg)

    if not search_results:
        return EvaluationScore(passing=False, reason="Empty input", score=0.0)

    counter = Counter()
    low_relevance_items: list[int] = []
    high_relevance_items: list[int] = []

    cumulative_metric_score = 0.0

    for i, score in enumerate(relevance_scores):
        if score.value >= HIGH_RELEVANCE_THRESHOLD:
            counter["HIGH"] += 1
            cumulative_metric_score += HIGH_RELEVANCE_SCORE
            high_relevance_items.append(i)
        elif score.value >= MEDIUM_RELEVANCE_THRESHOLD:
            counter["MEDIUM"] += 1
            cumulative_metric_score += MEDIUM_RELEVANCE_SCORE
        else:
            counter["LOW"] += 1
            low_relevance_items.append(i)

    metric_score = cumulative_metric_score / len(relevance_scores)

    if counter["LOW"] > len(relevance_scores) * 0.25:
        passing = False
        verdict = "Too many low relevance results."
    elif counter["HIGH"] > len(relevance_scores) * 0.5:
        passing = True
        verdict = "Results are mostly high relevance."
    else:
        passing = True
        verdict = "Results are within acceptable bounds."

    low_relevance_titles = [search_results[i].title for i in low_relevance_items]
    high_relevance_titles = [search_results[i].title for i in high_relevance_items]

    reason = (
        f"Verdict\n{verdict}"
        f"\n\nLow relevance\n{low_relevance_titles}"
        f"\n\nHigh relevance\n{high_relevance_titles}"
    )

    return EvaluationScore(passing=passing, reason=reason, score=metric_score)


def _volume_score(count: int) -> float:
    """Map result count to a float in ``[0, 1]`` for optimizers.

    Linear from 0 to ``SCORE_AT_TARGET`` over ``[0, TARGET_RESULT_COUNT]``,
    then a concave residual ``1 - SCORE_AT_TARGET`` for larger counts so
    volume past the target is only weakly rewarded (relevance remains the
    stronger co-objective).
    """
    if count <= 0:
        return 0.0
    if count <= TARGET_RESULT_COUNT:
        return SCORE_AT_TARGET * (count / TARGET_RESULT_COUNT)
    overshoot = count - TARGET_RESULT_COUNT
    residual = 1.0 - SCORE_AT_TARGET
    return SCORE_AT_TARGET + residual * (
        overshoot / (overshoot + OVERSHOOT_HALF_CREDIT)
    )


def search_result_count(search_results: list[PaperInfo]) -> EvaluationScore:
    """Scores the volume of search results.

    Relevance alone can be gamed by returning a single highly relevant
    paper. This metric rewards longer result lists using the integer
    count ``n = len(search_results)``:

    - Linear segment: for ``n <= TARGET_RESULT_COUNT``,
      ``score = SCORE_AT_TARGET * n / TARGET_RESULT_COUNT``
      (reaches ``SCORE_AT_TARGET`` at the target, not 1.0).
    - Concave tail: for ``n > TARGET_RESULT_COUNT``, the residual
      ``1 - SCORE_AT_TARGET`` is allocated with diminishing returns
      ``u / (u + OVERSHOOT_HALF_CREDIT)`` where ``u = n - target``.
      Extra papers past the target are weakly rewarded so optimizers
      prefer improving relevance over unbounded volume.
    - ``passing`` is true when ``n >= MIN_PASS_RESULT_COUNT`` (eval
      gate / suite pass rate). The pass floor is coarser than the
      continuous score so AI evals and optimization use different
      resolutions of the same count.

    Pair with ``search_result_relevance`` so optimizers cannot trade
    volume for junk: volume and relevance are separate criteria.
    Session storage deduplicates by ``PaperInfo`` equality, so a
    separate non-duplicate metric is not required.

    Args:
        search_results: Papers returned by the search agent or workflow.

    Returns:
        Evaluation score whose continuous value is the volume map and
        whose pass/fail reflects the minimum volume floor.
    """
    count = len(search_results)
    score = _volume_score(count)
    passing = count >= MIN_PASS_RESULT_COUNT
    if count >= TARGET_RESULT_COUNT:
        reason = (
            f"Returned {count} results "
            f"(target {TARGET_RESULT_COUNT} met, pass floor {MIN_PASS_RESULT_COUNT})."
        )
    elif passing:
        reason = (
            f"Returned {count} results "
            f"(target {TARGET_RESULT_COUNT}, pass floor {MIN_PASS_RESULT_COUNT})."
        )
    else:
        reason = (
            f"Returned {count} results; "
            f"need at least {MIN_PASS_RESULT_COUNT} to pass "
            f"(target {TARGET_RESULT_COUNT})."
        )
    return EvaluationScore(passing=passing, reason=reason, score=score)


def _validate_relevance_grades(doc_relevance_scores: dict[str, int]) -> None:
    """Raise if any ground-truth grade is outside the supported range."""
    for doc_id, grade in doc_relevance_scores.items():
        if not MIN_RELEVANCE_GRADE <= grade <= MAX_RELEVANCE_GRADE:
            error_msg = (
                f"Relevance grade for {doc_id!r} must be in "
                f"[{MIN_RELEVANCE_GRADE}, {MAX_RELEVANCE_GRADE}], got {grade}"
            )
            raise ValueError(error_msg)


def _dcg_at_k(relevances: list[int], k: int) -> float:
    """Discounted cumulative gain at rank ``k`` with exponential gain."""
    total = 0.0
    for i, rel in enumerate(relevances[:k]):
        total += (2**rel - 1) / log2(i + 2)
    return total


def ndcg_at_10(
    reranked_docs: list[str],
    doc_relevance_scores: dict[str, int],
) -> float:
    """Compute NDCG@10 for a reranked document list.

    Uses graded relevance in ``[0, 3]`` and the exponential gain
    ``2^rel - 1`` with logarithmic rank discount. Ideal DCG is taken
    over the same candidate set as ``reranked_docs``. Documents missing
    from ``doc_relevance_scores`` are treated as relevance 0. Returns
    0.0 when ideal DCG is zero (no relevant documents).

    Args:
        reranked_docs: Document identifiers in ranked order (best first).
        doc_relevance_scores: Ground-truth relevance grades keyed by
            document id. Each grade must be an integer in ``[0, 3]``.

    Returns:
        NDCG@10 in ``[0.0, 1.0]``.

    Raises:
        ValueError: If any relevance grade is outside ``[0, 3]``.
    """
    _validate_relevance_grades(doc_relevance_scores)
    relevances = [doc_relevance_scores.get(doc_id, 0) for doc_id in reranked_docs]
    dcg = _dcg_at_k(relevances, NDCG_AT_K)
    idcg = _dcg_at_k(sorted(relevances, reverse=True), NDCG_AT_K)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg
