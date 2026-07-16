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


def search_result_non_duplicate(search_results: list[PaperInfo]) -> EvaluationScore:
    """Evaluates whether there are any duplicate search results based on title."""
    title_counter = Counter()
    for result in search_results:
        title_counter[result.title] += 1
    duplicated_titles = [title for title, count in title_counter.items() if count > 1]
    if duplicated_titles:
        reason = (
            "Duplicate papers found (same title)."
            " Results must be unique by title. Duplicate titles:\n"
            f"{'\n'.join(duplicated_titles)}"
        )
        return EvaluationScore(passing=False, reason=reason, score=0.0)
    return EvaluationScore(passing=True, reason="No duplicate papers found.", score=1.0)


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
