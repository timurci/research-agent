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

LOW_RELEVANCE_THRESHOLD = 0.9
HIGH_RELEVANCE_SCORE = 1.0
MAX_LOW_RELEVANCE_FRACTION = 0.05
NDCG_AT_K = 10
MIN_RESULT_COUNT = 10
PEAK_RESULT_COUNT = 30
MAX_RESULT_COUNT = 100
MIN_RELEVANCE_GRADE = 0
MAX_RELEVANCE_GRADE = 3
SUGGESTION_MIN_WORDS = 150
SUGGESTION_MAX_WORDS = 500
SUGGESTION_QUALITY_SCORES: frozenset[float] = frozenset({0.0, 0.5, 1.0})


class SuggestionQualityBandError(Exception):
    """Raised when a quality score is not an allowed rubric band value."""


class RelevanceMetric(BaseModel):
    """Represents a relevance score."""

    model_config = ConfigDict(frozen=True)

    value: float = Field(..., ge=0.0, le=1.0)


def search_result_relevance(
    search_results: list[PaperInfo], relevance_scores: list[RelevanceMetric]
) -> EvaluationScore:
    """Score whether search results are relevant to the research topic.

    Each label is high when ``value >= LOW_RELEVANCE_THRESHOLD`` and low
    otherwise. The continuous score is the fraction of high-relevance
    papers. ``passing`` is false when the low-relevance fraction exceeds
    ``MAX_LOW_RELEVANCE_FRACTION``.

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
        if score.value < LOW_RELEVANCE_THRESHOLD:
            counter["LOW"] += 1
            low_relevance_items.append(i)
        else:
            counter["HIGH"] += 1
            cumulative_metric_score += HIGH_RELEVANCE_SCORE
            high_relevance_items.append(i)

    total = len(relevance_scores)
    metric_score = cumulative_metric_score / total
    low_fraction = counter["LOW"] / total

    if low_fraction > MAX_LOW_RELEVANCE_FRACTION:
        passing = False
        verdict = "Too many low relevance results."
    else:
        passing = True
        verdict = "Results meet the relevance threshold."

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

    Tent map over ``[MIN_RESULT_COUNT, MAX_RESULT_COUNT]`` peaking at
    ``PEAK_RESULT_COUNT``:

    - Linear rise from 0 at the minimum to 1.0 at the peak.
    - Linear fall from 1.0 at the peak to 0 at the maximum.
    - 0.0 outside the pass band.
    """
    if count < MIN_RESULT_COUNT or count > MAX_RESULT_COUNT:
        return 0.0
    if count <= PEAK_RESULT_COUNT:
        return (count - MIN_RESULT_COUNT) / (PEAK_RESULT_COUNT - MIN_RESULT_COUNT)
    return (MAX_RESULT_COUNT - count) / (MAX_RESULT_COUNT - PEAK_RESULT_COUNT)


def search_result_count(search_results: list[PaperInfo]) -> EvaluationScore:
    """Score search result volume with a peak-preferring tent map.

    Uses the integer count ``n = len(search_results)``:

    - ``passing`` is true when
      ``MIN_RESULT_COUNT <= n <= MAX_RESULT_COUNT``.
    - Continuous ``score`` rises linearly from 0 at
      ``MIN_RESULT_COUNT`` to 1.0 at ``PEAK_RESULT_COUNT``, then falls
      linearly to 0 at ``MAX_RESULT_COUNT``. Outside the band the score
      is 0.0.

    Pair with ``search_result_relevance`` so optimizers cannot trade
    volume for junk: volume and relevance are separate criteria.
    Session storage deduplicates by ``PaperInfo`` equality, so a
    separate non-duplicate metric is not required.

    Args:
        search_results: Papers returned by the search agent or workflow.

    Returns:
        Evaluation score whose continuous value is the tent map and
        whose pass/fail reflects the hard count band.
    """
    count = len(search_results)
    score = _volume_score(count)
    passing = MIN_RESULT_COUNT <= count <= MAX_RESULT_COUNT
    if count < MIN_RESULT_COUNT:
        reason = (
            f"Returned {count} results; "
            f"need at least {MIN_RESULT_COUNT} to pass "
            f"(peak {PEAK_RESULT_COUNT}, max {MAX_RESULT_COUNT})."
        )
    elif count > MAX_RESULT_COUNT:
        reason = (
            f"Returned {count} results; "
            f"need at most {MAX_RESULT_COUNT} to pass "
            f"(min {MIN_RESULT_COUNT}, peak {PEAK_RESULT_COUNT})."
        )
    elif count == PEAK_RESULT_COUNT:
        reason = (
            f"Returned {count} results "
            f"(peak {PEAK_RESULT_COUNT} met, "
            f"band [{MIN_RESULT_COUNT}, {MAX_RESULT_COUNT}])."
        )
    else:
        reason = (
            f"Returned {count} results "
            f"(peak {PEAK_RESULT_COUNT}, "
            f"band [{MIN_RESULT_COUNT}, {MAX_RESULT_COUNT}])."
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


def suggestion_length(suggestion: str) -> EvaluationScore:
    """Score suggestion length against the product word-count band.

    Word count is whitespace-tokenized via ``str.split()`` (no language-
    specific tokenization). Empty or whitespace-only text is 0 words.

    - ``passing`` is true when
      ``SUGGESTION_MIN_WORDS <= n <= SUGGESTION_MAX_WORDS``.
    - ``score`` is ``1.0`` inside the band, ``0.0`` when too short, and
      ``SUGGESTION_MAX_WORDS / n`` when over the max so longer overshoots
      degrade continuously within ``[0, 1]``.

    Args:
        suggestion: Free-text research-direction suggestion.

    Returns:
        Evaluation score for length compliance.
    """
    word_count = len(suggestion.split())
    if word_count < SUGGESTION_MIN_WORDS:
        return EvaluationScore(
            passing=False,
            reason="Suggestion is too short, it's likely to be insufficient.",
            score=0.0,
        )
    if word_count <= SUGGESTION_MAX_WORDS:
        return EvaluationScore(
            passing=True,
            reason="Suggestion has adequate length.",
            score=1.0,
        )
    return EvaluationScore(
        passing=False,
        reason=(
            "Suggestion is too verbose, it likely contains too much detail "
            "and hurts readability."
        ),
        score=SUGGESTION_MAX_WORDS / word_count,
    )


def suggestion_quality(
    *,
    score: float,
    passing: bool,
    reason: str,
) -> EvaluationScore:
    """Map a rubric-judge verdict to a domain evaluation score.

    Continuous quality scores follow the suggestion-quality rubric bands
    and must be exactly one of ``SUGGESTION_QUALITY_SCORES``
    (``0.0``, ``0.5``, ``1.0``). Pass/fail is taken from the judge's
    fail-condition assessment, not from comparing *score* to a threshold.

    Args:
        score: Discrete quality score from the rubric judge.
        passing: ``False`` when any rubric fail condition applies.
        reason: Human-readable rationale from the judge.

    Returns:
        Domain evaluation score for suggestion quality.

    Raises:
        SuggestionQualityBandError: If *score* is not an allowed rubric
            band value.
    """
    if score not in SUGGESTION_QUALITY_SCORES:
        msg = (
            f"suggestion quality score must be one of "
            f"{sorted(SUGGESTION_QUALITY_SCORES)}, got {score}"
        )
        raise SuggestionQualityBandError(msg)
    return EvaluationScore(passing=passing, reason=reason, score=score)
