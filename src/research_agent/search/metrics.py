"""AI Evaluation metrics for search.

Layer: Domain.
"""

from collections import Counter
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from research_agent.shared.metric import EvaluationScore, ToolCallObservation

if TYPE_CHECKING:
    from research_agent.search.models import PaperInfo

HIGH_RELEVANCE_THRESHOLD = 0.7
MEDIUM_RELEVANCE_THRESHOLD = 0.3
HIGH_RELEVANCE_SCORE = 1.0
MEDIUM_RELEVANCE_SCORE = 0.5

SEARCH_TOOL_NAME = "LiteratureSearch"


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


def search_result_non_hallucination(
    search_results: list[PaperInfo],
    tool_calls: list[ToolCallObservation[list[PaperInfo]]],
) -> EvaluationScore:
    """Evaluate whether search results are present in tool observations.

    Observation entries must be plain ``PaperInfo`` values. Callers that
    collect ReAct traces must unwrap any index-tagged tool observation
    wrappers to ``PaperInfo`` before building ``ToolCallObservation``
    lists; wrappers are not accepted here.
    """
    agent_papers: set[PaperInfo] = set(search_results)
    tool_papers: set[PaperInfo] = {
        result
        for call in tool_calls
        if call.tool_name == SEARCH_TOOL_NAME and call.observation is not None
        for result in call.observation
    }

    hallucinated_papers = agent_papers - tool_papers

    if len(hallucinated_papers) > 0:
        titles = "\n".join([p.title for p in hallucinated_papers])
        reason = f"Hallucinated paper details found for papers:\n{titles}"
        return EvaluationScore(passing=False, reason=reason, score=0.0)

    return EvaluationScore(
        passing=True, reason="Result contains no hallucinated search result.", score=1.0
    )


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
