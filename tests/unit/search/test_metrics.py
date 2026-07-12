"""Unit tests for the search evaluation metrics.

Exercises the three pure domain metric functions:
``search_result_relevance``, ``search_result_non_hallucination``, and
``search_result_non_duplicate``. No network, no LLM, no async.
"""

from __future__ import annotations

import pytest
from pydantic import HttpUrl

from research_agent.search.metrics import (
    RelevanceMetric,
    search_result_non_duplicate,
    search_result_non_hallucination,
    search_result_relevance,
)
from research_agent.search.models import (
    PaperInfo,
    PaperSource,
    SearchIndexReference,
    SearchIndexType,
    SearchResult,
)
from research_agent.shared.metric import EvaluationScore, ToolCallObservation

_ABSTRACT = (
    "A sufficiently long abstract describing the research methodology, "
    "experimental setup, results, and conclusions of this work in detail "
    "to satisfy the PaperInfo min_length=200 invariant enforced by Pydantic."
)

_TITLE_A = "Alpha Paper On Quantum Computing Advances"
_TITLE_B = "Beta Paper On Neural Network Optimization"
_TITLE_C = "Gamma Paper On Climate Modeling Techniques"
_TITLE_D = "Delta Paper On Genomic Sequence Analysis"


def _make_paper(title: str = _TITLE_A) -> PaperInfo:
    return PaperInfo(
        source=PaperSource(
            url=HttpUrl("https://example.com/paper"),
            open_access=False,
        ),
        title=title,
        abstract=_ABSTRACT,
        authors=("Alice Smith",),
    )


def _make_result(paper: PaperInfo | None = None) -> SearchResult:
    return SearchResult(
        paper=paper or _make_paper(),
        search_index_reference=(
            SearchIndexReference(index=SearchIndexType.ARXIV, id="1234.5678"),
        ),
    )


def _make_observation(
    results: list[SearchResult] | None,
    *,
    tool_name: str = "LiteratureSearch",
) -> ToolCallObservation[list[SearchResult]]:
    obs: ToolCallObservation[list[SearchResult]] = ToolCallObservation(
        tool_name=tool_name,
        call_args={"query": "q", "limit": 5},
        observation=results,
    )
    return obs


def test_search_result_relevance_raises_on_length_mismatch() -> None:
    results = [_make_result()]
    score = RelevanceMetric(value=0.5)
    with pytest.raises(ValueError, match="must have the same length"):
        search_result_relevance(results, [score, score])


def test_search_result_relevance_empty_input_returns_failure() -> None:
    score = search_result_relevance([], [])
    assert score == EvaluationScore(passing=False, reason="Empty input", score=0.0)


def test_search_result_relevance_length_mismatch_takes_precedence_over_empty() -> None:
    with pytest.raises(ValueError, match="must have the same length"):
        search_result_relevance([_make_result()], [])


@pytest.mark.parametrize(
    ("values", "expected_passing", "expected_verdict", "expected_score"),
    [
        ([0.7, 0.8, 0.9, 1.0], True, "Results are mostly high relevance.", 1.0),
        ([0.3, 0.4, 0.5, 0.6], True, "Results are within acceptable bounds.", 0.5),
        ([0.0, 0.1, 0.2, 0.29], False, "Too many low relevance results.", 0.0),
        (
            [0.9, 0.8, 0.5, 0.4, 0.1],
            True,
            "Results are within acceptable bounds.",
            0.6,
        ),
        (
            [0.9, 0.8, 0.7, 0.1],
            True,
            "Results are mostly high relevance.",
            0.75,
        ),
        (
            [0.9, 0.8, 0.1, 0.1],
            False,
            "Too many low relevance results.",
            0.5,
        ),
        (
            [0.9, 0.8, 0.5, 0.5],
            True,
            "Results are within acceptable bounds.",
            0.75,
        ),
        (
            [0.9, 0.8, 0.7, 0.5],
            True,
            "Results are mostly high relevance.",
            0.875,
        ),
    ],
)
def test_search_result_relevance_cases(
    values: list[float],
    expected_passing: bool,  # noqa: FBT001  # parametrize row passes (values, passing, verdict, score) positionally
    expected_verdict: str,
    expected_score: float,
) -> None:
    results = [
        _make_result(_make_paper(f"Paper Number {i:03d}")) for i in range(len(values))
    ]
    scores = [RelevanceMetric(value=v) for v in values]
    score = search_result_relevance(results, scores)
    assert score.passing is expected_passing
    assert score.score == pytest.approx(expected_score)
    assert expected_verdict in score.reason


@pytest.mark.parametrize(
    ("value", "tier"),
    [
        (0.7, "HIGH"),
        (0.3, "MEDIUM"),
        (0.2999, "LOW"),
    ],
)
def test_search_result_relevance_threshold_inclusion(value: float, tier: str) -> None:
    results = [_make_result(_make_paper(f"Paper Number {int(value * 1000):03d}"))]
    score = search_result_relevance(results, [RelevanceMetric(value=value)])
    if tier == "HIGH":
        assert score.score == 1.0
        assert "Results are mostly high relevance." in score.reason
    elif tier == "MEDIUM":
        assert score.score == 0.5
        assert "Results are within acceptable bounds." in score.reason
    else:
        assert score.passing is False
        assert "Too many low relevance results." in score.reason


def test_search_result_relevance_reason_groups_titles_by_tier() -> None:
    results = [
        _make_result(_make_paper(_TITLE_A)),
        _make_result(_make_paper(_TITLE_B)),
        _make_result(_make_paper(_TITLE_C)),
    ]
    score = search_result_relevance(
        results,
        [
            RelevanceMetric(value=0.1),
            RelevanceMetric(value=0.5),
            RelevanceMetric(value=0.9),
        ],
    )
    assert "Verdict" in score.reason
    assert "Low relevance" in score.reason
    assert "High relevance" in score.reason
    assert _TITLE_A in score.reason
    assert _TITLE_C in score.reason
    assert _TITLE_B not in score.reason.split("High relevance", 1)[1]


def test_search_result_non_hallucination_passes_when_agent_subset_of_tool() -> None:
    paper = _make_paper()
    result = _make_result(paper)
    tool_results = [_make_result(_make_paper(_TITLE_A)), _make_result(paper)]
    score = search_result_non_hallucination([result], [_make_observation(tool_results)])
    assert score.passing is True
    assert score.score == 1.0
    assert score.reason == "Result contains no hallucinated search result."


def test_search_result_non_hallucination_fails_when_agent_has_extra_paper() -> None:
    agent_paper = _make_paper(_TITLE_A)
    tool_paper = _make_paper(_TITLE_B)
    tool_results = [_make_result(tool_paper)]
    score = search_result_non_hallucination(
        [_make_result(agent_paper)],
        [_make_observation(tool_results)],
    )
    assert score.passing is False
    assert score.score == 0.0
    assert score.reason.startswith("Hallucinated paper details found for papers:")
    assert _TITLE_A in score.reason
    assert _TITLE_B not in score.reason


def test_search_result_non_hallucination_passes_when_tool_has_unused_papers() -> None:
    tool_results = [
        _make_result(_make_paper(_TITLE_A)),
        _make_result(_make_paper(_TITLE_B)),
    ]
    agent_paper = _make_paper(_TITLE_A)
    score = search_result_non_hallucination(
        [_make_result(agent_paper)],
        [_make_observation(tool_results)],
    )
    assert score.passing is True
    assert score.score == 1.0


def test_search_result_non_hallucination_passes_when_agent_returns_nothing() -> None:
    tool_results = [
        _make_result(_make_paper(_TITLE_A)),
        _make_result(_make_paper(_TITLE_B)),
    ]
    score = search_result_non_hallucination([], [_make_observation(tool_results)])
    assert score.passing is True
    assert score.score == 1.0


def test_search_result_non_hallucination_ignores_non_literature_search_tool_calls() -> (
    None
):
    agent_paper = _make_paper(_TITLE_A)
    search_observation = [_make_result(agent_paper)]
    unrelated_observation = [_make_result(_make_paper(_TITLE_B))]
    score = search_result_non_hallucination(
        [_make_result(agent_paper)],
        [
            _make_observation(unrelated_observation, tool_name="OtherTool"),
            _make_observation(search_observation, tool_name="LiteratureSearch"),
        ],
    )
    assert score.passing is True
    assert score.score == 1.0


def test_search_result_non_hallucination_ignores_observations_with_none() -> None:
    agent_paper = _make_paper(_TITLE_A)
    search_observation = [_make_result(agent_paper)]
    score = search_result_non_hallucination(
        [_make_result(agent_paper)],
        [
            _make_observation(None),
            _make_observation(search_observation),
        ],
    )
    assert score.passing is True
    assert score.score == 1.0


def test_search_result_non_hallucination_unions_multiple_tool_calls() -> None:
    paper = _make_paper(_TITLE_A)
    score = search_result_non_hallucination(
        [_make_result(paper)],
        [
            _make_observation([_make_result(_make_paper(_TITLE_B))]),
            _make_observation([_make_result(paper)]),
        ],
    )
    assert score.passing is True
    assert score.score == 1.0


def test_search_result_non_hallucination_passes_on_empty_inputs() -> None:
    score = search_result_non_hallucination([], [])
    assert score.passing is True
    assert score.score == 1.0


@pytest.mark.parametrize(
    ("titles", "expected_passing", "expected_reason_contains"),
    [
        ([], True, ["No duplicate search results found."]),
        (
            [_TITLE_A, _TITLE_B, _TITLE_C, _TITLE_D],
            True,
            ["No duplicate search results found."],
        ),
        ([_TITLE_A, _TITLE_B, _TITLE_A], False, ["Duplicate titles:", _TITLE_A]),
        (
            [_TITLE_A, _TITLE_B, _TITLE_A, _TITLE_C, _TITLE_B],
            False,
            ["Duplicate titles:", _TITLE_A, _TITLE_B],
        ),
    ],
)
def test_search_result_non_duplicate(
    titles: list[str],
    expected_passing: bool,  # noqa: FBT001  # parametrize row passes (titles, passing, reason) positionally
    expected_reason_contains: list[str],
) -> None:
    results = [_make_result(_make_paper(title)) for title in titles]
    score = search_result_non_duplicate(results)
    assert score.passing is expected_passing
    if expected_passing:
        assert score.score == 1.0
    else:
        assert score.score == 0.0
    for fragment in expected_reason_contains:
        assert fragment in score.reason
