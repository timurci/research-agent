"""Unit tests for the search evaluation metrics.

Exercises the pure domain metric functions:
``search_result_relevance``, ``search_result_non_duplicate``, and
``ndcg_at_10``. No network, no LLM, no async.
"""

from __future__ import annotations

import math

import pytest
from pydantic import HttpUrl

from research_agent.search.metrics import (
    RelevanceMetric,
    ndcg_at_10,
    search_result_non_duplicate,
    search_result_relevance,
)
from research_agent.search.models import PaperInfo
from research_agent.shared.metric import EvaluationScore

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
        title=title,
        abstract=_ABSTRACT,
        authors=("Alice Smith",),
        url=HttpUrl("https://example.com/paper"),
        open_access=False,
    )


def test_search_result_relevance_raises_on_length_mismatch() -> None:
    results = [_make_paper()]
    score = RelevanceMetric(value=0.5)
    with pytest.raises(ValueError, match="must have the same length"):
        search_result_relevance(results, [score, score])


def test_search_result_relevance_empty_input_returns_failure() -> None:
    score = search_result_relevance([], [])
    assert score == EvaluationScore(passing=False, reason="Empty input", score=0.0)


def test_search_result_relevance_length_mismatch_takes_precedence_over_empty() -> None:
    with pytest.raises(ValueError, match="must have the same length"):
        search_result_relevance([_make_paper()], [])


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
    results = [_make_paper(f"Paper Number {i:03d}") for i in range(len(values))]
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
    results = [_make_paper(f"Paper Number {int(value * 1000):03d}")]
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
        _make_paper(_TITLE_A),
        _make_paper(_TITLE_B),
        _make_paper(_TITLE_C),
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


@pytest.mark.parametrize(
    ("titles", "expected_passing", "expected_reason_contains"),
    [
        ([], True, ["No duplicate papers found."]),
        (
            [_TITLE_A, _TITLE_B, _TITLE_C, _TITLE_D],
            True,
            ["No duplicate papers found."],
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
    results = [_make_paper(title) for title in titles]
    score = search_result_non_duplicate(results)
    assert score.passing is expected_passing
    if expected_passing:
        assert score.score == 1.0
    else:
        assert score.score == 0.0
    for fragment in expected_reason_contains:
        assert fragment in score.reason


def _expected_dcg(relevances: list[int], k: int = 10) -> float:
    total = 0.0
    for i, rel in enumerate(relevances[:k]):
        total += (2**rel - 1) / math.log2(i + 2)
    return total


def _expected_ndcg(relevances: list[int], k: int = 10) -> float:
    dcg = _expected_dcg(relevances, k)
    idcg = _expected_dcg(sorted(relevances, reverse=True), k)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def test_ndcg_at_10_perfect_ranking_is_one() -> None:
    docs = ["a", "b", "c"]
    scores = {"a": 3, "b": 2, "c": 1}
    assert ndcg_at_10(docs, scores) == pytest.approx(1.0)


def test_ndcg_at_10_reversed_ranking_matches_formula() -> None:
    docs = ["c", "b", "a"]
    scores = {"a": 3, "b": 2, "c": 0}
    expected = _expected_ndcg([0, 2, 3])
    assert ndcg_at_10(docs, scores) == pytest.approx(expected)
    assert 0.0 < ndcg_at_10(docs, scores) < 1.0


def test_ndcg_at_10_empty_ranking_is_zero() -> None:
    assert ndcg_at_10([], {"a": 3}) == 0.0


def test_ndcg_at_10_all_zero_relevance_is_zero() -> None:
    docs = ["a", "b", "c"]
    scores = {"a": 0, "b": 0, "c": 0}
    assert ndcg_at_10(docs, scores) == 0.0


def test_ndcg_at_10_missing_label_treated_as_zero() -> None:
    docs = ["a", "b", "c"]
    scores = {"a": 3, "c": 1}
    expected = _expected_ndcg([3, 0, 1])
    assert ndcg_at_10(docs, scores) == pytest.approx(expected)


def test_ndcg_at_10_uses_only_top_ten() -> None:
    docs = [f"d{i}" for i in range(12)]
    scores = dict.fromkeys(docs, 0)
    scores["d0"] = 0
    scores["d9"] = 3
    scores["d10"] = 3
    scores["d11"] = 3
    expected = _expected_ndcg([scores[d] for d in docs], k=10)
    assert ndcg_at_10(docs, scores) == pytest.approx(expected)
    assert ndcg_at_10(docs, scores) < 1.0


def test_ndcg_at_10_ideal_uses_full_candidate_set_not_only_top_ten() -> None:
    docs = [f"d{i}" for i in range(12)]
    scores = dict.fromkeys(docs, 0)
    scores["d10"] = 3
    scores["d11"] = 2
    expected = _expected_ndcg([scores[d] for d in docs], k=10)
    assert ndcg_at_10(docs, scores) == pytest.approx(expected)
    assert ndcg_at_10(docs, scores) == 0.0


@pytest.mark.parametrize("bad_grade", [-1, 4, 10])
def test_ndcg_at_10_rejects_out_of_range_grades(bad_grade: int) -> None:
    with pytest.raises(ValueError, match=r"\[0, 3\]"):
        ndcg_at_10(["a"], {"a": bad_grade})
