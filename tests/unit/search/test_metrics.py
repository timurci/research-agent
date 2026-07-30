"""Unit tests for the search evaluation metrics.

Exercises the pure domain metric functions:
``search_result_relevance``, ``search_result_count``, ``ndcg_at_10``,
and ``suggestion_length``.
No network, no LLM, no async.
"""

from __future__ import annotations

import math

import pytest
from pydantic import HttpUrl

from research_agent.search.metrics import (
    LOW_RELEVANCE_THRESHOLD,
    MAX_LOW_RELEVANCE_FRACTION,
    MAX_RESULT_COUNT,
    MIN_RESULT_COUNT,
    PEAK_RESULT_COUNT,
    SUGGESTION_MAX_WORDS,
    SUGGESTION_MIN_WORDS,
    RelevanceMetric,
    ndcg_at_10,
    search_result_count,
    search_result_relevance,
    suggestion_length,
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
        (
            [0.9, 0.95, 1.0, 0.91],
            True,
            "Results meet the relevance threshold.",
            1.0,
        ),
        (
            [0.5, 0.6, 0.7, 0.8],
            False,
            "Too many low relevance results.",
            0.0,
        ),
        (
            [0.0, 0.1, 0.2, 0.49],
            False,
            "Too many low relevance results.",
            0.0,
        ),
        (
            [1.0, 0.95, 0.9, 0.4],
            False,
            "Too many low relevance results.",
            0.75,
        ),
        (
            [0.9] * 19 + [0.4],
            True,
            "Results meet the relevance threshold.",
            0.95,
        ),
        (
            [0.9] * 18 + [0.4],
            False,
            "Too many low relevance results.",
            18 / 19,
        ),
        (
            [0.8999],
            False,
            "Too many low relevance results.",
            0.0,
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
        (LOW_RELEVANCE_THRESHOLD, "HIGH"),
        (LOW_RELEVANCE_THRESHOLD - 1e-9, "LOW"),
    ],
)
def test_search_result_relevance_threshold_inclusion(value: float, tier: str) -> None:
    results = [_make_paper(f"Paper Number {int(value * 1000):03d}")]
    score = search_result_relevance(results, [RelevanceMetric(value=value)])
    if tier == "HIGH":
        assert score.score == 1.0
        assert score.passing is True
        assert "Results meet the relevance threshold." in score.reason
    else:
        assert score.score == 0.0
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
    assert _TITLE_B in score.reason
    assert _TITLE_C in score.reason
    assert _TITLE_A not in score.reason.split("High relevance", 1)[1]
    assert _TITLE_B not in score.reason.split("High relevance", 1)[1]
    assert score.passing is False
    assert score.score == pytest.approx(1 / 3)
    assert MAX_LOW_RELEVANCE_FRACTION < 1 / 3


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


def _expected_volume_score(count: int) -> float:
    if count < MIN_RESULT_COUNT or count > MAX_RESULT_COUNT:
        return 0.0
    if count <= PEAK_RESULT_COUNT:
        return (count - MIN_RESULT_COUNT) / (PEAK_RESULT_COUNT - MIN_RESULT_COUNT)
    return (MAX_RESULT_COUNT - count) / (MAX_RESULT_COUNT - PEAK_RESULT_COUNT)


@pytest.mark.parametrize(
    ("count", "expected_passing"),
    [
        (0, False),
        (MIN_RESULT_COUNT - 1, False),
        (MIN_RESULT_COUNT, True),
        (20, True),
        (PEAK_RESULT_COUNT, True),
        (50, True),
        (MAX_RESULT_COUNT, True),
        (MAX_RESULT_COUNT + 1, False),
    ],
)
def test_search_result_count_steps(
    count: int,
    expected_passing: bool,  # noqa: FBT001  # parametrize row is (count, passing)
) -> None:
    results = [_make_paper(f"Paper Number {i:03d}") for i in range(count)]
    score = search_result_count(results)
    assert score.passing is expected_passing
    assert score.score == pytest.approx(_expected_volume_score(count))
    assert str(count) in score.reason
    assert str(PEAK_RESULT_COUNT) in score.reason


def test_search_result_count_peak_is_one() -> None:
    results = [_make_paper(f"Paper Number {i:03d}") for i in range(PEAK_RESULT_COUNT)]
    score = search_result_count(results)
    assert score.passing is True
    assert score.score == pytest.approx(1.0)


def test_search_result_count_rises_then_falls() -> None:
    below_peak = search_result_count(
        [_make_paper(f"Paper Number {i:03d}") for i in range(20)]
    )
    at_peak = search_result_count(
        [_make_paper(f"Paper Number {i:03d}") for i in range(PEAK_RESULT_COUNT)]
    )
    above_peak = search_result_count(
        [_make_paper(f"Paper Number {i:03d}") for i in range(50)]
    )
    assert below_peak.score < at_peak.score
    assert above_peak.score < at_peak.score
    assert below_peak.score == pytest.approx(
        (20 - MIN_RESULT_COUNT) / (PEAK_RESULT_COUNT - MIN_RESULT_COUNT)
    )
    assert above_peak.score == pytest.approx(
        (MAX_RESULT_COUNT - 50) / (MAX_RESULT_COUNT - PEAK_RESULT_COUNT)
    )


def test_search_result_count_band_endpoints_score_zero_but_pass() -> None:
    at_min = search_result_count(
        [_make_paper(f"Paper Number {i:03d}") for i in range(MIN_RESULT_COUNT)]
    )
    at_max = search_result_count(
        [_make_paper(f"Paper Number {i:03d}") for i in range(MAX_RESULT_COUNT)]
    )
    assert at_min.passing is True
    assert at_max.passing is True
    assert at_min.score == pytest.approx(0.0)
    assert at_max.score == pytest.approx(0.0)


def test_search_result_count_below_min_reason() -> None:
    score = search_result_count([])
    assert score == EvaluationScore(
        passing=False,
        reason=(
            f"Returned 0 results; need at least {MIN_RESULT_COUNT} to pass "
            f"(peak {PEAK_RESULT_COUNT}, max {MAX_RESULT_COUNT})."
        ),
        score=0.0,
    )


def test_search_result_count_above_max_reason() -> None:
    count = MAX_RESULT_COUNT + 5
    results = [_make_paper(f"Paper Number {i:03d}") for i in range(count)]
    score = search_result_count(results)
    assert score.passing is False
    assert score.score == 0.0
    assert f"need at most {MAX_RESULT_COUNT}" in score.reason
    assert f"min {MIN_RESULT_COUNT}" in score.reason
    assert f"peak {PEAK_RESULT_COUNT}" in score.reason


def test_search_result_count_pass_between_min_and_peak_reason() -> None:
    count = MIN_RESULT_COUNT + 5
    results = [_make_paper(f"Paper Number {i:03d}") for i in range(count)]
    score = search_result_count(results)
    assert score.passing is True
    assert score.score == pytest.approx(_expected_volume_score(count))
    assert f"peak {PEAK_RESULT_COUNT}" in score.reason
    assert f"band [{MIN_RESULT_COUNT}, {MAX_RESULT_COUNT}]" in score.reason


def test_search_result_count_meets_peak_reason() -> None:
    results = [_make_paper(f"Paper Number {i:03d}") for i in range(PEAK_RESULT_COUNT)]
    score = search_result_count(results)
    assert score.passing is True
    assert score.score == pytest.approx(1.0)
    assert f"peak {PEAK_RESULT_COUNT} met" in score.reason
    assert f"band [{MIN_RESULT_COUNT}, {MAX_RESULT_COUNT}]" in score.reason


def _words(count: int) -> str:
    return " ".join(f"w{i}" for i in range(count))


_OVER_BY_ONE = SUGGESTION_MAX_WORDS + 1
_OVER_BY_ONE_SCORE = SUGGESTION_MAX_WORDS / _OVER_BY_ONE


@pytest.mark.parametrize(
    ("text", "expected_passing", "expected_score"),
    [
        ("", False, 0.0),
        ("   \n\t  ", False, 0.0),
        ("one", False, 0.0),
        (_words(SUGGESTION_MIN_WORDS - 1), False, 0.0),
        (_words(SUGGESTION_MIN_WORDS), True, 1.0),
        (_words(SUGGESTION_MAX_WORDS), True, 1.0),
        (_words(_OVER_BY_ONE), False, _OVER_BY_ONE_SCORE),
        (_words(SUGGESTION_MAX_WORDS * 2), False, 0.5),
    ],
)
def test_suggestion_length(
    text: str,
    expected_passing: bool,  # noqa: FBT001  # parametrize row is (text, passing, score)
    expected_score: float,
) -> None:
    score = suggestion_length(text)
    assert score.passing is expected_passing
    assert score.score == pytest.approx(expected_score)


def test_suggestion_length_whitespace_tokenization() -> None:
    score = suggestion_length(_words(SUGGESTION_MIN_WORDS))
    assert score.passing is True
    assert score.score == 1.0


def test_suggestion_length_over_limit_reason() -> None:
    score = suggestion_length(_words(SUGGESTION_MAX_WORDS + 50))
    assert score.passing is False
    assert "verbose" in score.reason.lower()


def test_suggestion_length_too_short_reason() -> None:
    score = suggestion_length("")
    assert score == EvaluationScore(
        passing=False,
        reason="Suggestion is too short, it's likely to be insufficient.",
        score=0.0,
    )
