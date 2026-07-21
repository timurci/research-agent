"""Unit tests for the eval harness sampling."""

from __future__ import annotations

from evals.harness import EVAL_SEED, sample_rows


def test_sample_rows_returns_all_rows_without_limit() -> None:
    rows = list(range(10))
    assert sample_rows(rows, limit=None, seed=EVAL_SEED) == rows


def test_sample_rows_returns_all_rows_under_limit() -> None:
    rows = list(range(10))
    assert sample_rows(rows, limit=10, seed=EVAL_SEED) == rows
    assert sample_rows(rows, limit=25, seed=EVAL_SEED) == rows


def test_sample_rows_caps_to_limit_in_dataset_order() -> None:
    rows = list(range(100))
    sampled = sample_rows(rows, limit=10, seed=EVAL_SEED)

    assert len(sampled) == 10
    assert all(row in rows for row in sampled)
    assert sampled == sorted(sampled)


def test_sample_rows_is_deterministic_for_same_seed() -> None:
    rows = list(range(100))
    first = sample_rows(rows, limit=10, seed=7)
    second = sample_rows(rows, limit=10, seed=7)

    assert first == second
