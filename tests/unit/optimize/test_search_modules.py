"""Unit tests for search optimize module registry."""

from __future__ import annotations

from unittest.mock import patch

from optimize.search.modules import (
    MODULE_NAMES,
    SEARCH_SAMPLE_LIMIT,
    SEARCH_TRAIN_FRACTION,
    build_modules,
    sample_examples,
    split_train_val,
)
from research_agent.shared.agent import LMConfig

_SEARCH = LMConfig(model="openai/test-search")
_RERANK = LMConfig(model="infinity/test-rerank")


def test_module_names() -> None:
    assert frozenset({"search-search"}) == MODULE_NAMES


def test_default_sample_plan_is_50_with_80_20() -> None:
    assert SEARCH_SAMPLE_LIMIT == 50
    assert SEARCH_TRAIN_FRACTION == 0.8


def test_build_modules_names_and_factories() -> None:
    modules = build_modules(
        search_lm_config=_SEARCH,
        labeler_lm_config=_RERANK,
    )
    assert set(modules) == MODULE_NAMES
    assert "search-e2e" not in modules
    for name, module in modules.items():
        assert module.name == name
        assert callable(module.load_trainset)
        assert callable(module.metric)
        assert callable(module.build_student)
        assert module.sample_limit == SEARCH_SAMPLE_LIMIT
        assert module.train_fraction == SEARCH_TRAIN_FRACTION


def test_build_modules_binds_labeler_lm_config() -> None:
    with patch("optimize.search.modules.search_query_metric") as metric_factory:
        metric_factory.return_value = object()
        build_modules(
            search_lm_config=_SEARCH,
            labeler_lm_config=_RERANK,
        )

    metric_factory.assert_called_once_with(lm_config=_RERANK)


def test_build_modules_injects_student_lm_config() -> None:
    modules = build_modules(
        search_lm_config=_SEARCH,
        labeler_lm_config=_RERANK,
    )
    with patch("optimize.search.modules.SearchProgram") as student_cls:
        student_cls.return_value = object()
        modules["search-search"].build_student()

    student_cls.assert_called_once_with(lm_config=_SEARCH)


def test_sample_examples_respects_limit_and_seed() -> None:
    examples = list(range(10))
    sampled = sample_examples(examples, limit=3, seed=1)
    assert len(sampled) == 3
    assert sampled == sorted(sampled)
    assert sample_examples(examples, limit=3, seed=1) == sampled
    assert sample_examples(examples, limit=None, seed=1) == examples


def test_split_train_val_50_is_40_and_10() -> None:
    pool = list(range(50))
    train, val = split_train_val(pool, train_fraction=0.8, seed=0)
    assert len(train) == 40
    assert len(val) == 10
    assert sorted(train + val) == pool
    assert set(train).isdisjoint(set(val))


def test_split_train_val_is_deterministic() -> None:
    pool = list(range(50))
    assert split_train_val(pool, train_fraction=0.8, seed=7) == split_train_val(
        pool,
        train_fraction=0.8,
        seed=7,
    )
    assert split_train_val(pool, train_fraction=0.8, seed=7) != split_train_val(
        pool,
        train_fraction=0.8,
        seed=8,
    )


def test_split_train_val_single_example_duplicates() -> None:
    train, val = split_train_val([42], train_fraction=0.8, seed=0)
    assert train == [42]
    assert val == [42]
