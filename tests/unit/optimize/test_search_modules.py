"""Unit tests for search optimize module registry."""

from __future__ import annotations

from unittest.mock import patch

from optimize.search.modules import (
    MODULE_NAMES,
    SEARCH_SAMPLE_LIMIT,
    build_modules,
    sample_examples,
)
from research_agent.shared.agent import LMConfig

_SEARCH = LMConfig(model="openai/test-search")
_RERANK = LMConfig(model="infinity/test-rerank")


def test_module_names() -> None:
    assert frozenset({"search-search"}) == MODULE_NAMES


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
        assert callable(module.build_metric)
        assert callable(module.build_student)
        assert module.sample_limit == SEARCH_SAMPLE_LIMIT


def test_build_modules_injects_labeler_lm_config() -> None:
    modules = build_modules(
        search_lm_config=_SEARCH,
        labeler_lm_config=_RERANK,
    )
    with patch("optimize.search.modules.search_query_metric") as metric_factory:
        metric_factory.return_value = object()
        modules["search-search"].build_metric()

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
