"""Unit tests for search eval module registry and query adapters."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

from pydantic import HttpUrl

from evals.search.agents import search_agent, suggestion_generator
from evals.search.modules import (
    MODULE_NAMES,
    SEARCH_SAMPLE_LIMIT,
    SUGGEST_SAMPLE_LIMIT,
    as_query_task,
    as_suggest_task,
    build_modules,
    query_module,
    suggest_module,
)
from evals.search.scorers import search_query_scorers, search_suggest_scorers
from research_agent.search.models import PaperInfo, ResearchQuery
from research_agent.shared.config.models import LMConfig

if TYPE_CHECKING:
    from evals.harness import EvalModule

_ABSTRACT = (
    "A sufficiently long abstract describing the research methodology, "
    "experimental setup, results, and conclusions of this work in detail "
    "to satisfy the PaperInfo min_length=200 invariant enforced by Pydantic."
)

_SEARCH = LMConfig(model="openai/test-search")
_RERANK = LMConfig(model="infinity/test-rerank")
_SUGGEST = LMConfig(model="openai/test-suggest")
_JUDGE = LMConfig(model="openai/test-judge")


def _build() -> dict[str, EvalModule]:
    return build_modules(
        search_lm_config=_SEARCH,
        rerank_lm_config=_RERANK,
        suggest_lm_config=_SUGGEST,
        judge_lm_config=_JUDGE,
    )


def test_module_names() -> None:
    assert frozenset({"search-search", "search-suggest"}) == MODULE_NAMES


def test_build_modules_names_and_factories() -> None:
    modules = _build()
    assert set(modules) == MODULE_NAMES
    for name, module in modules.items():
        assert module.name == name
        assert callable(module.load_data)
        assert callable(module.build_task)
        assert callable(module.build_scorers)


def test_build_modules_sets_sample_limits() -> None:
    modules = _build()
    assert modules["search-search"].sample_limit == SEARCH_SAMPLE_LIMIT
    assert modules["search-suggest"].sample_limit == SUGGEST_SAMPLE_LIMIT


def test_build_modules_injects_lm_configs() -> None:
    modules = _build()
    with (
        patch("evals.search.modules.search_agent") as search_factory,
        patch("evals.search.modules.search_query_scorers") as scorers_factory,
        patch("evals.search.modules.suggestion_generator") as suggest_factory,
        patch("evals.search.modules.search_suggest_scorers") as suggest_scorers,
    ):
        search_factory.return_value = object()
        scorers_factory.return_value = ()
        suggest_factory.return_value = object()
        suggest_scorers.return_value = ()

        modules["search-search"].build_task()
        modules["search-search"].build_scorers()
        modules["search-suggest"].build_task()
        modules["search-suggest"].build_scorers()

    search_factory.assert_called_once_with(
        lm_config=_SEARCH,
        instructions_path=None,
    )
    scorers_factory.assert_called_once_with(lm_config=_RERANK)
    suggest_factory.assert_called_once_with(
        lm_config=_SUGGEST,
        instructions_path=None,
    )
    suggest_scorers.assert_called_once_with(lm_config=_JUDGE)


def test_build_modules_forwards_search_instructions() -> None:
    instructions = {
        "search-search": Path("data/optimize/output/search-search.json"),
        "search-suggest": Path("data/optimize/output/search-suggest.json"),
    }
    modules = build_modules(
        search_lm_config=_SEARCH,
        rerank_lm_config=_RERANK,
        suggest_lm_config=_SUGGEST,
        judge_lm_config=_JUDGE,
        instructions=instructions,
    )
    with (
        patch("evals.search.modules.search_agent") as search_factory,
        patch("evals.search.modules.search_query_scorers") as scorers_factory,
        patch("evals.search.modules.suggestion_generator") as suggest_factory,
        patch("evals.search.modules.search_suggest_scorers") as suggest_scorers,
    ):
        search_factory.return_value = object()
        scorers_factory.return_value = ()
        suggest_factory.return_value = object()
        suggest_scorers.return_value = ()

        modules["search-search"].build_task()
        modules["search-search"].build_scorers()
        modules["search-suggest"].build_task()
        modules["search-suggest"].build_scorers()

    search_factory.assert_called_once_with(
        lm_config=_SEARCH,
        instructions_path=instructions["search-search"],
    )
    scorers_factory.assert_called_once_with(lm_config=_RERANK)
    suggest_factory.assert_called_once_with(
        lm_config=_SUGGEST,
        instructions_path=instructions["search-suggest"],
    )
    suggest_scorers.assert_called_once_with(lm_config=_JUDGE)


def test_query_module_binds_name() -> None:
    module = query_module("custom", lambda: search_agent(lm_config=_SEARCH))
    assert module.name == "custom"
    assert callable(module.build_task)
    assert module.build_scorers is search_query_scorers


def test_suggest_module_binds_name() -> None:
    module = suggest_module(
        "custom-suggest",
        lambda: suggestion_generator(lm_config=_SUGGEST),
    )
    assert module.name == "custom-suggest"
    assert callable(module.build_task)
    assert module.build_scorers is search_suggest_scorers


def test_as_query_task_accepts_dataset_item() -> None:
    paper = PaperInfo(
        title="Alpha Paper On Quantum Computing Advances",
        abstract=_ABSTRACT,
        authors=("Alice Smith",),
        url=HttpUrl("https://example.com/paper"),
        open_access=False,
    )
    received: list[ResearchQuery] = []

    async def stub_agent(query: ResearchQuery) -> list[PaperInfo]:
        received.append(query)
        return [paper]

    task = as_query_task(stub_agent)
    query = ResearchQuery(text="quantum error correction codes")

    result = task({"query": query})
    assert result == {"papers": [paper]}
    assert received == [query]

    result2 = task({"query": {"text": "quantum error correction codes"}})
    assert result2 == {"papers": [paper]}
    assert received == [
        query,
        ResearchQuery(text="quantum error correction codes"),
    ]


def test_as_suggest_task_accepts_dataset_item() -> None:
    paper = PaperInfo(
        title="Alpha Paper On Quantum Computing Advances",
        abstract=_ABSTRACT,
        authors=("Alice Smith",),
        url=HttpUrl("https://example.com/paper"),
        open_access=False,
    )
    received: list[tuple[ResearchQuery, list[PaperInfo]]] = []

    async def stub_agent(data: tuple[ResearchQuery, list[PaperInfo]]) -> str:
        received.append(data)
        return "read the survey first"

    task = as_suggest_task(stub_agent)
    query = ResearchQuery(text="quantum error correction codes")
    result = task(
        {
            "query": query.model_dump(mode="json"),
            "papers": [paper.model_dump(mode="json")],
        },
    )

    assert result["suggestion"] == "read the survey first"
    assert result["query"]["text"] == query.text
    assert len(result["papers"]) == 1
    assert received == [(query, [paper])]
