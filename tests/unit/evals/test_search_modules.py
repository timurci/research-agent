"""Unit tests for search eval module registry and query adapters."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from pydantic import HttpUrl

from evals.search.agents import search_agent
from evals.search.modules import (
    MODULE_NAMES,
    SEARCH_SAMPLE_LIMIT,
    as_query_task,
    build_modules,
    query_module,
)
from evals.search.scorers import search_query_scorers
from research_agent.search.models import PaperInfo, ResearchQuery
from research_agent.shared.config.models import LMConfig

_ABSTRACT = (
    "A sufficiently long abstract describing the research methodology, "
    "experimental setup, results, and conclusions of this work in detail "
    "to satisfy the PaperInfo min_length=200 invariant enforced by Pydantic."
)

_SEARCH = LMConfig(model="openai/test-search")
_RERANK = LMConfig(model="infinity/test-rerank")


def test_module_names() -> None:
    assert frozenset({"search-search"}) == MODULE_NAMES


def test_build_modules_names_and_factories() -> None:
    modules = build_modules(
        search_lm_config=_SEARCH,
        rerank_lm_config=_RERANK,
    )
    assert set(modules) == MODULE_NAMES
    for name, module in modules.items():
        assert module.name == name
        assert callable(module.load_data)
        assert callable(module.build_task)
        assert callable(module.build_scorers)


def test_build_modules_sets_sample_limits() -> None:
    modules = build_modules(
        search_lm_config=_SEARCH,
        rerank_lm_config=_RERANK,
    )
    assert modules["search-search"].sample_limit == SEARCH_SAMPLE_LIMIT


def test_build_modules_injects_lm_configs() -> None:
    modules = build_modules(
        search_lm_config=_SEARCH,
        rerank_lm_config=_RERANK,
    )
    with (
        patch("evals.search.modules.search_agent") as search_factory,
        patch("evals.search.modules.search_query_scorers") as scorers_factory,
    ):
        search_factory.return_value = object()
        scorers_factory.return_value = ()

        modules["search-search"].build_task()
        modules["search-search"].build_scorers()

    search_factory.assert_called_once_with(
        lm_config=_SEARCH,
        instructions_path=None,
    )
    scorers_factory.assert_called_once_with(lm_config=_RERANK)


def test_build_modules_forwards_search_instructions() -> None:
    instructions = {"search-search": Path("data/optimize/output/search-search.json")}
    modules = build_modules(
        search_lm_config=_SEARCH,
        rerank_lm_config=_RERANK,
        instructions=instructions,
    )
    with (
        patch("evals.search.modules.search_agent") as search_factory,
        patch("evals.search.modules.search_query_scorers") as scorers_factory,
    ):
        search_factory.return_value = object()
        scorers_factory.return_value = ()

        modules["search-search"].build_task()
        modules["search-search"].build_scorers()

    search_factory.assert_called_once_with(
        lm_config=_SEARCH,
        instructions_path=instructions["search-search"],
    )
    scorers_factory.assert_called_once_with(lm_config=_RERANK)


def test_query_module_binds_name() -> None:
    module = query_module("custom", lambda: search_agent(lm_config=_SEARCH))
    assert module.name == "custom"
    assert callable(module.build_task)
    assert module.build_scorers is search_query_scorers


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
