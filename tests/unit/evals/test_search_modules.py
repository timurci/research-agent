"""Unit tests for search eval module registry and query adapters."""

from __future__ import annotations

import pytest
from pydantic import HttpUrl

from evals.search.agents import search_agent
from evals.search.modules import MODULES, as_query_predict_fn, query_module
from evals.search.scorers import search_query_scorers
from research_agent.search.models import PaperInfo, ResearchQuery

_ABSTRACT = (
    "A sufficiently long abstract describing the research methodology, "
    "experimental setup, results, and conclusions of this work in detail "
    "to satisfy the PaperInfo min_length=200 invariant enforced by Pydantic."
)


def test_search_modules_registry() -> None:
    assert set(MODULES) == {"search", "search-e2e"}
    for name, module in MODULES.items():
        assert module.name == name
        assert module.build_scorers is search_query_scorers


def test_query_module_binds_name() -> None:
    module = query_module("custom", search_agent)
    assert module.name == "custom"
    assert callable(module.build_predict_fn)
    assert module.build_scorers is search_query_scorers


@pytest.mark.asyncio
async def test_as_query_predict_fn_accepts_model_and_mapping() -> None:
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

    predict_fn = as_query_predict_fn("test-predict", stub_agent)
    query = ResearchQuery(text="quantum error correction codes")

    assert await predict_fn(query) == [paper]
    assert await predict_fn({"text": "quantum error correction codes"}) == [paper]
    assert received == [
        query,
        ResearchQuery(text="quantum error correction codes"),
    ]
