"""Unit tests for search eval agents and session isolation."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from pydantic import HttpUrl

from evals.search.agents import paper_search_workflow, reranker, search_agent
from research_agent.search.agents import Reranker
from research_agent.search.models import ResearchQuery
from research_agent.search.workflows import PaperSearchWorkflow
from research_agent.shared.config.models import LMConfig
from research_agent.shared.session import InMemorySession

if TYPE_CHECKING:
    from research_agent.search.models import PaperInfo

_SEARCH_CONFIG = LMConfig(
    model="openai/test-search",
    api_key="search-key",
    base_url=HttpUrl("http://search.example/v1"),
)
_RERANK_CONFIG = LMConfig(
    model="infinity/test-rerank",
    api_key="rerank-key",
    base_url=HttpUrl("http://rerank.example/v1"),
)


def test_search_agent_is_callable() -> None:
    assert callable(search_agent(lm_config=_SEARCH_CONFIG))


def test_reranker_returns_reranker() -> None:
    assert isinstance(reranker(lm_config=_RERANK_CONFIG), Reranker)


def test_paper_search_workflow_returns_workflow() -> None:
    workflow = paper_search_workflow(
        search_lm_config=_SEARCH_CONFIG,
        rerank_lm_config=_RERANK_CONFIG,
    )
    assert isinstance(workflow, PaperSearchWorkflow)


@pytest.mark.asyncio
async def test_search_agent_builds_new_session_per_call() -> None:
    agent = search_agent(lm_config=_SEARCH_CONFIG)
    query = ResearchQuery(text="quantum error correction codes")
    constructed: list[tuple[object, object, object, object]] = []

    class _StubSearchAgent:
        def __init__(
            self,
            lm_config: object,
            session: object,
            lit_search: object,
            *,
            instructions_path: object = None,
        ) -> None:
            constructed.append((lm_config, session, lit_search, instructions_path))

        async def __call__(self, data: ResearchQuery) -> list[PaperInfo]:
            assert data is query
            return []

    with patch("evals.search.agents.SearchAgent", _StubSearchAgent):
        first = await agent(query)
        second = await agent(query)

    assert first == []
    assert second == []
    assert len(constructed) == 2
    assert constructed[0][1] is not constructed[1][1]
    assert isinstance(constructed[0][1], InMemorySession)
    assert isinstance(constructed[1][1], InMemorySession)
    assert constructed[0][2] is constructed[1][2]
    assert constructed[0][3] is None
    assert constructed[1][3] is None


@pytest.mark.asyncio
async def test_search_agent_defaults_to_yaml_search_role() -> None:
    captured: list[tuple[LMConfig, object]] = []
    yaml_config = LMConfig(model="openai/from-yaml-search")

    class _StubSearchAgent:
        def __init__(
            self,
            lm_config: LMConfig,
            _session: object,
            _lit_search: object,
            *,
            instructions_path: object = None,
        ) -> None:
            captured.append((lm_config, instructions_path))

        async def __call__(self, _data: ResearchQuery) -> list[PaperInfo]:
            return []

    with (
        patch("evals.search.agents.load_lm_config", return_value=yaml_config) as load,
        patch("evals.search.agents.SearchAgent", _StubSearchAgent),
    ):
        agent = search_agent()
        await agent(ResearchQuery(text="default config check query text"))

    load.assert_called_once_with("search-search")
    assert captured == [(yaml_config, None)]


@pytest.mark.asyncio
async def test_search_agent_uses_injected_lm_config() -> None:
    custom = LMConfig(
        model="openai/custom-search",
        api_key="search-key",
        base_url=HttpUrl("http://search.example/v1"),
    )
    captured: list[tuple[LMConfig, object]] = []

    class _StubSearchAgent:
        def __init__(
            self,
            lm_config: LMConfig,
            _session: object,
            _lit_search: object,
            *,
            instructions_path: object = None,
        ) -> None:
            captured.append((lm_config, instructions_path))

        async def __call__(self, _data: ResearchQuery) -> list[PaperInfo]:
            return []

    with patch("evals.search.agents.SearchAgent", _StubSearchAgent):
        agent = search_agent(lm_config=custom)
        await agent(ResearchQuery(text="injected config check query text"))

    assert len(captured) == 1
    assert captured[0][0] is custom
    assert captured[0][1] is None


def test_reranker_defaults_to_yaml_rerank_role() -> None:
    yaml_config = LMConfig(model="infinity/from-yaml-rerank")
    with (
        patch("evals.search.agents.load_lm_config", return_value=yaml_config) as load,
        patch("evals.search.agents.Reranker") as reranker_cls,
    ):
        reranker_cls.return_value = object()
        reranker()

    load.assert_called_once_with("search-rerank")
    assert reranker_cls.call_args.args[0] is yaml_config


def test_reranker_uses_injected_lm_config() -> None:
    custom = LMConfig(
        model="infinity/custom-rerank",
        api_key="rerank-key",
        base_url=HttpUrl("http://rerank.example/v1"),
    )

    with patch("evals.search.agents.Reranker") as reranker_cls:
        reranker_cls.return_value = object()
        reranker(lm_config=custom)

    assert reranker_cls.call_args.args[0] is custom


@pytest.mark.asyncio
async def test_paper_search_workflow_forwards_lm_configs() -> None:
    search_config = LMConfig(
        model="openai/workflow-search",
        api_key="search-key",
        base_url=HttpUrl("http://search.example/v1"),
    )
    rerank_config = LMConfig(
        model="infinity/workflow-rerank",
        api_key="rerank-key",
        base_url=HttpUrl("http://rerank.example/v1"),
    )
    captured_search: list[tuple[LMConfig, object]] = []

    class _StubSearchAgent:
        def __init__(
            self,
            lm_config: LMConfig,
            _session: object,
            _lit_search: object,
            *,
            instructions_path: object = None,
        ) -> None:
            captured_search.append((lm_config, instructions_path))

        async def __call__(self, _data: ResearchQuery) -> list[PaperInfo]:
            return []

    with (
        patch("evals.search.agents.SearchAgent", _StubSearchAgent),
        patch("evals.search.agents.Reranker") as reranker_cls,
    ):
        reranker_cls.return_value = object()
        workflow = paper_search_workflow(
            search_lm_config=search_config,
            rerank_lm_config=rerank_config,
        )
        assert isinstance(workflow, PaperSearchWorkflow)
        await workflow(ResearchQuery(text="workflow lm config forward check"))

    assert captured_search == [(search_config, None)]
    assert reranker_cls.call_args.args[0] is rerank_config
