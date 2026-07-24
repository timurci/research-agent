"""Unit tests for search optimize agents and session isolation."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from pydantic import HttpUrl

from optimize.search.agents import relevance_labeler, search_agent
from research_agent.search.agents import Reranker
from research_agent.search.models import ResearchQuery
from research_agent.shared.agent import LMConfig
from research_agent.shared.session import InMemorySession

if TYPE_CHECKING:
    from research_agent.search.models import PaperInfo

_LOAD_LM_CONFIG = "optimize.search.agents.load_lm_config"

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


def test_relevance_labeler_returns_reranker() -> None:
    assert isinstance(relevance_labeler(lm_config=_RERANK_CONFIG), Reranker)


@pytest.mark.asyncio
async def test_search_agent_builds_new_session_per_call() -> None:
    agent = search_agent(lm_config=_SEARCH_CONFIG)
    query = ResearchQuery(text="quantum error correction codes")
    constructed: list[tuple[object, object, object]] = []

    class _StubSearchAgent:
        def __init__(
            self,
            lm_config: object,
            session: object,
            lit_search: object,
        ) -> None:
            constructed.append((lm_config, session, lit_search))

        async def __call__(self, data: ResearchQuery) -> list[PaperInfo]:
            assert data is query
            return []

    with patch("optimize.search.agents.SearchAgent", _StubSearchAgent):
        first = await agent(query)
        second = await agent(query)

    assert first == []
    assert second == []
    assert len(constructed) == 2
    assert constructed[0][1] is not constructed[1][1]
    assert isinstance(constructed[0][1], InMemorySession)
    assert isinstance(constructed[1][1], InMemorySession)
    assert constructed[0][2] is constructed[1][2]


@pytest.mark.asyncio
async def test_search_agent_defaults_to_yaml_search_role() -> None:
    captured: list[LMConfig] = []
    yaml_config = LMConfig(model="openai/from-yaml-search")

    class _StubSearchAgent:
        def __init__(
            self,
            lm_config: LMConfig,
            _session: object,
            _lit_search: object,
        ) -> None:
            captured.append(lm_config)

        async def __call__(self, _data: ResearchQuery) -> list[PaperInfo]:
            return []

    with (
        patch(_LOAD_LM_CONFIG, return_value=yaml_config) as load,
        patch("optimize.search.agents.SearchAgent", _StubSearchAgent),
    ):
        agent = search_agent()
        await agent(ResearchQuery(text="default config check query text"))

    load.assert_called_once_with("search-search")
    assert captured == [yaml_config]


@pytest.mark.asyncio
async def test_search_agent_uses_injected_lm_config() -> None:
    custom = LMConfig(
        model="openai/custom-search",
        api_key="search-key",
        base_url=HttpUrl("http://search.example/v1"),
    )
    captured: list[LMConfig] = []

    class _StubSearchAgent:
        def __init__(
            self,
            lm_config: LMConfig,
            _session: object,
            _lit_search: object,
        ) -> None:
            captured.append(lm_config)

        async def __call__(self, _data: ResearchQuery) -> list[PaperInfo]:
            return []

    with patch("optimize.search.agents.SearchAgent", _StubSearchAgent):
        agent = search_agent(lm_config=custom)
        await agent(ResearchQuery(text="injected config check query text"))

    assert len(captured) == 1
    assert captured[0] is custom


def test_relevance_labeler_defaults_to_yaml_rerank_role() -> None:
    yaml_config = LMConfig(model="infinity/from-yaml-rerank")
    with (
        patch(_LOAD_LM_CONFIG, return_value=yaml_config) as load,
        patch("optimize.search.agents.Reranker") as reranker_cls,
    ):
        reranker_cls.return_value = object()
        relevance_labeler()

    load.assert_called_once_with("search-rerank")
    assert reranker_cls.call_args.args[0] is yaml_config


def test_relevance_labeler_uses_injected_lm_config() -> None:
    custom = LMConfig(
        model="infinity/custom-rerank",
        api_key="rerank-key",
        base_url=HttpUrl("http://rerank.example/v1"),
    )

    with patch("optimize.search.agents.Reranker") as reranker_cls:
        reranker_cls.return_value = object()
        relevance_labeler(lm_config=custom)

    assert reranker_cls.call_args.args[0] is custom
