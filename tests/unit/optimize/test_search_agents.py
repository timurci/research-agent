"""Unit tests for search optimize agents and session isolation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import dspy
import pytest
from pydantic import HttpUrl

from optimize.search.agents import SearchProgram, relevance_labeler, search_agent
from research_agent.search.agents import Reranker
from research_agent.search.models import ResearchQuery
from research_agent.shared.agent import LMConfig

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


def test_search_program_is_dspy_module_with_react() -> None:
    program = SearchProgram(lm_config=_SEARCH_CONFIG)
    assert isinstance(program, dspy.Module)
    names = [name for name, _ in program.named_predictors()]
    assert names
    assert any(name.startswith("react") for name in names)


@pytest.mark.asyncio
async def test_search_agent_delegates_to_search_program() -> None:
    query = ResearchQuery(text="quantum error correction codes")
    program = MagicMock(
        return_value=dspy.Prediction(search_results=[]),
    )

    with patch("optimize.search.agents.SearchProgram", return_value=program) as cls:
        agent = search_agent(lm_config=_SEARCH_CONFIG)
        result = await agent(query)

    cls.assert_called_once()
    assert cls.call_args.kwargs["lm_config"] is _SEARCH_CONFIG
    program.assert_called_once_with(research_query=query)
    assert result == []


@pytest.mark.asyncio
async def test_search_agent_defaults_to_yaml_search_role() -> None:
    yaml_config = LMConfig(model="openai/from-yaml-search")
    program = MagicMock(return_value=dspy.Prediction(search_results=[]))

    with (
        patch(_LOAD_LM_CONFIG, return_value=yaml_config) as load,
        patch("optimize.search.agents.SearchProgram", return_value=program) as cls,
    ):
        agent = search_agent()
        await agent(ResearchQuery(text="default config check query text"))

    load.assert_called_once_with("search-search")
    assert cls.call_args.kwargs["lm_config"] is yaml_config


@pytest.mark.asyncio
async def test_search_agent_uses_injected_lm_config() -> None:
    custom = LMConfig(
        model="openai/custom-search",
        api_key="search-key",
        base_url=HttpUrl("http://search.example/v1"),
    )
    program = MagicMock(return_value=dspy.Prediction(search_results=[]))

    with patch("optimize.search.agents.SearchProgram", return_value=program) as cls:
        agent = search_agent(lm_config=custom)
        await agent(ResearchQuery(text="injected config check query text"))

    assert cls.call_args.kwargs["lm_config"] is custom


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
