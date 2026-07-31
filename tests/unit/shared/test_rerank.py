# Copyright (c) 2026 Timur Çakmakoğlu

"""Unit tests for ``research_agent.shared.rerank``."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from openrouter.components import ProviderPreferences
from openrouter.operations import DocumentRequest
from pydantic import HttpUrl

from research_agent.shared.config.models import LMConfig
from research_agent.shared.rerank import RerankError, build_rerank_client

_LITELLM_MODEL: str = "infinity/LFM2.5-ColBERT-350M"
_OPENROUTER_MODEL: str = "openrouter/cohere/rerank-x"


@pytest.fixture
def litellm_config() -> LMConfig:
    return LMConfig(
        model=_LITELLM_MODEL,
        api_key="litellm-key",
        base_url=HttpUrl("http://localhost:8080/v1"),
        provider_config={"provider": {"order": ["OpenAI"]}},
    )


@pytest.fixture
def openrouter_config() -> LMConfig:
    return LMConfig(
        model=_OPENROUTER_MODEL,
        api_key="or-key",
        base_url=HttpUrl("https://openrouter.ai/api/v1"),
        provider_config={"order": ["Cohere"], "allow_fallbacks": True},
    )


def test_factory_selects_litellm_client_for_plain_model(
    litellm_config: LMConfig,
) -> None:
    with patch("research_agent.shared.rerank._LiteLLMRerankClient") as client_cls:
        client = build_rerank_client(litellm_config)

    client_cls.assert_called_once_with(litellm_config)
    assert client is client_cls.return_value


def test_factory_selects_openrouter_client_for_openrouter_model(
    openrouter_config: LMConfig,
) -> None:
    with patch("research_agent.shared.rerank._OpenRouterRerankClient") as client_cls:
        client = build_rerank_client(openrouter_config)

    client_cls.assert_called_once_with(openrouter_config)
    assert client is client_cls.return_value


@pytest.mark.asyncio
async def test_litellm_client_delegates_and_normalizes(
    litellm_config: LMConfig,
) -> None:
    ranking = SimpleNamespace(
        results=[
            {"index": 1, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.2},
        ]
    )
    with patch(
        "research_agent.shared.rerank.litellm.arerank",
        new=AsyncMock(return_value=ranking),
    ) as mock_arerank:
        scores = await build_rerank_client(litellm_config).rerank(
            query="machine learning",
            documents=["doc-a", "doc-b"],
        )

    mock_arerank.assert_awaited_once_with(
        model=_LITELLM_MODEL,
        api_key="litellm-key",
        api_base="http://localhost:8080/v1",
        query="machine learning",
        documents=["doc-a", "doc-b"],
        extra_body={"provider": {"order": ["OpenAI"]}},
    )
    assert scores == [
        {"index": 1, "relevance_score": 0.9},
        {"index": 0, "relevance_score": 0.2},
    ]


@pytest.mark.asyncio
async def test_openrouter_client_strips_prefix_and_maps_provider(
    openrouter_config: LMConfig,
) -> None:
    response = SimpleNamespace(
        results=[
            SimpleNamespace(index=1, relevance_score=0.9),
            SimpleNamespace(index=0, relevance_score=0.2),
        ]
    )
    with patch("research_agent.shared.rerank.OpenRouter") as mock_client:
        mock_rerank_async = AsyncMock(return_value=response)
        mock_client.return_value.rerank.rerank_async = mock_rerank_async
        scores = await build_rerank_client(openrouter_config).rerank(
            query="machine learning",
            documents=["doc-a", "doc-b"],
        )

    mock_client.assert_called_once_with(
        api_key="or-key",
        server_url="https://openrouter.ai/api/v1",
    )
    mock_rerank_async.assert_awaited_once_with(
        model="cohere/rerank-x",
        query="machine learning",
        documents=[DocumentRequest(text="doc-a"), DocumentRequest(text="doc-b")],
        provider=ProviderPreferences(order=["Cohere"], allow_fallbacks=True),
    )
    assert scores == [
        {"index": 1, "relevance_score": 0.9},
        {"index": 0, "relevance_score": 0.2},
    ]


@pytest.mark.asyncio
async def test_openrouter_client_raises_on_non_json_response(
    openrouter_config: LMConfig,
) -> None:
    with patch("research_agent.shared.rerank.OpenRouter") as mock_client:
        mock_rerank_async = AsyncMock(return_value="not-json")
        mock_client.return_value.rerank.rerank_async = mock_rerank_async

        with pytest.raises(RerankError, match="unexpected OpenRouter rerank"):
            await build_rerank_client(openrouter_config).rerank(
                query="q",
                documents=["doc-a"],
            )
