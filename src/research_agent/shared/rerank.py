# Copyright (c) 2026 Timur Çakmakoğlu

"""Provider-dispatch rerank client.

Layer: Infrastructure.

litellm's rerank coverage is limited (cohere, azure_ai, infinity, jina_ai,
hosted_vllm, huggingface, deepinfra, nvidia_nim, together_ai, bedrock), so
rerank models named with an ``openrouter/`` prefix are routed to the
OpenRouter SDK instead. ``build_rerank_client`` is the single dispatch
point; consumers depend on the ``RerankClient`` protocol only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypedDict

import litellm
from openrouter import OpenRouter
from openrouter.components import ProviderPreferences
from openrouter.operations import DocumentRequest
from openrouter.types import UNSET

if TYPE_CHECKING:
    from research_agent.shared.config.models import LMConfig

_OPENROUTER_PREFIX: str = "openrouter/"


class RerankScore(TypedDict):
    """Relevance score of a reranked document."""

    index: int
    relevance_score: float


class RerankClient(Protocol):
    """Provider-agnostic document rerank client.

    Model, credentials, and provider preferences are bound at
    construction; ``rerank`` exposes only the rerank payload.
    """

    async def rerank(self, *, query: str, documents: list[str]) -> list[RerankScore]:
        """Rerank *documents* against *query* into relevance scores."""
        raise NotImplementedError


class RerankError(Exception):
    """Raised when a rerank provider returns an unexpected response."""


class _LiteLLMRerankClient:
    """Rerank client backed by ``litellm.arerank``."""

    def __init__(self, config: LMConfig) -> None:
        self._model = config.model
        self._api_key = config.api_key
        self._api_base = str(config.base_url) if config.base_url else None
        self._provider_config = config.provider_config

    async def rerank(self, *, query: str, documents: list[str]) -> list[RerankScore]:
        ranking = await litellm.arerank(
            model=self._model,
            api_key=self._api_key,
            api_base=self._api_base,
            query=query,
            documents=documents,
            extra_body=self._provider_config,
        )
        return [
            {"index": item["index"], "relevance_score": item["relevance_score"]}
            for item in ranking.results
        ]


class _OpenRouterRerankClient:
    """Rerank client backed by the OpenRouter SDK.

    The ``openrouter/`` prefix on the configured model name is stripped
    before the SDK call.
    """

    def __init__(self, config: LMConfig) -> None:
        self._model = config.model.removeprefix(_OPENROUTER_PREFIX)
        self._provider = (
            ProviderPreferences(**config.provider_config)
            if config.provider_config
            else UNSET
        )
        self._client = OpenRouter(
            api_key=config.api_key,
            server_url=str(config.base_url) if config.base_url else None,
        )

    async def rerank(self, *, query: str, documents: list[str]) -> list[RerankScore]:
        response = await self._client.rerank.rerank_async(
            model=self._model,
            query=query,
            documents=[DocumentRequest(text=doc) for doc in documents],
            provider=self._provider,
        )
        if isinstance(response, str):
            msg = f"unexpected OpenRouter rerank response: {response}"
            raise RerankError(msg)
        return [
            {"index": item.index, "relevance_score": item.relevance_score}
            for item in response.results
        ]


def build_rerank_client(config: LMConfig) -> RerankClient:
    """Build the rerank client for *config*.

    Models named with an ``openrouter/`` prefix are served by the
    OpenRouter SDK; all other models go to litellm.
    """
    if config.model.startswith(_OPENROUTER_PREFIX):
        return _OpenRouterRerankClient(config)
    return _LiteLLMRerankClient(config)
