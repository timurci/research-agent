"""Shared agent interface.

Layer: Application.
"""

from typing import Any, Protocol

from openai import BaseModel
from pydantic import ConfigDict, HttpUrl


class Agent[InputT, OutputT](Protocol):
    """Protocol for an AI agent."""

    async def __call__(self, data: InputT) -> OutputT:
        """Call the agent with the given input data."""
        raise NotImplementedError


class LMConfig(BaseModel):
    """Language model configuration.

    Assuming LiteLLM conventions in model names.

    ``provider_config`` is an optional provider-specific request body
    (for example OpenRouter routing). Infrastructure adapters forward it
    to LiteLLM as ``extra_body``.
    """

    model_config = ConfigDict(frozen=True)

    model: str
    api_key: str | None = None
    base_url: HttpUrl | None = None
    provider_config: dict[str, Any] | None = None
