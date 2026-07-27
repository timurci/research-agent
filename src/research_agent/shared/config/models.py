"""Pydantic models for Infrastructure configuration values.

Layer: Infrastructure.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, HttpUrl


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
