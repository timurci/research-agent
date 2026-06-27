"""Thin async wrapper around ``litellm.acompletion`` for raw text completions.

The model string follows LiteLLM conventions (e.g. ``"openai/gpt-4o-mini"``,
``"anthropic/claude-haiku"``).
"""

from __future__ import annotations

from typing import Any

import litellm

from datagen.errors import LLMContractError


class LLMClient:
    """A minimal async client for raw LLM text completions."""

    def __init__(
        self,
        model: str,
        api_key: str,
        *,
        reasoning_effort: str | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        """Initialise the client with a LiteLLM-compatible model string.

        Args:
            model: Model identifier in LiteLLM format
                (e.g. ``"openai/gpt-4o-mini"``,
                ``"anthropic/claude-haiku"``).
            api_key: API key for the model provider.
            reasoning_effort: Optional effort level forwarded to LiteLLM as
                ``reasoning_effort=...``. Supported by OpenAI o-series and
                gpt-5 (``"low" | "medium" | "high"``, ``"minimal"`` on
                gpt-5), Google Gemini (``"low" | "medium" | "high"``;
                ``"none"`` to disable), and OpenRouter models that expose
                reasoning. ``None`` leaves it at the provider default.
                Anthropic extended thinking is not exposed here — use
                ``extra_body`` for ``thinking={...}`` if needed.
            extra_body: Optional dict merged into the LiteLLM request body
                via ``extra_body=...``. The common use case is OpenRouter
                provider routing, for example pinning providers with
                fallbacks allowed:

                .. code-block:: python

                    {
                        "provider": {
                            "order": ["Anthropic", "OpenAI"],
                            "allow_fallbacks": True,
                        }
                    }

                Or pinning providers with fallbacks disabled:

                .. code-block:: python

                    {
                        "provider": {
                            "order": ["openai", "together"],
                            "allow_fallbacks": False,
                        }
                    }
        """
        self._model = model
        self._api_key = api_key
        self._reasoning_effort = reasoning_effort
        self._extra_body = extra_body

    async def complete(self, prompt: str) -> str:
        """Return a single raw text completion for the prompt."""
        response = await litellm.acompletion(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            api_key=self._api_key,
            reasoning_effort=self._reasoning_effort,
            extra_body=self._extra_body,
        )
        if not response.choices:
            msg = "LLM returned no completion"
            raise LLMContractError(msg)
        content = response.choices[0].message.content
        if not content:
            msg = "LLM returned no completion"
            raise LLMContractError(msg)
        return content
