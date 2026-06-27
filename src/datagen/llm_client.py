"""Thin wrapper around dspy.LM for raw completions.

DSPy uses LiteLLM under the hood, so the model string follows LiteLLM
conventions (e.g. "openai/gpt-4o-mini", "anthropic/claude-haiku").
"""

from __future__ import annotations

import dspy

from datagen.errors import LLMContractError


class LLMClient:
    """A minimal client for raw LLM text completions."""

    def __init__(self, model: str, api_key: str) -> None:
        """Initialise the client with a LiteLLM-compatible model string.

        Args:
            model: Model identifier in LiteLLM format
                (e.g. ``"openai/gpt-4o-mini"``,
                ``"anthropic/claude-haiku"``).
            api_key: API key for the model provider.
        """
        self._lm = dspy.LM(model, api_key=api_key)

    def complete(self, prompt: str) -> str:
        """Return a single raw text completion for the prompt."""
        responses = self._lm(prompt=prompt)
        if not responses:
            msg = "LLM returned no completion"
            raise LLMContractError(msg)
        return responses[0]
