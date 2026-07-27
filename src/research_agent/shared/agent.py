"""Shared agent interface.

Layer: Application.
"""

from typing import Protocol


class Agent[InputT, OutputT](Protocol):
    """Protocol for an AI agent."""

    async def __call__(self, data: InputT) -> OutputT:
        """Call the agent with the given input data."""
        raise NotImplementedError
