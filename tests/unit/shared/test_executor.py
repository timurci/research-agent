"""Unit tests for the shared executor pool."""

from __future__ import annotations

import pytest

from research_agent.shared.executor import run_async


@pytest.mark.asyncio
async def test_run_async_executes_blocking_callable() -> None:
    def blocking_add(a: int, b: int) -> int:
        return a + b

    result = await run_async(blocking_add, 3, 4)

    assert result == 7
