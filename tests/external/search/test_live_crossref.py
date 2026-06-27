"""Live network tests for the CrossRef search tool.

Search-slice live test. Hits the real CrossRef REST API. Tagged
``live`` and skipped by default (see root ``conftest.py``); run
explicitly with ``uv run pytest -m live``.
"""

from __future__ import annotations

import asyncio

import pytest

from research_agent.search.tools import CrossRefSearch

_TIMEOUT_SECONDS: float = 30.0


@pytest.mark.live
@pytest.mark.asyncio
async def test_search_finds_results() -> None:
    """Search returns papers with DOIs and titles populated."""
    tool = CrossRefSearch()

    results = await asyncio.wait_for(
        tool("deep learning neural networks", limit=5),
        timeout=_TIMEOUT_SECONDS,
    )

    assert len(results) > 0
    for sample in results:
        assert sample.title is not None
        assert sample.reference.doi is not None
        assert sample.reference.source.index.value == "crossref"
