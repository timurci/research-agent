"""Live network tests for the OpenAlex search tool.

Search-slice live test. Hits the real OpenAlex Works API. Tagged
``live`` and skipped by default (see root ``conftest.py``); run
explicitly with ``uv run pytest -m live``.
"""

from __future__ import annotations

import asyncio

import pytest

from research_agent.search.models import SearchIndex
from research_agent.search.tools import LiteratureSearch

_TIMEOUT_SECONDS: float = 30.0


@pytest.mark.live
@pytest.mark.asyncio
async def test_search_finds_results() -> None:
    """Search returns papers with expected fields populated."""
    tool = LiteratureSearch()

    results = await asyncio.wait_for(
        tool(SearchIndex.OPENALEX, "CRISPR gene editing", limit=5),
        timeout=_TIMEOUT_SECONDS,
    )

    assert len(results) > 0
    sample = results[0]
    assert len(sample.title) > 10
    assert len(sample.abstract) >= 200
    assert len(sample.authors) > 0
    assert sample.url is not None
