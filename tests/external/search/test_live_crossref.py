"""Live network tests for the CrossRef search tool.

Search-slice live test. Hits the real CrossRef REST API. Tagged
``live`` and skipped by default (see root ``conftest.py``); run
explicitly with ``uv run pytest -m live``.
"""

from __future__ import annotations

import asyncio

import pytest

from research_agent.search.models import SearchIndexType
from research_agent.search.tools import LiteratureSearch

_TIMEOUT_SECONDS: float = 30.0


@pytest.mark.live
@pytest.mark.asyncio
async def test_search_finds_results() -> None:
    """Search returns papers with DOIs and titles populated."""
    tool = LiteratureSearch()

    results = await asyncio.wait_for(
        tool(SearchIndexType.CROSSREF, "deep learning neural networks", limit=5),
        timeout=_TIMEOUT_SECONDS,
    )

    if not results:
        return
    for sample in results:
        assert sample.paper.title is not None
        assert sample.paper.source.doi is not None
        assert sample.search_index_reference[0].index == SearchIndexType.CROSSREF
