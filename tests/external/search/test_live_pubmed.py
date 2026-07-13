"""Live network tests for the PubMed search tool.

Search-slice live test. Hits the real NCBI E-utilities API. Tagged
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
async def test_search_finds_brca1_papers() -> None:
    """Search returns BRCA1-mutation papers with expected fields populated."""
    tool = LiteratureSearch()

    results = await asyncio.wait_for(
        tool(SearchIndexType.PUBMED, "BRCA1 mutation cancer[Title/Abstract]", limit=5),
        timeout=_TIMEOUT_SECONDS,
    )

    assert len(results) > 0
    sample = results[0]
    assert sample.title is not None
    assert len(sample.title) > 10
    assert len(sample.authors) > 0
    assert sample.url is not None
    assert "pubmed.ncbi.nlm.nih.gov" in str(sample.url)
