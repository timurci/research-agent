"""Live network tests for the arXiv search tool.

Search-slice live test. Hits the real arXiv API. Tagged ``live`` and
skipped by default (see root ``conftest.py``); run explicitly with
``uv run pytest -m live``.
"""

from __future__ import annotations

import asyncio

import pytest

from research_agent.search.models import SearchIndexType
from research_agent.search.tools import LiteratureSearch

_TIMEOUT_SECONDS: float = 30.0


@pytest.mark.live
@pytest.mark.asyncio
async def test_search_finds_bert_paper() -> None:
    """Search returns the BERT paper with expected fields populated."""
    tool = LiteratureSearch()

    results = await asyncio.wait_for(
        tool(
            SearchIndexType.ARXIV,
            "BERT: Pre-training of Deep Bidirectional Transformers",
            limit=5,
        ),
        timeout=_TIMEOUT_SECONDS,
    )

    assert len(results) > 0
    sample = results[0]
    assert sample.paper.title is not None
    assert "BERT" in sample.paper.title
    authors = sample.paper.authors
    assert any("Devlin" in a for a in authors), f"Authors: {authors}"
    assert sample.paper.source.open_access is True
