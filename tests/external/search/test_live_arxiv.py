"""Live network tests for the arXiv search tool.

Search-slice live test. Hits the real arXiv API. Tagged ``live`` and
skipped by default (see root ``conftest.py``); run explicitly with
``uv run pytest -m live``.
"""

from __future__ import annotations

import asyncio

import pytest

from research_agent.search.tools import ArXivSearch

_TIMEOUT_SECONDS: float = 30.0


@pytest.mark.live
@pytest.mark.asyncio
async def test_search_finds_bert_paper() -> None:
    """Search returns the BERT paper with expected fields populated."""
    tool = ArXivSearch()

    results = await asyncio.wait_for(
        tool("BERT: Pre-training of Deep Bidirectional Transformers"),
        timeout=_TIMEOUT_SECONDS,
    )

    assert len(results) > 0
    sample = results[0]
    assert sample.title is not None
    assert "BERT" in sample.title
    assert any("Devlin" in a for a in sample.authors), f"Authors: {sample.authors}"
    assert sample.is_open_access is True
