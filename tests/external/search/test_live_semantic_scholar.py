"""Live network tests for the Semantic Scholar search tool.

Search-slice live test. Hits the real Semantic Scholar API. Subject to
flakiness, transient rate limits, and (for unauthenticated traffic) the
shared 1,000 req/s pool. Tagged ``live`` and skipped by default (see
root ``conftest.py``); run explicitly with ``uv run pytest -m live``.

The unit test in ``tests/unit/search/test_semantic_scholar.py`` covers
the normalization logic deterministically.
"""

from __future__ import annotations

import asyncio

import dspy
import pytest

from research_agent.search.tools import LiteratureSearch

_TIMEOUT_SECONDS: float = 60.0


@pytest.mark.live
@pytest.mark.asyncio
async def test_search_finds_bert_paper() -> None:
    """Search returns the BERT paper with expected fields populated."""
    tool = dspy.Tool(LiteratureSearch(), name="literature_search")

    results = await asyncio.wait_for(
        tool.acall(
            search_index="semantic_scholar",
            query="BERT: Pre-training of Deep Bidirectional Transformers",
            limit=5,
        ),
        timeout=_TIMEOUT_SECONDS,
    )

    sample = results[0]
    assert "Pre-training" in (sample.paper.title or "")
    assert "Deep Bidirectional Transformers" in (sample.paper.title or "")
    assert any("Devlin" in a for a in sample.paper.authors)
