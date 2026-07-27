"""Live network tests for the search agent.

Search-slice live test. Hits the local LFM language model and the real
PubMed and CrossRef APIs through the ``LiteratureSearch`` tool.
Tagged ``live`` and skipped by default (see root ``conftest.py``); run
explicitly with ``uv run pytest -m live``.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from research_agent.search.agents import SearchAgent
from research_agent.search.models import ResearchQuery
from research_agent.search.tools import LiteratureSearch
from research_agent.shared.session import InMemorySession

if TYPE_CHECKING:
    from research_agent.shared.config.models import LMConfig

_TIMEOUT_SECONDS: float = 30.0


def _make_agent(search_lm_config: LMConfig) -> SearchAgent:
    session = InMemorySession()
    return SearchAgent(search_lm_config, session, LiteratureSearch())


@pytest.mark.live
@pytest.mark.asyncio
async def test_search_agent_finds_bert_paper(search_lm_config: LMConfig) -> None:
    """Search returns BERT-related results via the ReAct agent."""
    agent = _make_agent(search_lm_config)
    query = ResearchQuery(
        text="BERT Pre-training of Deep Bidirectional Transformers",
    )

    results = await asyncio.wait_for(
        agent(query),
        timeout=_TIMEOUT_SECONDS,
    )

    if results:
        assert any("BERT" in r.title for r in results)
        assert any("Devlin" in a for r in results for a in r.authors)


@pytest.mark.live
@pytest.mark.asyncio
async def test_search_agent_handles_domains_scoped_query(
    search_lm_config: LMConfig,
) -> None:
    """Search accepts a query with domains and returns relevant results."""
    agent = _make_agent(search_lm_config)
    query = ResearchQuery(
        text="transformer architecture attention mechanism",
        domains=("machine learning",),
    )

    results = await asyncio.wait_for(
        agent(query),
        timeout=_TIMEOUT_SECONDS,
    )

    if results:
        assert any(
            "transformer" in r.title.lower() or "attention" in r.title.lower()
            for r in results
        )
