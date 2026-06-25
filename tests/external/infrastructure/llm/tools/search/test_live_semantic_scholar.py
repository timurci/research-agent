"""Live network tests for the Semantic Scholar search tool.

Hits the real Semantic Scholar API. Subject to flakiness, transient
rate limits, and (for unauthenticated traffic) the shared 1,000 req/s
pool. Run selectively with `uv run pytest tests/external/`.

The test is bounded by a short timeout and skips on any network failure
(rate limit, connection refused, DNS, etc.) so CI is never blocked by
upstream conditions. The unit test in `tests/unit/` covers the
normalization logic deterministically.
"""

from __future__ import annotations

import asyncio

import pytest

from research_agent.infrastructure.llm.tools.search import build_search_tools

_TIMEOUT_SECONDS: float = 15.0


@pytest.mark.asyncio
async def test_search_finds_bert_paper() -> None:
    tools = build_search_tools()
    tool = tools[0]

    try:
        results = await asyncio.wait_for(
            tool.acall(
                query="BERT: Pre-training of Deep Bidirectional Transformers",
                limit=5,
            ),
            timeout=_TIMEOUT_SECONDS,
        )
    except Exception as exc:  # noqa: BLE001  # any network failure should skip, not fail
        pytest.skip(
            f"semantic scholar unreachable or rate-limited: "
            f"{type(exc).__name__}: {str(exc)[:200]}"
        )

    sample = results[0]
    assert "Pre-training" in (sample.title or "")
    assert "Deep Bidirectional Transformers" in (sample.title or "")
    assert any("Devlin" in a for a in sample.authors)
