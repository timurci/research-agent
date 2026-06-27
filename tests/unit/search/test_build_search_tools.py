"""Unit tests for ``build_search_tools``.

Verifies the DSPy tool factory returns the configured tool suite.
Async ``__call__`` paths and the normalisation chain are covered by
``tests/unit/search/test_{arxiv,pubmed,crossref,semantic_scholar}.py``
(pure inputs) and ``tests/external/search/test_live_*.py`` (real APIs).
"""

from __future__ import annotations

from dspy import Tool

from research_agent.search.tools import build_search_tools


def test_build_search_tools_returns_four_tools() -> None:
    tools = build_search_tools()

    assert len(tools) == 4
    names = {t.name for t in tools}
    assert names == {
        "semantic_scholar_search",
        "arxiv_search",
        "pubmed_search",
        "crossref_search",
    }


def test_build_search_tools_instances_are_tools() -> None:
    tools = build_search_tools()

    for t in tools:
        assert isinstance(t, Tool)


def test_build_search_tools_with_api_keys() -> None:
    tools = build_search_tools(s2_api_key="test_s2", pubmed_api_key="test_pubmed")

    assert len(tools) == 4
