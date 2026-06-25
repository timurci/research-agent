"""Factory for the search-index tool suite.

Wraps each search tool class in a `dspy.Tool` with a stable snake_case name.
Tool classes are plain async callables; this module is the only place that
knows about DSPy.
"""

import dspy

from research_agent.infrastructure.llm.tools.search.semantic_scholar import (
    SemanticScholarSearch,
)


def build_search_tools(
    *,
    s2_api_key: str | None = None,
) -> list[dspy.Tool]:
    """Return the configured search-index tool suite.

    Args:
        s2_api_key: Optional Semantic Scholar API key. Unauthenticated traffic
            shares a global 1,000 req/s pool; an authenticated key is
            recommended for any non-interactive use.

    Returns:
        A list of `dspy.Tool` instances, one per configured search index.
    """
    return [
        dspy.Tool(
            SemanticScholarSearch(api_key=s2_api_key),
            name="semantic_scholar_search",
        )
    ]
