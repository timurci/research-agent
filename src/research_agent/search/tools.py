"""Search-index tools for the research agent.

Layer: Infrastructure.

Wraps the `semanticscholar` SDK's async client and normalizes the response
into the domain `SearchResult` shape. `SemanticScholarSearch` is a plain
async callable with no awareness of DSPy; `build_search_tools` is the only
place in this slice that knows about DSPy.
"""

from __future__ import annotations

import dspy
from semanticscholar.AsyncSemanticScholar import AsyncSemanticScholar
from semanticscholar.Paper import Paper

from research_agent.search.models import (
    PaperReference,
    SearchIndexId,
    SearchIndexType,
    SearchResult,
)

_FIELDS: list[str] = list(Paper.SEARCH_FIELDS)
_MAX_LIMIT: int = 100
_MIN_LIMIT: int = 1


class SemanticScholarSearch:
    """Search Semantic Scholar for academic papers matching a free-text query.

    Returns paper metadata including title, abstract, authors, year, venue,
    citation count, and open-access PDF link when available.
    """

    def __init__(self, *, api_key: str | None = None, timeout: int = 30) -> None:
        """Initialize the search tool with an async Semantic Scholar client.

        Args:
            api_key: Optional Semantic Scholar API key. Unauthenticated
                traffic shares a global 1,000 req/s pool; an authenticated
                key raises the per-IP rate to 1 req/s intro (more on
                request).
            timeout: Per-request timeout in seconds. Defaults to 30.
        """
        if api_key is None:
            self._client = AsyncSemanticScholar(timeout=timeout)
        else:
            self._client = AsyncSemanticScholar(api_key=api_key, timeout=timeout)

    async def __call__(self, query: str, *, limit: int = 10) -> list[SearchResult]:
        """Search Semantic Scholar for papers matching a free-text query.

        Args:
            query: Plain-text search query.
            limit: Maximum number of results to return. Clamped to the
                Semantic Scholar API range of [1, 100]; defaults to 10.

        Returns:
            A list of normalized `SearchResult` objects, one per matched
            paper. May be shorter than `limit` (or empty) when the API
            returns fewer matches.
        """
        clamped = max(_MIN_LIMIT, min(limit, _MAX_LIMIT))
        results = await self._client.search_paper(query, fields=_FIELDS, limit=clamped)
        if isinstance(results, Paper):
            return [self._to_search_result(results)]
        return [self._to_search_result(paper) for paper in results.items]

    def _to_search_result(self, paper: Paper) -> SearchResult:
        external = paper.externalIds or {}
        pdf = paper.openAccessPdf or {}
        return SearchResult(
            title=paper.title,
            abstract=paper.abstract,
            authors=[a.name for a in (paper.authors or []) if a.name],
            reference=PaperReference(
                source=SearchIndexId(
                    index=SearchIndexType.SEMANTIC_SCHOLAR,
                    id=paper.paperId,
                ),
                doi=external.get("DOI"),
            ),
            url=paper.url,
            pdf_url=pdf.get("url"),
            publication_year=paper.year,
            venue=paper.venue,
            citation_count=paper.citationCount,
            is_open_access=paper.isOpenAccess,
            topics=paper.fieldsOfStudy,
            tldr=paper.tldr.text if paper.tldr else None,
            raw_metadata=paper.raw_data,
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
