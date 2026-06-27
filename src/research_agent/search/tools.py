"""Search-index tools for the research agent.

Layer: Infrastructure.

Wraps synchronous SDKs for arXiv, PubMed/NCBI, and CrossRef (plus the
existing async Semantic Scholar client) and normalises every response
into the domain ``SearchResult`` shape.  Each tool class is a plain async
callable with no awareness of DSPy; ``build_search_tools`` is the only
place in this slice that knows about DSPy.
"""

from __future__ import annotations

import re
from typing import Any

import arxiv
import dspy
from Bio import Entrez
from habanero import Crossref
from semanticscholar.AsyncSemanticScholar import AsyncSemanticScholar
from semanticscholar.Paper import Paper

from research_agent.search.models import (
    PaperReference,
    SearchIndexId,
    SearchIndexType,
    SearchResult,
)
from research_agent.shared.executor import run_async

_FIELDS: list[str] = list(Paper.SEARCH_FIELDS)
_MAX_LIMIT: int = 100
_MIN_LIMIT: int = 1

_DEFAULT_MAILTO: str = "research-agent@example.com"
_STRIP_JATS = re.compile(r"<[^>]+>")


class SemanticScholarSearch:
    """Search Semantic Scholar for academic papers matching a free-text query.

    Returns paper metadata including title, abstract, authors, year, venue,
    citation count, and open-access PDF link when available.
    """

    def __init__(self, *, api_key: str | None = None, timeout: int = 30) -> None:
        """Initialise the search tool with an async Semantic Scholar client.

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
            A list of normalised ``SearchResult`` objects, one per matched
            paper. May be shorter than *limit* (or empty) when the API
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


class ArXivSearch:
    """Search arXiv for academic papers matching a free-text query.

    Returns paper metadata including title, abstract, authors, category
    tags, and links to the abstract page and PDF.
    """

    def __init__(
        self,
        *,
        page_size: int = 100,
        delay_seconds: float = 3.0,
        num_retries: int = 3,
    ) -> None:
        """Initialise the arXiv search tool.

        Args:
            page_size: Results fetched per API request (max 2000).
            delay_seconds: Wait between consecutive API requests. arXiv
                asks users to respect a 3-second delay.
            num_retries: Number of times to retry a failing request.
        """
        self._page_size = page_size
        self._delay_seconds = delay_seconds
        self._num_retries = num_retries

    async def __call__(self, query: str, *, limit: int = 10) -> list[SearchResult]:
        """Search arXiv for papers matching a free-text query.

        Args:
            query: Plain-text search query. May include the same field
                prefixes as the arXiv API (``ti:``, ``au:``, ``abs:``,
                ``cat:``, ``all:``).
            limit: Maximum number of results to return. Defaults to 10.

        Returns:
            A list of normalised ``SearchResult`` objects, one per matched
            paper. May be shorter than *limit* (or empty) when the API
            returns fewer matches.
        """
        clamped = max(_MIN_LIMIT, min(limit, 2000))

        def _search() -> list[arxiv.Result]:
            client = arxiv.Client(
                page_size=self._page_size,
                delay_seconds=self._delay_seconds,
                num_retries=self._num_retries,
            )
            search = arxiv.Search(query=query, max_results=clamped)
            return list(client.results(search))

        results = await run_async(_search)
        return [self._to_search_result(r) for r in results]

    def _to_search_result(self, result: arxiv.Result) -> SearchResult:
        pdf_url = result.pdf_url or next(
            (link.href for link in (result.links or []) if link.title == "pdf"),
            None,
        )
        return SearchResult(
            title=result.title or None,
            abstract=result.summary or None,
            authors=[a.name for a in (result.authors or []) if a.name],
            reference=PaperReference(
                source=SearchIndexId(
                    index=SearchIndexType.ARXIV,
                    id=result.entry_id,
                ),
                doi=result.doi or None,
            ),
            url=result.entry_id,
            pdf_url=pdf_url,
            publication_year=(
                result.published.year
                if result.published and result.published.year > 1
                else None
            ),
            venue=result.journal_ref or result.comment or None,
            citation_count=None,
            is_open_access=True,
            topics=list(result.categories) if result.categories else None,
            tldr=None,
            raw_metadata={
                "primary_category": result.primary_category,
                "comment": result.comment,
                "journal_ref": result.journal_ref,
                "updated": result.updated.isoformat() if result.updated else None,
            },
        )


class PubMedSearch:
    """Search PubMed / NCBI for biomedical literature matching a free-text query.

    Returns paper metadata including title, abstract, authors, journal
    venue, and DOI when available.
    """

    def __init__(
        self,
        *,
        email: str = _DEFAULT_MAILTO,
        api_key: str | None = None,
    ) -> None:
        """Initialise the PubMed search tool.

        Args:
            email: Email address sent to NCBI for identification.  NCBI
                requires a ``tool`` or ``email`` value with every request.
            api_key: Optional NCBI API key. Without a key the rate limit
                is ~3 req/s; with a key it rises to 10 req/s.
        """
        object.__setattr__(Entrez, "email", email)
        if api_key is not None:
            object.__setattr__(Entrez, "api_key", api_key)

    async def __call__(self, query: str, *, limit: int = 10) -> list[SearchResult]:
        """Search PubMed for papers matching a free-text query.

        Args:
            query: Plain-text search query. Supports the full PubMed
                query syntax (MeSH terms, field tags, boolean operators).
            limit: Maximum number of results to return. Defaults to 10.

        Returns:
            A list of normalised ``SearchResult`` objects, one per matched
            paper. May be shorter than *limit* (or empty) when the API
            returns fewer matches.
        """
        clamped = max(_MIN_LIMIT, min(limit, 1000))

        def _search() -> list[dict[str, Any]]:
            handle = Entrez.esearch(db="pubmed", term=query, retmax=clamped)
            record = Entrez.read(handle)
            handle.close()
            id_list = record.get("IdList", [])
            if not id_list:
                return []

            ids = ",".join(id_list)
            handle = Entrez.efetch(
                db="pubmed",
                id=ids,
                rettype="abstract",
                retmode="xml",
            )
            records = Entrez.read(handle)
            handle.close()
            return records.get("PubmedArticle", [])

        articles = await run_async(_search)
        return [self._to_search_result(a) for a in articles]

    @staticmethod
    def _to_search_result(article: dict[str, Any]) -> SearchResult:
        med = article["MedlineCitation"]
        art = med["Article"]

        title = str(art.get("ArticleTitle", ""))
        abstract_parts = art.get("Abstract", {}).get("AbstractText", [])
        if isinstance(abstract_parts, list):
            abstract = " ".join(str(p) for p in abstract_parts)
        else:
            abstract = str(abstract_parts) if abstract_parts else ""

        author_list = art.get("AuthorList", [])
        authors: list[str] = []
        for a in author_list:
            last = a.get("LastName", "")
            fore = a.get("ForeName", "")
            collective = a.get("CollectiveName", "")
            if collective:
                authors.append(str(collective))
            elif last or fore:
                authors.append(f"{fore} {last}".strip())

        journal = art.get("Journal", {})
        venue = str(journal.get("Title", ""))
        pub_date = journal.get("JournalIssue", {}).get("PubDate", {})
        year_raw = pub_date.get("Year", "")
        publication_year = int(year_raw) if year_raw else None

        doi: str | None = None
        for eid in art.get("ELocationID", []):
            attrs = getattr(eid, "attributes", {}) or {}
            if attrs.get("EIdType") == "doi":
                doi = str(eid)
                break

        pmid = str(med.get("PMID", ""))

        return SearchResult(
            title=title or None,
            abstract=abstract or None,
            authors=authors,
            reference=PaperReference(
                source=SearchIndexId(
                    index=SearchIndexType.PUBMED,
                    id=pmid,
                ),
                doi=doi,
            ),
            url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            pdf_url=None,
            publication_year=publication_year,
            venue=venue or None,
            citation_count=None,
            is_open_access=None,
            topics=None,
            tldr=None,
            raw_metadata={
                "pmid": pmid,
                "publication_types": [
                    str(p) for p in art.get("PublicationTypeList", [])
                ],
            },
        )


class CrossRefSearch:
    """Search CrossRef for academic works matching a free-text query.

    Returns paper metadata including title, abstract (when deposited),
    authors, DOI, container title, and citation counts.  Best for
    DOI-centric and bibliographic lookup.
    """

    def __init__(self, *, mailto: str = _DEFAULT_MAILTO) -> None:
        """Initialise the CrossRef search tool.

        Args:
            mailto: Email address sent to the CrossRef polite pool.
                Required for reliable, unthrottled access.
        """
        self._mailto = mailto

    async def __call__(self, query: str, *, limit: int = 10) -> list[SearchResult]:
        """Search CrossRef for works matching a free-text query.

        Args:
            query: Plain-text search query.
            limit: Maximum number of results to return. Defaults to 10.

        Returns:
            A list of normalised ``SearchResult`` objects, one per matched
            work. May be shorter than *limit* (or empty) when the API
            returns fewer matches.
        """
        clamped = max(_MIN_LIMIT, min(limit, 1000))

        def _search() -> list[dict[str, Any]]:
            cr = Crossref(mailto=self._mailto)
            response = cr.works(query=query, limit=clamped)
            if not isinstance(response, dict):
                msg = f"CrossRef returned unexpected type {type(response).__name__}"
                raise TypeError(msg)
            return response.get("message", {}).get("items", [])

        items = await run_async(_search)
        return [self._to_search_result(item) for item in items]

    @staticmethod
    def _to_search_result(item: dict[str, Any]) -> SearchResult:
        title_list = item.get("title", [])
        title = title_list[0] if title_list else None

        raw_abstract = item.get("abstract", "")
        abstract = _STRIP_JATS.sub("", raw_abstract).strip() if raw_abstract else None

        author_list = item.get("author", [])
        authors: list[str] = []
        for a in author_list:
            given = a.get("given", "")
            family = a.get("family", "")
            name = f"{given} {family}".strip()
            if name:
                authors.append(name)

        doi = item.get("DOI")

        container = item.get("container-title", [])
        venue = container[0] if container else None

        pub = (
            item.get("published-print")
            or item.get("published-online")
            or item.get("issued")
            or item.get("created")
        )
        date_parts = pub.get("date-parts", [[None]]) if pub else [[None]]
        publication_year = (
            date_parts[0][0]
            if date_parts and date_parts[0] and date_parts[0][0]
            else None
        )

        resource_url = item.get("resource", {}).get("primary", {}).get("URL")

        return SearchResult(
            title=title,
            abstract=abstract,
            authors=authors,
            reference=PaperReference(
                source=SearchIndexId(
                    index=SearchIndexType.CROSSREF,
                    id=doi or "",
                ),
                doi=doi,
            ),
            url=item.get("URL"),
            pdf_url=resource_url,
            publication_year=publication_year,
            venue=venue,
            citation_count=item.get("is-referenced-by-count"),
            is_open_access=None,
            topics=None,
            tldr=None,
            raw_metadata={
                "type": item.get("type"),
                "publisher": item.get("publisher"),
                "issn": item.get("ISSN"),
            },
        )


def build_search_tools(
    *,
    s2_api_key: str | None = None,
    pubmed_api_key: str | None = None,
) -> list[dspy.Tool]:
    """Return the configured search-index tool suite.

    Args:
        s2_api_key: Optional Semantic Scholar API key. Unauthenticated
            traffic shares a global 1,000 req/s pool; an authenticated key
            is recommended for any non-interactive use.
        pubmed_api_key: Optional NCBI API key for elevated rate limits
            (10 req/s vs ~3 req/s unauthenticated).

    Returns:
        A list of ``dspy.Tool`` instances, one per configured search index.
    """
    return [
        dspy.Tool(
            SemanticScholarSearch(api_key=s2_api_key),
            name="semantic_scholar_search",
        ),
        dspy.Tool(
            ArXivSearch(),
            name="arxiv_search",
        ),
        dspy.Tool(
            PubMedSearch(api_key=pubmed_api_key),
            name="pubmed_search",
        ),
        dspy.Tool(
            CrossRefSearch(),
            name="crossref_search",
        ),
    ]
