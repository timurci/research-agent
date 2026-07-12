"""Search-index tools for the research agent.

Layer: Infrastructure.

Wraps synchronous SDKs for arXiv, PubMed/NCBI, and CrossRef (plus the
existing async Semantic Scholar client) and normalises every response
into the domain ``SearchResult`` shape.  ``LiteratureSearch`` is the
public surface: it is a single async callable that dispatches to one of
four private per-index handlers (``_SemanticScholarSearch``,
``_ArXivSearch``, ``_PubMedSearch``, ``_CrossRefSearch``) based on the
requested ``SearchIndexType``.  The private handlers carry the
index-specific implementation; ``LiteratureSearch`` is what the search
node wraps in ``dspy.Tool`` and exposes to the agent.
"""

from __future__ import annotations

import re
from typing import Any, Protocol

import arxiv
from Bio import Entrez
from habanero import Crossref
from pydantic import HttpUrl
from semanticscholar.AsyncSemanticScholar import AsyncSemanticScholar
from semanticscholar.Paper import Paper

from research_agent.search.models import (
    PaperInfo,
    PaperSource,
    SearchIndexReference,
    SearchIndexType,
    SearchResult,
)
from research_agent.shared.executor import run_async

_FIELDS: list[str] = list(Paper.SEARCH_FIELDS)
_MAX_LIMIT: int = 100
_MIN_LIMIT: int = 1

_DEFAULT_MAILTO: str = "research-agent@example.com"
_STRIP_JATS = re.compile(r"<[^>]+>")


class UnknownIndexError(Exception):
    """Raised when ``LiteratureSearch`` receives an unknown index."""


def _require_url(value: str | None, *, context: str) -> str:
    """Return *value* if non-empty, else raise ``ValueError`` with context.

    ``PaperSource.url`` is required, but several upstream records may
    legitimately lack a URL.  A missing URL is a data quality bug at the
    tool layer: it cannot be silently dropped, and a fabricated fallback
    would mislead downstream consumers, so propagate the issue.
    """
    if not value:
        msg = f"{context}: cannot normalise record without a URL"
        raise ValueError(msg)
    return value


class _SemanticScholarSearch:
    """Search Semantic Scholar for academic papers matching a free-text query.

    Returns paper metadata including title, abstract, authors, year, venue,
    citation count, and open-access PDF link when available.
    """

    _client: AsyncSemanticScholar

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

    async def __call__(self, query: str, *, limit: int) -> list[SearchResult]:
        """Search Semantic Scholar for papers matching a free-text query.

        Args:
            query: Plain-text search query.
            limit: Maximum number of results to return. Clamped to the
                Semantic Scholar API range of [1, 100].

        Returns:
            A list of normalised ``SearchResult`` objects, one per matched
            paper. May be shorter than *limit* (or empty) when the API
            returns fewer matches, or when matched records lack a title
            or abstract (silently dropped here).
        """
        clamped = max(_MIN_LIMIT, min(limit, _MAX_LIMIT))
        results = await self._client.search_paper(query, fields=_FIELDS, limit=clamped)
        if isinstance(results, Paper):
            papers: list[Paper] = [results]
        else:
            papers = list(results.items)
        papers = [
            p for p in papers if (p.title or "").strip() and (p.abstract or "").strip()
        ]
        return [self._to_search_result(p) for p in papers]

    def _to_search_result(self, paper: Paper) -> SearchResult:
        external = paper.externalIds or {}
        pdf = paper.openAccessPdf or {}
        url = paper.url or f"https://www.semanticscholar.org/paper/{paper.paperId}"
        pdf_url = HttpUrl(pdf["url"]) if pdf.get("url") else None
        title = (paper.title or "").strip()
        abstract = (paper.abstract or "").strip()
        doi = external.get("DOI") or None
        return SearchResult(
            paper=PaperInfo(
                source=PaperSource(
                    url=HttpUrl(_require_url(url, context="Semantic Scholar")),
                    open_access=bool(paper.isOpenAccess) and pdf_url is not None,
                    doi=doi,
                    pdf_url=pdf_url,
                ),
                title=title,
                abstract=abstract,
                authors=tuple(a.name for a in (paper.authors or []) if a.name),
                publication_year=paper.year,
                citation_count=paper.citationCount,
            ),
            search_index_reference=(
                SearchIndexReference(
                    index=SearchIndexType.SEMANTIC_SCHOLAR,
                    id=paper.paperId,
                ),
            ),
        )


class _ArXivSearch:
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

    async def __call__(self, query: str, *, limit: int) -> list[SearchResult]:
        """Search arXiv for papers matching a free-text query.

        Args:
            query: Plain-text search query. May include the same field
                prefixes as the arXiv API (``ti:``, ``au:``, ``abs:``,
                ``cat:``, ``all:``).
            limit: Maximum number of results to return.

        Returns:
            A list of normalised ``SearchResult`` objects, one per matched
            paper. May be shorter than *limit* (or empty) when the API
            returns fewer matches, or when matched records lack a title
            or abstract (silently dropped here).
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
        results = [
            r for r in results if (r.title or "").strip() and (r.summary or "").strip()
        ]
        return [self._to_search_result(r) for r in results]

    def _to_search_result(self, result: arxiv.Result) -> SearchResult:
        pdf_url = result.pdf_url or next(
            (link.href for link in (result.links or []) if link.title == "pdf"),
            None,
        )
        title = (result.title or "").strip()
        summary = (result.summary or "").strip()
        resolved_pdf = HttpUrl(pdf_url) if pdf_url else None
        return SearchResult(
            paper=PaperInfo(
                source=PaperSource(
                    url=HttpUrl(_require_url(result.entry_id, context="arXiv")),
                    open_access=resolved_pdf is not None,
                    doi=result.doi or None,
                    pdf_url=resolved_pdf,
                ),
                title=title,
                abstract=summary,
                authors=tuple(a.name for a in (result.authors or []) if a.name),
                publication_year=(
                    result.published.year
                    if result.published and result.published.year > 1
                    else None
                ),
                citation_count=None,
            ),
            search_index_reference=(
                SearchIndexReference(
                    index=SearchIndexType.ARXIV,
                    id=result.entry_id,
                ),
            ),
        )


class _PubMedSearch:
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

    async def __call__(self, query: str, *, limit: int) -> list[SearchResult]:
        """Search PubMed for papers matching a free-text query.

        Args:
            query: Plain-text search query. Supports the full PubMed
                query syntax (MeSH terms, field tags, boolean operators).
            limit: Maximum number of results to return.

        Returns:
            A list of normalised ``SearchResult`` objects, one per matched
            paper. May be shorter than *limit* (or empty) when the API
            returns fewer matches, or when matched records lack a title
            or abstract (silently dropped here).
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
        articles = [a for a in articles if _PubMedSearch._has_title_and_abstract(a)]
        return [_PubMedSearch._to_search_result(a) for a in articles]

    @staticmethod
    def _has_title_and_abstract(article: dict[str, Any]) -> bool:
        art = article["MedlineCitation"]["Article"]
        title = str(art.get("ArticleTitle", "")).strip()
        abstract_parts = art.get("Abstract", {}).get("AbstractText", [])
        if isinstance(abstract_parts, list):
            abstract = " ".join(str(p) for p in abstract_parts).strip()
        else:
            abstract = str(abstract_parts).strip() if abstract_parts else ""
        return bool(title) and bool(abstract)

    @staticmethod
    def _to_search_result(article: dict[str, Any]) -> SearchResult:
        med = article["MedlineCitation"]
        art = med["Article"]

        title = str(art.get("ArticleTitle", "")).strip()
        abstract_parts = art.get("Abstract", {}).get("AbstractText", [])
        if isinstance(abstract_parts, list):
            abstract = " ".join(str(p) for p in abstract_parts).strip()
        else:
            abstract = str(abstract_parts).strip() if abstract_parts else ""

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
        pub_date = journal.get("JournalIssue", {}).get("PubDate", {})
        year_raw = pub_date.get("Year", "")
        publication_year = int(year_raw) if year_raw else None

        doi: str | None = None
        for eid in art.get("ELocationID", []):
            attrs = getattr(eid, "attributes", {}) or {}
            if attrs.get("EIdType") == "doi":
                doi_candidate = str(eid).strip() or None
                if doi_candidate:
                    doi = doi_candidate
                break

        pmid = str(med.get("PMID", ""))

        return SearchResult(
            paper=PaperInfo(
                source=PaperSource(
                    url=HttpUrl(f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"),
                    open_access=False,
                    doi=doi,
                    pdf_url=None,
                ),
                title=title,
                abstract=abstract,
                authors=tuple(authors),
                publication_year=publication_year,
                citation_count=None,
            ),
            search_index_reference=(
                SearchIndexReference(
                    index=SearchIndexType.PUBMED,
                    id=pmid,
                ),
            ),
        )


class _CrossRefSearch:
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

    async def __call__(self, query: str, *, limit: int) -> list[SearchResult]:
        """Search CrossRef for works matching a free-text query.

        Args:
            query: Plain-text search query.
            limit: Maximum number of results to return.

        Returns:
            A list of normalised ``SearchResult`` objects, one per matched
            work. May be shorter than *limit* (or empty) when the API
            returns fewer matches, or when matched records lack a title
            or abstract (silently dropped here).
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
        items = [i for i in items if _CrossRefSearch._has_title_and_abstract(i)]
        return [_CrossRefSearch._to_search_result(i) for i in items]

    @staticmethod
    def _has_title_and_abstract(item: dict[str, Any]) -> bool:
        title_list = item.get("title", [])
        title = (title_list[0] if title_list else "").strip()
        raw_abstract = item.get("abstract", "")
        abstract = _STRIP_JATS.sub("", raw_abstract).strip() if raw_abstract else ""
        return bool(title) and bool(abstract)

    @staticmethod
    def _to_search_result(item: dict[str, Any]) -> SearchResult:
        title_list = item.get("title", [])
        title = (title_list[0] if title_list else "").strip()

        raw_abstract = item.get("abstract", "")
        abstract = _STRIP_JATS.sub("", raw_abstract).strip() if raw_abstract else ""

        author_list = item.get("author", [])
        authors: list[str] = []
        for a in author_list:
            given = a.get("given", "")
            family = a.get("family", "")
            name = f"{given} {family}".strip()
            if name:
                authors.append(name)

        doi = item.get("DOI") or None
        url = item.get("URL") or (f"https://doi.org/{doi}" if doi else "")

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
        resolved_pdf = HttpUrl(resource_url) if resource_url else None

        return SearchResult(
            paper=PaperInfo(
                source=PaperSource(
                    url=HttpUrl(_require_url(url, context="CrossRef")),
                    open_access=resolved_pdf is not None,
                    doi=doi,
                    pdf_url=resolved_pdf,
                ),
                title=title,
                abstract=abstract,
                authors=tuple(authors),
                publication_year=publication_year,
                citation_count=item.get("is-referenced-by-count"),
            ),
            search_index_reference=(
                SearchIndexReference(
                    index=SearchIndexType.CROSSREF,
                    id=doi or "",
                ),
            ),
        )


class LiteratureSearch:
    """Unified literature search that dispatches to one of four private index handlers.

    The single async callable the search node wraps in ``dspy.Tool`` and
    exposes to the agent.  The agent picks the index per call via
    *search_index*; the actual API call is delegated to the matching
    private handler.

    Each ``SearchIndexType`` value in the domain enum maps to exactly one
    private handler.  ``__call__`` is keyword-only on ``limit`` and
    rejects unknown indices with ``LiteratureSearchError``.
    """

    def __init__(
        self,
        *,
        s2_api_key: str | None = None,
        pubmed_api_key: str | None = None,
    ) -> None:
        """Initialise the unified search tool and its private index handlers.

        Args:
            s2_api_key: Optional Semantic Scholar API key forwarded to
                the private Semantic Scholar handler.  Unauthenticated
                traffic shares a global 1,000 req/s pool; an
                authenticated key is recommended for any non-interactive
                use.
            pubmed_api_key: Optional NCBI API key forwarded to the
                private PubMed handler for elevated rate limits
                (10 req/s vs ~3 req/s unauthenticated).
        """
        self._handlers: dict[SearchIndexType, _IndexSearch] = {
            SearchIndexType.SEMANTIC_SCHOLAR: _SemanticScholarSearch(
                api_key=s2_api_key,
            ),
            SearchIndexType.ARXIV: _ArXivSearch(),
            SearchIndexType.PUBMED: _PubMedSearch(api_key=pubmed_api_key),
            SearchIndexType.CROSSREF: _CrossRefSearch(),
        }

    async def __call__(
        self,
        search_index: SearchIndexType,
        query: str,
        *,
        limit: int,
    ) -> list[SearchResult]:
        """Search the chosen index for papers matching a free-text query.

        Args:
            search_index: Which index to query.  One of the
                ``SearchIndexType`` enum values (Semantic Scholar, arXiv,
                PubMed, CrossRef).
            query: Plain-text search query.  Query syntax is delegated to
                the chosen index's handler.
            limit: Maximum number of results to return.  Required; each
                handler clamps it to its own API range.

        Returns:
            A list of normalised ``SearchResult`` objects returned by the
            chosen index.  May be shorter than *limit* (or empty) when
            the index returns fewer matches.

        Raises:
            UnknownIndexError: If *search_index* is not a recognised
                ``SearchIndexType`` value.
        """
        handler = self._handlers.get(search_index)
        if handler is None:
            msg = f"unknown search index: {search_index!r}"
            raise UnknownIndexError(msg)
        return await handler(query, limit=limit)


class _IndexSearch(Protocol):
    """Structural shape satisfied by the four private index handlers.

    Local to this module and not exported.  ``LiteratureSearch`` only
    relies on the ``async __call__(query, *, limit) -> list[SearchResult]``
    shape.
    """

    async def __call__(self, query: str, *, limit: int) -> list[SearchResult]: ...
