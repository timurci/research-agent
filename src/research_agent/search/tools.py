"""Search-index tools for the research agent.

Layer: Infrastructure.

Wraps synchronous SDKs for PubMed/NCBI and CrossRef, and the OpenAlex
REST API (async ``httpx``), and normalises every response into the
domain ``PaperInfo`` shape.  ``LiteratureSearch`` dispatches to private
per-index handlers.  ``IndexedLiteratureSearch`` composes a pure
``LiteratureSearch`` with a ``Session``: it appends hits to
``search_results`` and returns full papers tagged with absolute list
indices for LM selection.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Protocol
from urllib.parse import urlencode

import httpx
from Bio import Entrez
from habanero import Crossref
from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from research_agent.search.models import PaperInfo, SearchIndexType
from research_agent.shared.executor import run_async

if TYPE_CHECKING:
    from research_agent.shared.session import Session

_MAX_LIMIT: int = 100
_MIN_LIMIT: int = 1
_OPENALEX_MAX_PER_PAGE: int = 100

_DEFAULT_MAILTO: str = "research-agent@example.com"
_STRIP_JATS = re.compile(r"<[^>]+>")
_DOI_URL_PREFIX: str = "https://doi.org/"
_OPENALEX_WORKS_URL: str = "https://api.openalex.org/works"
_OPENALEX_TIMEOUT_SECONDS: float = 30.0

SEARCH_RESULTS_KEY: str = "search_results"


class UnknownIndexError(Exception):
    """Raised when ``LiteratureSearch`` receives an unknown index."""


class _ResultValidator(BaseModel):
    results: list[PaperInfo]


def _require_url(value: str | None, *, context: str) -> str:
    """Return *value* if non-empty, else raise ``ValueError`` with context.

    ``PaperInfo.url`` is required, but several upstream records may
    legitimately lack a URL.  A missing URL is a data quality bug at the
    tool layer: it cannot be silently dropped, and a fabricated fallback
    would mislead downstream consumers, so propagate the issue.
    """
    if not value:
        msg = f"{context}: cannot normalise record without a URL"
        raise ValueError(msg)
    return value


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

    async def __call__(self, query: str, *, limit: int) -> list[PaperInfo]:
        """Search PubMed for papers matching a free-text query.

        Args:
            query: Plain-text search query. Supports the full PubMed
                query syntax (MeSH terms, field tags, boolean operators).
            limit: Maximum number of results to return.

        Returns:
            A list of normalised ``PaperInfo`` objects, one per matched
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
        return [_PubMedSearch._to_paper_info(a) for a in articles]

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
    def _to_paper_info(article: dict[str, Any]) -> PaperInfo:
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

        return PaperInfo(
            title=title,
            abstract=abstract,
            authors=tuple(authors),
            url=HttpUrl(f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"),
            open_access=False,
            doi=doi,
            pdf_url=None,
            publication_year=publication_year,
            citation_count=None,
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

    async def __call__(self, query: str, *, limit: int) -> list[PaperInfo]:
        """Search CrossRef for works matching a free-text query.

        Args:
            query: Plain-text search query.
            limit: Maximum number of results to return.

        Returns:
            A list of normalised ``PaperInfo`` objects, one per matched
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
        return [_CrossRefSearch._to_paper_info(i) for i in items]

    @staticmethod
    def _has_title_and_abstract(item: dict[str, Any]) -> bool:
        title_list = item.get("title", [])
        title = (title_list[0] if title_list else "").strip()
        raw_abstract = item.get("abstract", "")
        abstract = _STRIP_JATS.sub("", raw_abstract).strip() if raw_abstract else ""
        return bool(title) and bool(abstract)

    @staticmethod
    def _to_paper_info(item: dict[str, Any]) -> PaperInfo:
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

        return PaperInfo(
            title=title,
            abstract=abstract,
            authors=tuple(authors),
            url=HttpUrl(_require_url(url, context="CrossRef")),
            open_access=resolved_pdf is not None,
            doi=doi,
            pdf_url=resolved_pdf,
            publication_year=publication_year,
            citation_count=item.get("is-referenced-by-count"),
        )


class _OpenAlexSearch:
    """Search OpenAlex for works matching a free-text keyword query.

    Uses the async OpenAlex Works API (``GET /works?search=...``).
    Keyword search costs about $0.001 per call on the freemium plan;
    a free API key yields about $1/day (~1k searches). Peak RPS can
    reach 100; daily budget usually binds first for AI eval suites.
    The API key is constructor-injected only — never loaded from the
    environment here.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        mailto: str = _DEFAULT_MAILTO,
    ) -> None:
        """Initialise the OpenAlex search tool.

        Args:
            api_key: Optional OpenAlex API key.  Callers obtain and inject
                it; this handler never reads the environment.  ``None``
                uses the unauthenticated free tier.
            mailto: Contact address sent as the polite ``mailto`` query
                parameter.
        """
        self._api_key = api_key
        self._mailto = mailto

    async def __call__(self, query: str, *, limit: int) -> list[PaperInfo]:
        """Search OpenAlex for works matching a free-text query.

        Args:
            query: Plain-text keyword search query.
            limit: Maximum number of results to return (clamped to 1-100).

        Returns:
            A list of normalised ``PaperInfo`` objects. May be shorter
            than *limit* (or empty) when the API returns fewer matches,
            or when matched records lack title, abstract, or authors
            (silently dropped here).
        """
        clamped = max(_MIN_LIMIT, min(limit, _OPENALEX_MAX_PER_PAGE))
        works = await self._fetch_works(query, per_page=clamped)
        works = [w for w in works if _OpenAlexSearch._is_complete(w)]
        return [_OpenAlexSearch._to_paper_info(w) for w in works]

    async def _fetch_works(self, query: str, *, per_page: int) -> list[dict[str, Any]]:
        """Fetch raw work dicts from the OpenAlex Works search endpoint."""
        params: dict[str, str | int] = {
            "search": query,
            "filter": "has_abstract:true",
            "per_page": per_page,
            "mailto": self._mailto,
        }
        if self._api_key is not None:
            params["api_key"] = self._api_key
        url = f"{_OPENALEX_WORKS_URL}?{urlencode(params)}"
        async with httpx.AsyncClient(timeout=_OPENALEX_TIMEOUT_SECONDS) as client:
            response = await client.get(url)
            response.raise_for_status()
            payload = response.json()
        if not isinstance(payload, dict):
            msg = f"OpenAlex returned unexpected type {type(payload).__name__}"
            raise TypeError(msg)
        results = payload.get("results", [])
        if not isinstance(results, list):
            msg = "OpenAlex response 'results' must be a list"
            raise TypeError(msg)
        return results

    @staticmethod
    def _reconstruct_abstract(inverted: dict[str, list[int]] | None) -> str:
        """Rebuild plain-text abstract from an OpenAlex inverted index."""
        if not inverted:
            return ""
        by_pos: dict[int, str] = {}
        for token, positions in inverted.items():
            for pos in positions:
                by_pos[pos] = token
        return " ".join(by_pos[i] for i in sorted(by_pos))

    @staticmethod
    def _is_complete(work: dict[str, Any]) -> bool:
        title = str(work.get("display_name") or "").strip()
        abstract = _OpenAlexSearch._reconstruct_abstract(
            work.get("abstract_inverted_index")
        )
        authors = _OpenAlexSearch._author_names(work)
        return bool(title) and bool(abstract) and bool(authors)

    @staticmethod
    def _author_names(work: dict[str, Any]) -> list[str]:
        names: list[str] = []
        for authorship in work.get("authorships") or []:
            author = authorship.get("author") or {}
            name = str(author.get("display_name") or "").strip()
            if name:
                names.append(name)
        return names

    @staticmethod
    def _pdf_url(work: dict[str, Any]) -> str | None:
        best = work.get("best_oa_location") or {}
        pdf = best.get("pdf_url")
        if pdf:
            return str(pdf)
        for location in work.get("locations") or []:
            loc_pdf = location.get("pdf_url")
            if loc_pdf:
                return str(loc_pdf)
        return None

    @staticmethod
    def _resolve_url(work: dict[str, Any], doi: str | None) -> str:
        primary = work.get("primary_location") or {}
        landing = primary.get("landing_page_url")
        if landing:
            return str(landing)
        if doi:
            return f"{_DOI_URL_PREFIX}{doi}"
        openalex_id = work.get("id")
        if openalex_id:
            return str(openalex_id)
        return ""

    @staticmethod
    def _normalise_doi(raw: str | None) -> str | None:
        if not raw:
            return None
        doi = str(raw).strip()
        if doi.startswith(_DOI_URL_PREFIX):
            doi = doi.removeprefix(_DOI_URL_PREFIX)
        return doi or None

    @staticmethod
    def _to_paper_info(work: dict[str, Any]) -> PaperInfo:
        title = str(work.get("display_name") or "").strip()
        abstract = _OpenAlexSearch._reconstruct_abstract(
            work.get("abstract_inverted_index")
        )
        authors = _OpenAlexSearch._author_names(work)
        doi = _OpenAlexSearch._normalise_doi(work.get("doi"))
        url = _OpenAlexSearch._resolve_url(work, doi)
        pdf_raw = _OpenAlexSearch._pdf_url(work)
        pdf_url = HttpUrl(pdf_raw) if pdf_raw else None
        year = work.get("publication_year")
        publication_year = int(year) if year is not None else None
        citation_count = work.get("cited_by_count")
        if citation_count is not None:
            citation_count = int(citation_count)

        return PaperInfo(
            title=title,
            abstract=abstract,
            authors=tuple(authors),
            url=HttpUrl(_require_url(url, context="OpenAlex")),
            open_access=pdf_url is not None,
            doi=doi,
            pdf_url=pdf_url,
            publication_year=publication_year,
            citation_count=citation_count,
        )


class LiteratureSearch:
    """Pure literature index dispatcher (not the ReAct tool).

    Routes each call to one private PubMed, CrossRef, or OpenAlex
    handler via *search_index*.  Does not touch session state.

    The ReAct-facing tool is ``IndexedLiteratureSearch``, which the search
    agent wraps in ``dspy.Tool`` under the name ``LiteratureSearch``.
    Unknown indices raise ``UnknownIndexError``.
    """

    def __init__(
        self,
        *,
        pubmed_api_key: str | None = None,
        openalex_api_key: str | None = None,
    ) -> None:
        """Initialise the unified search tool and its private index handlers.

        Args:
            pubmed_api_key: Optional NCBI API key forwarded to the
                private PubMed handler for elevated rate limits
                (10 req/s vs ~3 req/s unauthenticated).
            openalex_api_key: Optional OpenAlex API key forwarded to the
                private OpenAlex handler.  Injected by the caller; never
                loaded from the environment inside this class.
        """
        self._handlers: dict[SearchIndexType, _IndexSearch] = {
            SearchIndexType.PUBMED: _PubMedSearch(api_key=pubmed_api_key),
            SearchIndexType.CROSSREF: _CrossRefSearch(),
            SearchIndexType.OPENALEX: _OpenAlexSearch(api_key=openalex_api_key),
        }

    async def __call__(
        self,
        search_index: SearchIndexType,
        query: str,
        *,
        limit: int,
    ) -> list[PaperInfo]:
        """Search the chosen index for papers matching a free-text query.

        Args:
            search_index: Which index to query.  One of the
                ``SearchIndexType`` enum values (PubMed, CrossRef,
                OpenAlex).
            query: Plain-text search query.  Query syntax is delegated to
                the chosen index's handler.
            limit: Maximum number of results to return.  Required; each
                handler clamps it to its own API range.

        Returns:
            A list of normalised ``PaperInfo`` objects returned by the
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
    """Structural shape satisfied by the private index handlers.

    Local to this module and not exported.  ``LiteratureSearch`` only
    relies on the ``async __call__(query, *, limit) -> list[PaperInfo]``
    shape.
    """

    async def __call__(self, query: str, *, limit: int) -> list[PaperInfo]: ...


class IndexedPaper(BaseModel):
    """A paper recorded in session memory with its absolute list index."""

    model_config = ConfigDict(frozen=True)

    id: int = Field(..., ge=0, description="Absolute index in session search_results")
    paper: PaperInfo = Field(..., description="Full paper record")


class IndexedLiteratureSearch:
    """Literature search that indexes hits into session state for selection.

    Composes a pure ``LiteratureSearch`` with a ``Session``. Each call
    appends new papers to ``session[search_results]``; absolute list
    indices are the selectable ids. Observations include full
    ``PaperInfo`` bodies (not truncated cards).
    """

    def __init__(
        self,
        session: Session,
        literature_search: LiteratureSearch,
    ) -> None:
        """Wire session and the pure index client.

        Args:
            session: Session whose ``search_results`` list is appended to.
            literature_search: Pure literature index dispatcher.
        """
        self._session = session
        self._literature_search = literature_search

    async def __call__(
        self,
        search_index: SearchIndexType,
        query: str,
        *,
        limit: int,
    ) -> list[IndexedPaper]:
        """Search an index, record hits, return id-tagged full papers.

        Args:
            search_index: Which index to query.  One of the
                ``SearchIndexType`` enum values (PubMed, CrossRef,
                OpenAlex).
            query: Plain-text search query.  Query syntax is delegated to
                the chosen index's handler.
            limit: Maximum number of results to return.  Required; each
                handler clamps it to its own API range.

        Returns:
            Newly found papers tagged with absolute indices into the
            session ``search_results`` list.  May be shorter than *limit*
            (or empty) when the index returns fewer matches.
        """
        papers = await self._literature_search(search_index, query, limit=limit)
        bag = _ResultValidator.model_validate(
            {"results": self._session.get(SEARCH_RESULTS_KEY, [])}
        ).results
        start = len(bag)
        bag.extend(papers)
        self._session.set(SEARCH_RESULTS_KEY, bag)
        return [
            IndexedPaper(id=start + i, paper=paper) for i, paper in enumerate(papers)
        ]
