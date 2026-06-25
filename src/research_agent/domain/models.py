"""Domain models for the research agent.

These models carry business meaning and are implemented as Pydantic models
for input/output contract validation. They do not enforce domain invariants
beyond structural validity.
"""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel


class SearchIndexType(StrEnum):
    """Identifies which search index a result originated from."""

    OPENALEX = "openalex"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    ARXIV = "arxiv"
    PUBMED = "pubmed"
    CROSSREF = "crossref"
    CORE = "core"
    EUROPE_PMC = "europe_pmc"


class SearchIndexId(BaseModel):
    """A paper's native identifier within a specific search index.

    The native ID is only meaningful in the context of its index, so the
    two are kept together as a unit. An ID lifted out of its index is
    ambiguous; a `(index, id)` pair is not.
    """

    index: SearchIndexType
    id: str


class PaperReference(BaseModel):
    """How a paper is identified and located in the literature.

    `source` carries the index-specific `(index, id)` pair that pinpoints
    the paper inside a particular catalog. `doi` is kept separate because
    it is a cross-index standard that does not depend on any single index.
    """

    source: SearchIndexId
    doi: str | None = None


class ResearchQuery(BaseModel):
    """A free-text research question, optionally scoped to domains."""

    text: str
    domains: list[str] | None = None


class SearchResult(BaseModel):
    """A normalized search result returned by a search index tool.

    `title` and `abstract` are the primary signals for downstream relevance
    scoring. Results missing either are heavily penalized by the optimization
    metric in `optimize.metric` and should be treated as quality bugs at the
    tool layer; the fields are typed optional only because some upstream
    indexes (notably Semantic Scholar) omit them in a non-trivial fraction
    of records for legal reasons.
    """

    title: str | None = None
    abstract: str | None = None
    authors: list[str]

    reference: PaperReference

    url: str | None = None
    pdf_url: str | None = None
    publication_year: int | None = None
    venue: str | None = None
    citation_count: int | None = None
    is_open_access: bool | None = None
    topics: list[str] | None = None
    tldr: str | None = None

    raw_metadata: dict[str, Any] | None = None


class SearchResults(BaseModel):
    """A collection of search results."""

    results: list[SearchResult]
