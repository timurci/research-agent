"""Domain models for the research agent.

These models carry business meaning and are implemented as Pydantic models
for input/output contract validation. They do not enforce domain invariants
beyond structural validity.
"""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, HttpUrl


class SearchIndexType(StrEnum):
    """Identifies which search result originated from."""

    SEMANTIC_SCHOLAR = "semantic_scholar"
    ARXIV = "arxiv"
    PUBMED = "pubmed"
    CROSSREF = "crossref"


class SearchIndexReference(BaseModel):
    """A paper's identifier within a specific search index."""

    index: SearchIndexType
    id: str = Field(..., description="Identifier of the paper within the search index")


class PaperSource(BaseModel):
    """How a paper is identified and located in the literature."""

    url: HttpUrl = Field(..., description="URL of the paper")
    doi: str | None = Field(None, description="Cross-index standard DOI identifier")
    pdf_url: HttpUrl | None = Field(None, description="URL of the paper's PDF")


class PaperInfo(BaseModel):
    """Information about a paper, including its source and metadata."""

    source: PaperSource

    title: str | None = None
    abstract: str | None = None

    authors: list[str]
    publication_year: int | None = None
    citation_count: int | None = None
    is_open_access: bool | None = None

    raw_metadata: dict[str, Any] | None = None


class ResearchQuery(BaseModel):
    """A free-text research question, optionally scoped to domains."""

    text: str = Field(min_length=5)
    domains: list[str] | None = None


class SearchResult(BaseModel):
    """A unified search result returned by a search index tool."""

    paper: PaperInfo
    search_reference: SearchIndexReference
