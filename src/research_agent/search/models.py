"""Domain models for search.

Layer: Domain.
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class MissingOpenAccessPDFError(Exception):
    """Raised when a paper is open access but does not have a PDF URL."""

    def __init__(self) -> None:
        """Initialize the exception with a default message."""
        super().__init__("Missing PDF URL for open access paper")


class SearchIndexType(StrEnum):
    """Identifies which search result originated from."""

    ARXIV = "arxiv"
    PUBMED = "pubmed"
    CROSSREF = "crossref"


class SearchIndexReference(BaseModel):
    """A paper's identifier within a specific search index."""

    model_config = ConfigDict(frozen=True)

    index: SearchIndexType
    id: str = Field(..., description="Identifier of the paper within the search index")


class PaperSource(BaseModel):
    """How a paper is identified and located in the literature."""

    model_config = ConfigDict(frozen=True)

    url: HttpUrl = Field(..., description="URL of the paper")
    open_access: bool = Field(..., description="Whether the paper is open access")
    doi: str | None = Field(None, description="Cross-index standard DOI identifier")
    pdf_url: HttpUrl | None = Field(None, description="URL of the paper's PDF")

    @model_validator(mode="after")
    def _check_open_access_has_pdf(self) -> PaperSource:
        if self.open_access and self.pdf_url is None:
            raise MissingOpenAccessPDFError
        return self


class PaperInfo(BaseModel):
    """Information about a paper, including its source and metadata."""

    model_config = ConfigDict(frozen=True)

    source: PaperSource

    title: str = Field(..., min_length=10, description="Title of the paper")
    abstract: str = Field(..., min_length=200, description="Abstract of the paper")

    authors: tuple[str, ...] = Field(
        ..., min_length=1, description="Authors of the paper"
    )

    publication_year: int | None = None
    citation_count: int | None = None


class ResearchQuery(BaseModel):
    """A free-text research question, optionally scoped to domains."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(min_length=5)
    domains: tuple[str, ...] | None = None


class SearchResult(BaseModel):
    """A unified search result returned by a search index tool."""

    model_config = ConfigDict(frozen=True)

    paper: PaperInfo
    search_index_reference: tuple[SearchIndexReference, ...] = Field(..., min_length=1)
