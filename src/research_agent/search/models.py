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
    """Supported literature search indexes for tool dispatch."""

    PUBMED = "pubmed"
    CROSSREF = "crossref"
    OPENALEX = "openalex"


class SearchStatus(StrEnum):
    """Terminal status of a literature search ReAct episode.

    Reported by the search agent after tool use; not used to filter the
    session paper bag. Values:

    * ``complete`` — enough useful papers gathered.
    * ``insufficient_search`` — stopped early or under-explored.
    * ``irrelevant_results`` — hits found but do not answer the query.
    * ``missing_results`` — little or no hits despite reasonable queries.
    * ``tool_error`` — tool or API failures dominated the trajectory.
    """

    COMPLETE = "complete"
    INSUFFICIENT_SEARCH = "insufficient_search"
    IRRELEVANT_RESULTS = "irrelevant_results"
    MISSING_RESULTS = "missing_results"
    TOOL_ERROR = "tool_error"


class PaperInfo(BaseModel):
    """Information about a paper, including metadata and source location."""

    model_config = ConfigDict(frozen=True)

    title: str = Field(..., min_length=10, description="Title of the paper")
    abstract: str = Field(..., min_length=200, description="Abstract of the paper")
    authors: tuple[str, ...] = Field(
        ..., min_length=1, description="Authors of the paper"
    )

    url: HttpUrl = Field(..., description="URL of the paper")
    open_access: bool = Field(..., description="Whether the paper is open access")
    doi: str | None = Field(None, description="Cross-index standard DOI identifier")
    pdf_url: HttpUrl | None = Field(None, description="URL of the paper's PDF")

    publication_year: int | None = None
    citation_count: int | None = None

    @model_validator(mode="after")
    def _check_open_access_has_pdf(self) -> PaperInfo:
        if self.open_access and self.pdf_url is None:
            raise MissingOpenAccessPDFError
        return self


class ResearchQuery(BaseModel):
    """A free-text research question, optionally scoped to domains."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(min_length=5)
    domains: tuple[str, ...] | None = None
