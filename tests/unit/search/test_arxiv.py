"""Unit tests for the arXiv search tool.

Exercises the pure normalisation path with a fabricated ``arxiv.Result``,
no network required.
"""

from __future__ import annotations

from datetime import UTC, datetime

import arxiv

from research_agent.search.models import SearchIndexType, SearchResult
from research_agent.search.tools import _ArXivSearch


def _make_result(  # noqa: PLR0913  # test helper mirroring arxiv.Result constructor
    *,
    title: str = "Test Paper",
    summary: str = "Test abstract.",
    authors: list[arxiv.Result.Author] | None = None,
    doi: str | None = None,
    categories: list[str] | None = None,
    pdf_link: str | None = None,
    entry_id: str = "http://arxiv.org/abs/2306.04338v1",
    journal_ref: str | None = None,
    comment: str | None = None,
    published: datetime | None = None,
) -> arxiv.Result:
    links: list[arxiv.Result.Link] = [
        arxiv.Result.Link(
            href=entry_id,
            title=None,
            rel="alternate",
            content_type=None,
        )
    ]
    if pdf_link:
        links.append(
            arxiv.Result.Link(
                href=pdf_link,
                title="pdf",
                rel="related",
                content_type="application/pdf",
            )
        )
    return arxiv.Result(
        entry_id=entry_id,
        title=title,
        summary=summary,
        authors=authors or [],
        doi=doi or "",
        categories=categories or [],
        primary_category=(categories[0] if categories else ""),
        journal_ref=journal_ref or "",
        comment=comment or "",
        links=links,
        published=published or datetime(1, 1, 1, tzinfo=UTC),
    )


def test_to_search_result_handles_missing_doi() -> None:
    sr = _ArXivSearch()._to_search_result(
        _make_result(doi=None, authors=[arxiv.Result.Author(name="Alice")])
    )
    assert isinstance(sr, SearchResult)
    assert sr.paper.source.doi is None


def test_to_search_result_uses_journal_ref_as_venue() -> None:
    sr = _ArXivSearch()._to_search_result(
        _make_result(
            journal_ref="Nature 2020",
            authors=[arxiv.Result.Author(name="Alice")],
        )
    )
    assert isinstance(sr, SearchResult)
    assert sr.paper.raw_metadata is not None
    assert sr.paper.raw_metadata["venue"] == "Nature 2020"


def test_to_search_result_falls_back_to_comment_when_no_journal_ref() -> None:
    sr = _ArXivSearch()._to_search_result(
        _make_result(
            journal_ref="",
            comment="Accepted at NeurIPS",
            authors=[arxiv.Result.Author(name="Alice")],
        )
    )
    assert isinstance(sr, SearchResult)
    assert sr.paper.raw_metadata is not None
    assert sr.paper.raw_metadata["venue"] == "Accepted at NeurIPS"


def test_to_search_result_extracts_pdf_from_links_when_no_pdf_url() -> None:
    result = _make_result(
        pdf_link=None,
        authors=[arxiv.Result.Author(name="Alice")],
    )
    result.links = [
        arxiv.Result.Link(
            href="http://arxiv.org/abs/2306.04338v1",
            title=None,
            rel="alternate",
            content_type=None,
        ),
        arxiv.Result.Link(
            href="http://arxiv.org/pdf/2306.04338v1",
            title="pdf",
            rel="related",
            content_type="application/pdf",
        ),
    ]
    sr = _ArXivSearch()._to_search_result(result)
    assert isinstance(sr, SearchResult)
    assert str(sr.paper.source.pdf_url) == "http://arxiv.org/pdf/2306.04338v1"


def test_to_search_result_sentinel_year_is_none() -> None:
    sr = _ArXivSearch()._to_search_result(
        _make_result(
            authors=[arxiv.Result.Author(name="Alice")],
            published=datetime(1, 1, 1, tzinfo=UTC),
        )
    )
    assert isinstance(sr, SearchResult)
    assert sr.paper.publication_year is None


def test_to_search_result_filters_none_author_names() -> None:
    sr = _ArXivSearch()._to_search_result(
        _make_result(
            authors=[
                arxiv.Result.Author(name="Alice"),
                arxiv.Result.Author(name=""),
                arxiv.Result.Author(name="Bob"),
            ],
        )
    )
    assert isinstance(sr, SearchResult)
    assert sr.paper.authors == ["Alice", "Bob"]


def test_to_search_result_index_is_arxiv() -> None:
    sr = _ArXivSearch()._to_search_result(
        _make_result(authors=[arxiv.Result.Author(name="Alice")])
    )
    assert sr.search_reference.index == SearchIndexType.ARXIV
