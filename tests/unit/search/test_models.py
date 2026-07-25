"""Hashability and coercion regression tests for the frozen domain models.

Models that are ``frozen=True`` and contain only hashable fields should be
hashable so that downstream code (e.g. ``set[PaperInfo]``) can use them as
set / dict keys. ``ToolCallObservation`` is frozen for immutability but
intentionally unhashable because ``call_args`` is a ``dict``.

This file pins those invariants: equal hashable models produce equal hashes;
an equal model is deduplicated in a set; a model whose field differs is not
equal to the original; sequence fields are coerced from lists to tuples.
"""

from __future__ import annotations

import pytest
from pydantic import HttpUrl

from research_agent.search.models import (
    MissingOpenAccessPDFError,
    PaperInfo,
    ResearchQuery,
    SearchStatus,
)
from research_agent.shared.metric import ToolCallObservation

_ABSTRACT = (
    "A sufficiently long abstract describing the research methodology, "
    "experimental setup, results, and conclusions of this work in detail "
    "to satisfy the PaperInfo min_length=200 invariant enforced by Pydantic."
)
_TITLE = "Test Paper Title Long Enough"


def _paper(
    *,
    title: str = _TITLE,
    authors: tuple[str, ...] = ("Alice",),
) -> PaperInfo:
    return PaperInfo(
        title=title,
        abstract=_ABSTRACT,
        authors=authors,
        url=HttpUrl("https://example.com/paper"),
        open_access=False,
    )


def test_paper_info_is_hashable() -> None:
    a = _paper()
    b = _paper()
    assert hash(a) == hash(b)
    assert {a, b} == {a}


def test_paper_info_with_different_authors_is_not_equal() -> None:
    a = _paper(authors=("Alice",))
    b = _paper(authors=("Bob",))
    assert a != b
    assert {a, b} == {a, b}


def test_paper_info_with_different_url_is_not_equal() -> None:
    a = _paper()
    b = PaperInfo(
        title=_TITLE,
        abstract=_ABSTRACT,
        authors=("Alice",),
        url=HttpUrl("https://example.com/other"),
        open_access=False,
    )
    assert a != b


def test_paper_info_authors_coerced_from_list_to_tuple() -> None:
    paper = _paper(authors=["Alice", "Bob"])  # ty: ignore[invalid-argument-type]  # Pydantic coerces list to tuple
    assert isinstance(paper.authors, tuple)
    assert paper.authors == ("Alice", "Bob")


def test_paper_info_open_access_requires_pdf_url() -> None:
    with pytest.raises(MissingOpenAccessPDFError):
        PaperInfo(
            title=_TITLE,
            abstract=_ABSTRACT,
            authors=("Alice",),
            url=HttpUrl("https://example.com/paper"),
            open_access=True,
            pdf_url=None,
        )


def test_paper_info_open_access_with_pdf_url_ok() -> None:
    paper = PaperInfo(
        title=_TITLE,
        abstract=_ABSTRACT,
        authors=("Alice",),
        url=HttpUrl("https://example.com/paper"),
        open_access=True,
        pdf_url=HttpUrl("https://example.com/paper.pdf"),
    )
    assert paper.open_access is True
    assert str(paper.pdf_url) == "https://example.com/paper.pdf"


def test_search_status_values() -> None:
    assert SearchStatus.COMPLETE == "complete"
    assert {member.value for member in SearchStatus} == {
        "complete",
        "insufficient_search",
        "irrelevant_results",
        "missing_results",
        "tool_error",
    }


def test_research_query_is_hashable() -> None:
    a = ResearchQuery(text="hello world")
    b = ResearchQuery(text="hello world")
    assert hash(a) == hash(b)
    assert {a, b} == {a}


def test_research_query_with_different_domains_is_not_equal() -> None:
    a = ResearchQuery(text="hello world", domains=("cs",))
    b = ResearchQuery(text="hello world", domains=("bio",))
    assert a != b


def test_research_query_domains_coerced_from_list_to_tuple() -> None:
    query = ResearchQuery(text="hello world", domains=["cs", "bio"])  # ty: ignore[invalid-argument-type]  # Pydantic coerces list to tuple
    assert isinstance(query.domains, tuple)
    assert query.domains == ("cs", "bio")


def test_tool_call_observation_is_frozen_but_unhashable() -> None:
    obs = ToolCallObservation(
        tool_name="LiteratureSearch",
        call_args={"query": "q", "limit": 5},
        observation=None,
    )
    assert isinstance(obs.call_args, dict)
    with pytest.raises(TypeError):
        hash(obs)


def test_tool_call_observation_stores_call_args_as_dict() -> None:
    obs = ToolCallObservation(
        tool_name="LiteratureSearch",
        call_args={"query": "q"},
        observation=None,
    )
    assert isinstance(obs.call_args, dict)
    assert obs.call_args == {"query": "q"}


def test_tool_call_observation_with_different_call_args_is_not_equal() -> None:
    a = ToolCallObservation(
        tool_name="t",
        call_args={"q": "x"},
        observation=None,
    )
    b = ToolCallObservation(
        tool_name="t",
        call_args={"q": "y"},
        observation=None,
    )
    assert a != b
