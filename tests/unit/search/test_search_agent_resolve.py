"""Unit tests for session paper bag hydration."""

from __future__ import annotations

import pytest
from pydantic import HttpUrl, ValidationError

from research_agent.search.models import PaperInfo
from research_agent.search.tools import SEARCH_RESULTS_KEY, SessionLiteratureSearch
from research_agent.shared.session import InMemorySession

_ABSTRACT = (
    "A sufficiently long abstract describing the research methodology, "
    "experimental setup, results, and conclusions of this work in detail "
    "to satisfy the PaperInfo min_length=200 invariant enforced by Pydantic."
)


def _make_paper(title: str = "Alpha Paper On Quantum Computing Advances") -> PaperInfo:
    return PaperInfo(
        title=title,
        abstract=_ABSTRACT,
        authors=("Alice Smith",),
        url=HttpUrl("https://example.com/paper"),
        open_access=False,
    )


def test_papers_returns_set_from_session() -> None:
    session = InMemorySession()
    papers = {
        _make_paper("Alpha Paper On Quantum Computing Advances"),
        _make_paper("Beta Paper On Neural Network Optimization"),
    }
    session.set(SEARCH_RESULTS_KEY, papers)
    assert SessionLiteratureSearch.papers(session) == papers


def test_papers_missing_key_returns_empty() -> None:
    session = InMemorySession()
    assert SessionLiteratureSearch.papers(session) == set()


def test_papers_none_value_raises() -> None:
    session = InMemorySession()
    session.set(SEARCH_RESULTS_KEY, None)
    with pytest.raises(TypeError, match="set"):
        SessionLiteratureSearch.papers(session)


def test_papers_non_set_bag_raises() -> None:
    session = InMemorySession()
    session.set(SEARCH_RESULTS_KEY, "not-a-set")
    with pytest.raises(TypeError, match="set"):
        SessionLiteratureSearch.papers(session)


def test_papers_list_bag_is_coerced_to_set() -> None:
    session = InMemorySession()
    paper = _make_paper()
    session.set(SEARCH_RESULTS_KEY, [paper, paper])
    result = SessionLiteratureSearch.papers(session)
    assert isinstance(result, set)
    assert result == {paper}


def test_papers_non_paper_entry_raises() -> None:
    session = InMemorySession()
    session.set(SEARCH_RESULTS_KEY, {"not-a-paper"})
    with pytest.raises(ValidationError):
        SessionLiteratureSearch.papers(session)
