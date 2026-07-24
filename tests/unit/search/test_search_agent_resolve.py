"""Unit tests for search-agent session hydration."""

from __future__ import annotations

import pytest
from pydantic import HttpUrl

from research_agent.search.agents import UnknownSelectedIdError, _papers_for_ids
from research_agent.search.models import PaperInfo
from research_agent.search.tools import SEARCH_RESULTS_KEY
from research_agent.shared.session import (
    InMemorySession,
    InvalidSessionStateError,
)

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


def test_papers_for_ids_preserves_order() -> None:
    session = InMemorySession()
    papers = [
        _make_paper("Alpha Paper On Quantum Computing Advances"),
        _make_paper("Beta Paper On Neural Network Optimization"),
    ]
    session.set(SEARCH_RESULTS_KEY, papers)
    assert _papers_for_ids(session, [1, 0]) == [papers[1], papers[0]]


def test_papers_for_ids_out_of_range_raises() -> None:
    session = InMemorySession()
    session.set(SEARCH_RESULTS_KEY, [_make_paper()])
    with pytest.raises(UnknownSelectedIdError, match="out of range"):
        _papers_for_ids(session, [9])


def test_papers_for_ids_negative_id_raises() -> None:
    session = InMemorySession()
    session.set(SEARCH_RESULTS_KEY, [_make_paper()])
    with pytest.raises(UnknownSelectedIdError, match="out of range"):
        _papers_for_ids(session, [-1])


def test_papers_for_ids_len_id_raises() -> None:
    session = InMemorySession()
    papers = [_make_paper()]
    session.set(SEARCH_RESULTS_KEY, papers)
    with pytest.raises(UnknownSelectedIdError, match="out of range"):
        _papers_for_ids(session, [len(papers)])


def test_papers_for_ids_bool_id_raises() -> None:
    session = InMemorySession()
    session.set(SEARCH_RESULTS_KEY, [_make_paper(), _make_paper("Beta Paper Title")])
    selected_ids: list[int] = [True]
    with pytest.raises(UnknownSelectedIdError, match="non-bool int"):
        _papers_for_ids(session, selected_ids)


def test_papers_for_ids_missing_key_raises() -> None:
    session = InMemorySession()
    with pytest.raises(InvalidSessionStateError, match="list"):
        _papers_for_ids(session, [0])


def test_papers_for_ids_non_list_bag_raises() -> None:
    session = InMemorySession()
    session.set(SEARCH_RESULTS_KEY, "not-a-list")
    with pytest.raises(InvalidSessionStateError, match="list"):
        _papers_for_ids(session, [0])


def test_papers_for_ids_non_paper_entry_raises() -> None:
    session = InMemorySession()
    session.set(SEARCH_RESULTS_KEY, ["not-a-paper"])
    with pytest.raises(InvalidSessionStateError, match="PaperInfo"):
        _papers_for_ids(session, [0])
