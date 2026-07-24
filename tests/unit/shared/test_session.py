"""Unit tests for Session / InMemorySession."""

from __future__ import annotations

from research_agent.shared.session import InMemorySession


def test_set_get_round_trip() -> None:
    session = InMemorySession()
    session.set("k", [1, 2])
    assert session.get("k") == [1, 2]


def test_get_missing_returns_none() -> None:
    session = InMemorySession()
    assert session.get("missing") is None


def test_get_missing_with_default() -> None:
    session = InMemorySession()
    assert session.get("missing", []) == []
    assert session.get("missing", "fallback") == "fallback"


def test_delete_existing_and_missing() -> None:
    session = InMemorySession()
    session.set("k", "v")
    session.delete("k")
    assert session.get("k") is None
    session.delete("k")
