"""Unit tests for Session / InMemorySession."""

from __future__ import annotations

import pytest

from research_agent.shared.session import InMemorySession, MissingSessionKeyError


def test_set_get_round_trip() -> None:
    session = InMemorySession()
    session.set("k", [1, 2])
    assert session.get("k") == [1, 2]


def test_get_missing_raises() -> None:
    session = InMemorySession()
    with pytest.raises(MissingSessionKeyError, match="missing"):
        session.get("missing")


def test_delete_existing_and_missing() -> None:
    session = InMemorySession()
    session.set("k", "v")
    session.delete("k")
    with pytest.raises(MissingSessionKeyError):
        session.get("k")
    session.delete("k")
