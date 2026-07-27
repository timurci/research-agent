"""Unit tests for ``research_agent.search.agents``."""

from __future__ import annotations

from research_agent.search.agents import SearchOutcome


def test_search_outcome_values() -> None:
    assert SearchOutcome.COMPLETE == "complete"
    assert {member.value for member in SearchOutcome} == {
        "complete",
        "insufficient_search",
        "irrelevant_results",
        "missing_results",
        "tool_error",
    }
