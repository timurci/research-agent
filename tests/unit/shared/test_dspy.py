"""Unit tests for ``research_agent.shared.dspy``."""

from __future__ import annotations

from research_agent.shared.dspy import extract_tool_calls
from research_agent.shared.metric import ToolCallObservation


def test_extract_tool_calls_returns_empty_list_for_empty_trajectory() -> None:
    assert extract_tool_calls({}) == []


def test_extract_tool_calls_extracts_multiple_steps() -> None:
    trajectory = {
        "thought_0": "first thought",
        "tool_name_0": "LiteratureSearch",
        "tool_args_0": {"query": "q", "limit": 5},
        "observation_0": ["result"],
        "thought_1": "second thought",
        "tool_name_1": "finish",
        "tool_args_1": {},
        "observation_1": None,
    }
    calls = extract_tool_calls(trajectory)
    assert calls == [
        ToolCallObservation(
            tool_name="LiteratureSearch",
            call_args={"query": "q", "limit": 5},
            observation=["result"],
        ),
        ToolCallObservation(
            tool_name="finish",
            call_args={},
            observation=None,
        ),
    ]


def test_extract_tool_calls_defaults_empty_args() -> None:
    trajectory = {
        "tool_name_0": "LiteratureSearch",
        "observation_0": None,
    }
    calls = extract_tool_calls(trajectory)
    assert calls == [
        ToolCallObservation(
            tool_name="LiteratureSearch",
            call_args={},
            observation=None,
        ),
    ]
