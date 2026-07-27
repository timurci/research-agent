"""Unit tests for SearchAgent optimized instruction loading."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import patch

import dspy
import pytest

from research_agent.search.agents import (
    SearchAgent,
    _SearchProgram,
    build_search_react,
)
from research_agent.search.tools import LiteratureSearch, SessionLiteratureSearch
from research_agent.shared.config.models import LMConfig
from research_agent.shared.session import InMemorySession

if TYPE_CHECKING:
    from pathlib import Path


def _signature_instructions(signature: object) -> str:
    """Return *signature.instructions* after a runtime type check."""
    instructions = getattr(signature, "instructions", None)
    assert isinstance(instructions, str)
    return instructions


def _fake_search_program_json(instructions: str) -> dict[str, object]:
    return {
        "react.react": {
            "traces": [],
            "train": [],
            "demos": [],
            "signature": {
                "instructions": instructions,
                "fields": [],
            },
            "lm": None,
        },
        "react.extract.predict": {
            "traces": [],
            "train": [],
            "demos": [],
            "signature": {
                "instructions": instructions,
                "fields": [],
            },
            "lm": None,
        },
        "metadata": {
            "dependency_versions": {
                "python": "3.14",
                "dspy": "3.2.1",
                "cloudpickle": "3.1",
            },
        },
    }


@pytest.fixture
def lm_config_fixture() -> LMConfig:
    return LMConfig(model="openai/test-model")


@pytest.fixture
def literature_search() -> LiteratureSearch:
    return LiteratureSearch()


@pytest.fixture
def session_search(literature_search: LiteratureSearch) -> SessionLiteratureSearch:
    return SessionLiteratureSearch(InMemorySession(), literature_search)


def test_search_program_has_react_predictor(
    session_search: SessionLiteratureSearch,
) -> None:
    program = _SearchProgram(session_search)

    assert isinstance(program.react, dspy.ReAct)
    assert hasattr(program.react, "react")
    assert hasattr(program.react, "extract")


def test_search_program_loads_custom_instructions(
    session_search: SessionLiteratureSearch,
    tmp_path: Path,
) -> None:
    custom = "Custom optimized instructions for testing."
    program_path = tmp_path / "search-search.json"
    program_path.write_text(
        json.dumps(_fake_search_program_json(custom)),
        encoding="utf-8",
    )

    program = _SearchProgram(session_search)
    program.load(str(program_path))

    assert _signature_instructions(program.react.react.signature) == custom
    assert _signature_instructions(program.react.extract.predict.signature) == custom


def test_search_agent_loads_instructions_when_path_given(
    lm_config_fixture: LMConfig,
    literature_search: LiteratureSearch,
    tmp_path: Path,
) -> None:
    custom = "Custom optimized instructions for testing."
    program_path = tmp_path / "search-search.json"
    program_path.write_text(
        json.dumps(_fake_search_program_json(custom)),
        encoding="utf-8",
    )

    agent = SearchAgent(
        lm_config_fixture,
        InMemorySession(),
        literature_search,
        instructions_path=program_path,
    )

    assert _signature_instructions(agent._program.react.react.signature) == custom
    assert (
        _signature_instructions(agent._program.react.extract.predict.signature)
        == custom
    )


def test_search_agent_uses_default_prompts_when_no_path(
    lm_config_fixture: LMConfig,
    literature_search: LiteratureSearch,
) -> None:
    agent = SearchAgent(
        lm_config_fixture,
        InMemorySession(),
        literature_search,
    )

    default_react = build_search_react(
        SessionLiteratureSearch(InMemorySession(), literature_search),
    )
    assert _signature_instructions(
        agent._program.react.react.signature
    ) == _signature_instructions(default_react.react.signature)
    assert _signature_instructions(
        agent._program.react.extract.predict.signature
    ) == _signature_instructions(default_react.extract.predict.signature)


def test_search_agent_calls_program_load(
    lm_config_fixture: LMConfig,
    literature_search: LiteratureSearch,
    tmp_path: Path,
) -> None:
    program_path = tmp_path / "search-search.json"
    program_path.write_text(
        json.dumps(_fake_search_program_json("x")),
        encoding="utf-8",
    )

    with patch.object(_SearchProgram, "load") as mock_load:
        SearchAgent(
            lm_config_fixture,
            InMemorySession(),
            literature_search,
            instructions_path=program_path,
        )

    mock_load.assert_called_once_with(str(program_path))


def test_search_agent_no_load_when_path_missing(
    lm_config_fixture: LMConfig,
    literature_search: LiteratureSearch,
) -> None:
    with patch.object(_SearchProgram, "load") as mock_load:
        SearchAgent(
            lm_config_fixture,
            InMemorySession(),
            literature_search,
        )

    mock_load.assert_not_called()
