"""Unit tests for ``research_agent.search.agents``."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import patch

import dspy
import pytest
from pydantic import HttpUrl

from research_agent.search.agents import (
    Reranker,
    SearchOutcome,
    SuggestionGenerator,
    _SuggestionGeneratorProgram,
)
from research_agent.search.models import PaperInfo, ResearchQuery
from research_agent.shared.config.models import LMConfig

if TYPE_CHECKING:
    from pathlib import Path

    from research_agent.shared.rerank import RerankScore


def test_search_outcome_values() -> None:
    assert SearchOutcome.COMPLETE == "complete"
    assert {member.value for member in SearchOutcome} == {
        "complete",
        "insufficient_search",
        "irrelevant_results",
        "missing_results",
        "tool_error",
    }


def _fake_suggestion_program_json(instructions: str) -> dict[str, object]:
    return {
        "predict": {
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


def _signature_instructions(signature: object) -> str:
    """Return *signature.instructions* after a runtime type check."""
    instructions = getattr(signature, "instructions", None)
    assert isinstance(instructions, str)
    return instructions


class _FakeRerankClient:
    """Rerank client stub recording payloads and returning fixed scores."""

    def __init__(self, scores: list[RerankScore]) -> None:
        self._scores = scores
        self.calls: list[tuple[str, list[str]]] = []

    async def rerank(self, *, query: str, documents: list[str]) -> list[RerankScore]:
        self.calls.append((query, documents))
        return self._scores


def _make_paper(title: str) -> PaperInfo:
    return PaperInfo(
        title=title,
        abstract=(
            "A sufficiently long abstract describing the research methodology, "
            "experimental setup, results, and conclusions of this work in detail "
            "to satisfy the PaperInfo min_length=200 invariant enforced by Pydantic."
        ),
        authors=("Alice",),
        url=HttpUrl("https://example.com/paper"),
        open_access=False,
    )


@pytest.fixture
def lm_config_fixture() -> LMConfig:
    return LMConfig(model="openai/test-model")


@pytest.fixture
def research_query() -> ResearchQuery:
    return ResearchQuery(text="quantum computing")


@pytest.fixture
def paper() -> PaperInfo:
    return PaperInfo(
        title="Alpha Paper On Quantum Computing Advances",
        abstract=(
            "A sufficiently long abstract describing the research methodology, "
            "experimental setup, results, and conclusions of this work in detail "
            "to satisfy the PaperInfo min_length=200 invariant enforced by Pydantic."
        ),
        authors=("Alice",),
        url=HttpUrl("https://example.com/paper"),
        open_access=False,
    )


def test_suggestion_program_has_predictor() -> None:
    program = _SuggestionGeneratorProgram()

    assert isinstance(program.predict, dspy.Predict)


def test_suggestion_program_loads_custom_instructions(
    tmp_path: Path,
) -> None:
    custom = "Custom optimized suggestion instructions."
    program_path = tmp_path / "suggest.json"
    program_path.write_text(
        json.dumps(_fake_suggestion_program_json(custom)),
        encoding="utf-8",
    )

    program = _SuggestionGeneratorProgram()
    program.load(str(program_path))

    assert _signature_instructions(program.predict.signature) == custom


def test_suggestion_generator_loads_instructions_when_path_given(
    lm_config_fixture: LMConfig,
    tmp_path: Path,
) -> None:
    custom = "Custom optimized suggestion instructions."
    program_path = tmp_path / "suggest.json"
    program_path.write_text(
        json.dumps(_fake_suggestion_program_json(custom)),
        encoding="utf-8",
    )

    agent = SuggestionGenerator(
        lm_config_fixture,
        instructions_path=program_path,
    )

    assert _signature_instructions(agent._program.predict.signature) == custom


def test_suggestion_generator_calls_program_load(
    lm_config_fixture: LMConfig,
    tmp_path: Path,
) -> None:
    program_path = tmp_path / "suggest.json"
    program_path.write_text(
        json.dumps(_fake_suggestion_program_json("x")),
        encoding="utf-8",
    )

    with patch.object(_SuggestionGeneratorProgram, "load") as mock_load:
        SuggestionGenerator(
            lm_config_fixture,
            instructions_path=program_path,
        )

    mock_load.assert_called_once_with(str(program_path))


def test_suggestion_generator_no_load_when_path_missing(
    lm_config_fixture: LMConfig,
) -> None:
    with patch.object(_SuggestionGeneratorProgram, "load") as mock_load:
        SuggestionGenerator(lm_config_fixture)

    mock_load.assert_not_called()


@pytest.mark.asyncio
async def test_suggestion_generator_returns_suggestion(
    lm_config_fixture: LMConfig,
    research_query: ResearchQuery,
    paper: PaperInfo,
) -> None:
    agent = SuggestionGenerator(lm_config_fixture)

    with patch.object(
        agent._program,
        "aforward",
        return_value="focus on quantum error correction",
    ):
        result = await agent((research_query, [paper]))

    assert result == "focus on quantum error correction"


def test_reranker_builds_client_from_config() -> None:
    config = LMConfig(model="openrouter/cohere/rerank-x")

    with patch("research_agent.search.agents.build_rerank_client") as builder:
        Reranker(config)

    builder.assert_called_once_with(config)


@pytest.mark.asyncio
async def test_reranker_orders_papers_by_client_scores(
    research_query: ResearchQuery,
) -> None:
    alpha = _make_paper("Alpha Paper")
    beta = _make_paper("Beta Paper")
    reranker = Reranker(LMConfig(model="infinity/test-rerank"))
    reranker._client = _FakeRerankClient(
        [
            {"index": 1, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.1},
        ]
    )

    out = await reranker((research_query, [alpha, beta]))

    assert [p.title for p in out] == ["Beta Paper", "Alpha Paper"]


@pytest.mark.asyncio
async def test_reranker_passes_domains_and_abstracts_to_client(
    paper: PaperInfo,
) -> None:
    query = ResearchQuery(text="quantum computing", domains=["physics", "cs"])
    fake = _FakeRerankClient([{"index": 0, "relevance_score": 0.5}])
    reranker = Reranker(LMConfig(model="infinity/test-rerank"))
    reranker._client = fake

    await reranker.relevance((query, [paper]))

    assert fake.calls == [
        (
            "Query: quantum computing; Domains: physics, cs",
            [f"Title: {paper.title}; Abstract: {paper.abstract[:2048]}"],
        )
    ]
