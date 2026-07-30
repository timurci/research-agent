"""Unit tests for search optimize agents and session isolation."""

from __future__ import annotations

import copy
import logging
from unittest.mock import MagicMock, patch

import dspy
import pytest
from pydantic import HttpUrl

from optimize.search.agents import (
    SearchProgram,
    SuggestionProgram,
    relevance_labeler,
    search_agent,
)
from research_agent.search.agents import Reranker
from research_agent.search.models import PaperInfo, ResearchQuery
from research_agent.shared.config.models import LMConfig

_LOAD_LM_CONFIG = "optimize.search.agents.load_lm_config"

_SEARCH_CONFIG = LMConfig(
    model="openai/test-search",
    api_key="search-key",
    base_url=HttpUrl("http://search.example/v1"),
)
_RERANK_CONFIG = LMConfig(
    model="infinity/test-rerank",
    api_key="rerank-key",
    base_url=HttpUrl("http://rerank.example/v1"),
)
_SUGGEST_CONFIG = LMConfig(
    model="openai/test-suggest",
    api_key="suggest-key",
    base_url=HttpUrl("http://suggest.example/v1"),
)

_ABSTRACT = (
    "A sufficiently long abstract describing the research methodology, "
    "experimental setup, results, and conclusions of this work in detail "
    "to satisfy the PaperInfo min_length=200 invariant enforced by Pydantic."
)


def test_search_agent_is_callable() -> None:
    assert callable(search_agent(lm_config=_SEARCH_CONFIG))


def test_relevance_labeler_returns_reranker() -> None:
    assert isinstance(relevance_labeler(lm_config=_RERANK_CONFIG), Reranker)


def test_search_program_is_dspy_module_with_react() -> None:
    program = SearchProgram(lm_config=_SEARCH_CONFIG)
    assert isinstance(program, dspy.Module)
    names = [name for name, _ in program.named_predictors()]
    assert names
    assert any(name.startswith("react") for name in names)


def test_suggestion_program_is_dspy_module_with_predict() -> None:
    program = SuggestionProgram(lm_config=_SUGGEST_CONFIG)
    assert isinstance(program, dspy.Module)
    names = [name for name, _ in program.named_predictors()]
    assert names
    assert any(name.startswith("predict") for name in names)


def test_suggestion_program_forward_uses_predict() -> None:
    program = SuggestionProgram(lm_config=_SUGGEST_CONFIG)
    query = ResearchQuery(text="quantum error correction codes")
    papers = [
        PaperInfo(
            title="Alpha Paper On Quantum Computing Advances",
            abstract=_ABSTRACT,
            authors=("Alice Smith",),
            url=HttpUrl("https://example.com/paper"),
            open_access=False,
        ),
    ]
    fake_pred = dspy.Prediction(suggestion="read the survey")
    with patch.object(program, "predict", return_value=fake_pred) as predict:
        result = program.forward(research_query=query, papers=papers)

    predict.assert_called_once_with(research_query=query, papers=papers)
    assert result.suggestion == "read the survey"


@pytest.mark.asyncio
async def test_search_agent_delegates_to_search_program() -> None:
    query = ResearchQuery(text="quantum error correction codes")
    program = MagicMock(
        return_value=dspy.Prediction(search_results=[]),
    )

    with patch("optimize.search.agents.SearchProgram", return_value=program) as cls:
        agent = search_agent(lm_config=_SEARCH_CONFIG)
        result = await agent(query)

    cls.assert_called_once()
    assert cls.call_args.kwargs["lm_config"] is _SEARCH_CONFIG
    program.assert_called_once_with(research_query=query)
    assert result == []


@pytest.mark.asyncio
async def test_search_agent_defaults_to_yaml_search_role() -> None:
    yaml_config = LMConfig(model="openai/from-yaml-search")
    program = MagicMock(return_value=dspy.Prediction(search_results=[]))

    with (
        patch(_LOAD_LM_CONFIG, return_value=yaml_config) as load,
        patch("optimize.search.agents.SearchProgram", return_value=program) as cls,
    ):
        agent = search_agent()
        await agent(ResearchQuery(text="default config check query text"))

    load.assert_called_once_with("search-search")
    assert cls.call_args.kwargs["lm_config"] is yaml_config


@pytest.mark.asyncio
async def test_search_agent_uses_injected_lm_config() -> None:
    custom = LMConfig(
        model="openai/custom-search",
        api_key="search-key",
        base_url=HttpUrl("http://search.example/v1"),
    )
    program = MagicMock(return_value=dspy.Prediction(search_results=[]))

    with patch("optimize.search.agents.SearchProgram", return_value=program) as cls:
        agent = search_agent(lm_config=custom)
        await agent(ResearchQuery(text="injected config check query text"))

    assert cls.call_args.kwargs["lm_config"] is custom


def test_relevance_labeler_defaults_to_yaml_rerank_role() -> None:
    yaml_config = LMConfig(model="infinity/from-yaml-rerank")
    with (
        patch(_LOAD_LM_CONFIG, return_value=yaml_config) as load,
        patch("optimize.search.agents.Reranker") as reranker_cls,
    ):
        reranker_cls.return_value = object()
        relevance_labeler()

    load.assert_called_once_with("search-rerank")
    assert reranker_cls.call_args.args[0] is yaml_config


def test_relevance_labeler_uses_injected_lm_config() -> None:
    custom = LMConfig(
        model="infinity/custom-rerank",
        api_key="rerank-key",
        base_url=HttpUrl("http://rerank.example/v1"),
    )

    with patch("optimize.search.agents.Reranker") as reranker_cls:
        reranker_cls.return_value = object()
        relevance_labeler(lm_config=custom)

    assert reranker_cls.call_args.args[0] is custom


def test_search_program_deepcopy_emits_no_warnings(
    caplog: pytest.LogCaptureFixture,
) -> None:
    program = SearchProgram(lm_config=_SEARCH_CONFIG)

    with caplog.at_level(logging.WARNING, logger="root"):
        copy.deepcopy(program)

    deep_copy_warnings = [
        record for record in caplog.records if "Failed to deep copy" in record.message
    ]
    assert deep_copy_warnings == []


def test_search_program_deepcopy_preserves_predictor_state() -> None:
    program = SearchProgram(lm_config=_SEARCH_CONFIG)
    original_predictors = list(program.named_predictors())
    original_pred = original_predictors[0][1]
    original_signature = original_pred.signature
    original_instructions = original_signature.instructions

    copy_program = copy.deepcopy(program)
    copy_predictors = list(copy_program.named_predictors())
    copy_pred = copy_predictors[0][1]
    copy_signature = copy_pred.signature

    assert copy_predictors[0][0] == original_predictors[0][0]
    assert copy_pred is not original_pred
    assert copy_signature is original_signature
    assert copy_signature.instructions == original_instructions

    copy_pred.signature = copy_signature.with_instructions("MUTATED IN COPY")
    assert original_pred.signature is original_signature
    assert original_signature.instructions == original_instructions
