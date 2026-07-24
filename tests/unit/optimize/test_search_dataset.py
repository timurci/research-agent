"""Unit tests for the search optimize dataset loader."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from pydantic import ValidationError

from optimize.search.dataset import (
    load_search_queries,
    load_search_trainset,
    research_query_from_row,
)
from research_agent.search.models import ResearchQuery


def test_research_query_from_row_with_domains() -> None:
    query = research_query_from_row(
        {
            "text": "causal inference with inverse probability weighting",
            "domains": ["statistics", "causal inference"],
        },
    )
    assert query == ResearchQuery(
        text="causal inference with inverse probability weighting",
        domains=("statistics", "causal inference"),
    )


def test_research_query_from_row_empty_domains_becomes_none() -> None:
    query = research_query_from_row({"text": "civil society topics", "domains": []})
    assert query.domains is None


def test_research_query_from_row_missing_domains() -> None:
    query = research_query_from_row({"text": "civil society topics"})
    assert query.domains is None


def test_research_query_from_row_rejects_short_text() -> None:
    with pytest.raises(ValidationError):
        research_query_from_row({"text": "ab", "domains": None})


def test_research_query_from_row_rejects_bad_domains() -> None:
    with pytest.raises(TypeError, match="list\\[str\\]"):
        research_query_from_row({"text": "long enough query text", "domains": [1]})


def test_load_search_queries_maps_train_split() -> None:
    rows = [
        {"text": "first research question text", "domains": ["physics"]},
        {"text": "second research question text", "domains": []},
    ]

    with patch("optimize.search.dataset.load_dataset", return_value=rows) as load:
        queries = load_search_queries()

    load.assert_called_once_with("tcakmako/research_queries", split="train")
    assert queries == [
        ResearchQuery(text="first research question text", domains=("physics",)),
        ResearchQuery(text="second research question text", domains=None),
    ]


def test_load_search_trainset_dspy_shape() -> None:
    rows = [
        {
            "text": "outline inverse probability weighting methods",
            "domains": ["statistics"],
        },
    ]

    with patch("optimize.search.dataset.load_dataset", return_value=rows):
        examples = load_search_trainset()

    assert len(examples) == 1
    assert examples[0].research_query == ResearchQuery(
        text="outline inverse probability weighting methods",
        domains=("statistics",),
    )
    assert list(examples[0].inputs()) == ["research_query"]
