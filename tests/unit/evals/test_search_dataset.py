"""Unit tests for the search eval dataset loader."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from pydantic import ValidationError

from evals.search.dataset import (
    load_search_eval_data,
    load_search_queries,
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


def test_load_search_queries_maps_split() -> None:
    rows = [
        {"text": "first research question text", "domains": ["physics"]},
        {"text": "second research question text", "domains": []},
    ]

    with patch("evals.search.dataset.load_dataset", return_value=rows) as load:
        queries = load_search_queries()

    load.assert_called_once_with("tcakmako/research_queries", split="test")
    assert queries == [
        ResearchQuery(text="first research question text", domains=("physics",)),
        ResearchQuery(text="second research question text", domains=None),
    ]


def test_load_search_eval_data_mlflow_shape() -> None:
    rows = [
        {
            "text": "outline inverse probability weighting methods",
            "domains": ["statistics"],
        },
    ]

    with patch("evals.search.dataset.load_dataset", return_value=rows):
        data = load_search_eval_data()

    assert data == [
        {
            "inputs": {
                "query": ResearchQuery(
                    text="outline inverse probability weighting methods",
                    domains=("statistics",),
                ),
            },
        },
    ]


def test_load_search_queries_accepts_path_and_split_override() -> None:
    with patch("evals.search.dataset.load_dataset", return_value=[]) as load:
        load_search_queries(path="org/other", split="train")

    load.assert_called_once_with("org/other", split="train")
