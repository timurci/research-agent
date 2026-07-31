"""Unit tests for the search optimize dataset loader."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from pydantic import HttpUrl, ValidationError

from optimize.search.dataset import (
    MalformedSuggestRowError,
    SuggestInputsError,
    load_search_queries,
    load_search_trainset,
    load_suggest_trainset,
    papers_from_payload,
    research_query_from_row,
    suggestion_pair_from_opik_row,
)
from research_agent.search.models import PaperInfo, ResearchQuery
from research_agent.search.workflows import SUGGESTION_TOP_N

if TYPE_CHECKING:
    from pathlib import Path

_ABSTRACT = (
    "A sufficiently long abstract describing the research methodology, "
    "experimental setup, results, and conclusions of this work in detail "
    "to satisfy the PaperInfo min_length=200 invariant enforced by Pydantic."
)

_PAPER = {
    "title": "Alpha Paper On Quantum Computing Advances",
    "abstract": _ABSTRACT,
    "authors": ["Alice Smith"],
    "url": "https://example.com/paper",
    "open_access": False,
}


def _opik_row(
    *,
    text: str = "outline inverse probability weighting methods",
    domains: list[str] | None = None,
    papers: object | None = None,
) -> dict[str, object]:
    return {
        "dataset.query": {
            "text": text,
            "domains": domains if domains is not None else ["statistics"],
        },
        "output.papers": [_PAPER] if papers is None else papers,
        "usage.total_tokens": 100,
    }


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
    with pytest.raises(MalformedSuggestRowError, match="list\\[str\\]"):
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


def test_papers_from_payload_validates_paper_info() -> None:
    papers = papers_from_payload([_PAPER])
    assert len(papers) == 1
    assert isinstance(papers[0], PaperInfo)


def test_suggestion_pair_from_opik_row_maps_keys() -> None:
    query, papers = suggestion_pair_from_opik_row(_opik_row(papers=[_PAPER]))
    assert query.domains == ("statistics",)
    assert len(papers) == 1


def test_load_suggest_trainset_dspy_shape(tmp_path: Path) -> None:
    export_path = tmp_path / "export.json"
    export_path.write_text(
        json.dumps(
            [
                _opik_row(papers=[_PAPER]),
                _opik_row(text="skipped missing papers", papers="-"),
            ],
        ),
        encoding="utf-8",
    )

    examples = load_suggest_trainset(path=export_path)

    assert len(examples) == 1
    assert examples[0].research_query == ResearchQuery(
        text="outline inverse probability weighting methods",
        domains=("statistics",),
    )
    assert len(examples[0].papers) == 1
    assert examples[0].papers[0].url == HttpUrl("https://example.com/paper")
    assert sorted(examples[0].inputs()) == ["papers", "research_query"]


def test_load_suggest_trainset_truncates_to_top_n(tmp_path: Path) -> None:
    many = [
        {
            **_PAPER,
            "title": f"Paper Number {index:03d} Title Here",
            "url": f"https://example.com/p{index}",
        }
        for index in range(SUGGESTION_TOP_N + 7)
    ]
    export_path = tmp_path / "export.json"
    export_path.write_text(
        json.dumps([_opik_row(papers=many)]),
        encoding="utf-8",
    )

    examples = load_suggest_trainset(path=export_path)
    assert len(examples[0].papers) == SUGGESTION_TOP_N


def test_load_suggest_trainset_missing_file(tmp_path: Path) -> None:
    with pytest.raises(SuggestInputsError, match="not found"):
        load_suggest_trainset(path=tmp_path / "missing.json")
