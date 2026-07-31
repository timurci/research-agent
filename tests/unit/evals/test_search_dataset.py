"""Unit tests for the search eval dataset loader."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from pydantic import HttpUrl, ValidationError

from evals.search.dataset import (
    MalformedSuggestRowError,
    SuggestInputsError,
    load_search_eval_data,
    load_search_queries,
    load_suggest_eval_data,
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


def test_load_search_eval_data_flat_shape() -> None:
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
            "query": ResearchQuery(
                text="outline inverse probability weighting methods",
                domains=("statistics",),
            ).model_dump(),
        },
    ]


def test_load_search_queries_accepts_path_and_split_override() -> None:
    with patch("evals.search.dataset.load_dataset", return_value=[]) as load:
        load_search_queries(path="org/other", split="train")

    load.assert_called_once_with("org/other", split="train")


def test_papers_from_payload_validates_paper_info() -> None:
    papers = papers_from_payload([_PAPER])
    assert papers == [
        PaperInfo(
            title="Alpha Paper On Quantum Computing Advances",
            abstract=_ABSTRACT,
            authors=("Alice Smith",),
            url=HttpUrl("https://example.com/paper"),
            open_access=False,
        ),
    ]


def test_papers_from_payload_rejects_non_list() -> None:
    with pytest.raises(MalformedSuggestRowError, match="papers must be list"):
        papers_from_payload("-")


def test_papers_from_payload_rejects_invalid_payload() -> None:
    with pytest.raises(MalformedSuggestRowError, match="list\\[PaperInfo\\]"):
        papers_from_payload([{"title": "short"}])


def test_suggestion_pair_from_opik_row_maps_keys() -> None:
    query, papers = suggestion_pair_from_opik_row(
        _opik_row(papers=[_PAPER]),
    )
    assert query.text == "outline inverse probability weighting methods"
    assert query.domains == ("statistics",)
    assert len(papers) == 1
    assert papers[0].title == "Alpha Paper On Quantum Computing Advances"


def test_suggestion_pair_from_opik_row_truncates_papers() -> None:
    many = [
        {
            **_PAPER,
            "title": f"Paper Number {index:03d} Title Here",
            "url": f"https://example.com/p{index}",
        }
        for index in range(SUGGESTION_TOP_N + 5)
    ]
    _, papers = suggestion_pair_from_opik_row(
        _opik_row(papers=many),
        paper_limit=SUGGESTION_TOP_N,
    )
    assert len(papers) == SUGGESTION_TOP_N


def test_load_suggest_eval_data_from_local_opik_export(tmp_path: Path) -> None:
    export_path = tmp_path / "export.json"
    export_path.write_text(
        json.dumps(
            [
                _opik_row(papers=[_PAPER]),
                _opik_row(
                    text="row with missing papers should be skipped",
                    papers="-",
                ),
                {
                    "dataset.query": {
                        "text": "second valid research query text",
                        "domains": [],
                    },
                    "output.papers": [_PAPER],
                },
            ],
        ),
        encoding="utf-8",
    )

    data = load_suggest_eval_data(path=export_path)

    assert len(data) == 2
    query = ResearchQuery.model_validate(data[0]["query"])
    raw_papers = data[0]["papers"]
    assert isinstance(raw_papers, list)
    papers = [PaperInfo.model_validate(paper) for paper in raw_papers]
    assert query == ResearchQuery(
        text="outline inverse probability weighting methods",
        domains=("statistics",),
    )
    assert len(papers) == 1
    assert papers[0].title == "Alpha Paper On Quantum Computing Advances"
    second = ResearchQuery.model_validate(data[1]["query"])
    assert second.text == "second valid research query text"
    assert second.domains is None


def test_load_suggest_eval_data_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(SuggestInputsError, match="not found"):
        load_suggest_eval_data(path=missing)
