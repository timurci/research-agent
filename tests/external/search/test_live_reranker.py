"""Live network tests for the reranker agent.

Search-slice live test. Hits the local LFM ColBERT rerank endpoint via
``litellm.arerank``. Tagged ``live`` and skipped by default (see root
``conftest.py``); run explicitly with ``uv run pytest -m live``.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest
from pydantic import HttpUrl

from research_agent.search.agents import Reranker
from research_agent.search.models import (
    PaperInfo,
    PaperSource,
    ResearchQuery,
    SearchIndexReference,
    SearchIndexType,
    SearchResult,
)

if TYPE_CHECKING:
    from research_agent.shared.agent import LMConfig

_TIMEOUT_SECONDS: float = 30.0

_ABSTRACT: str = (
    "A sufficiently long abstract describing the research methodology, "
    "experimental setup, results, and conclusions of this work in detail "
    "to satisfy the PaperInfo min_length=200 invariant enforced by Pydantic."
)


def _paper(title: str) -> PaperInfo:
    return PaperInfo(
        source=PaperSource(url=HttpUrl("https://example.com/p"), open_access=False),
        title=title,
        abstract=_ABSTRACT,
        authors=("Alice",),
    )


def _result(paper: PaperInfo) -> SearchResult:
    return SearchResult(
        paper=paper,
        search_index_reference=(
            SearchIndexReference(index=SearchIndexType.ARXIV, id="1"),
        ),
    )


@pytest.mark.live
@pytest.mark.asyncio
async def test_reranker_returns_permutation_of_inputs(
    reranker_lm_config: LMConfig,
) -> None:
    """Reranker output is a permutation of the input results."""
    reranker = Reranker(reranker_lm_config)
    results = [
        _result(_paper("Paper Alpha On Machine Learning Advances")),
        _result(_paper("Paper Beta On Machine Learning Advances")),
        _result(_paper("Paper Gamma On Machine Learning Advances")),
    ]
    query = ResearchQuery(text="machine learning")

    out = await asyncio.wait_for(
        reranker((query, results)),
        timeout=_TIMEOUT_SECONDS,
    )

    assert len(out) == len(results)
    input_titles = {r.paper.title for r in results}
    output_titles = {r.paper.title for r in out}
    assert output_titles == input_titles


@pytest.mark.live
@pytest.mark.asyncio
async def test_reranker_reorders_by_relevance(
    reranker_lm_config: LMConfig,
) -> None:
    """Reranker puts the most topically relevant paper first."""
    reranker = Reranker(reranker_lm_config)
    bert_paper = _paper("BERT Pre-training of Deep Bidirectional Transformers")
    irrelevant_paper = _paper("Quantum Computing with Superconducting Qubits")
    somewhat_relevant_paper = _paper("Attention Mechanisms in Neural Networks")
    query = ResearchQuery(text="BERT language model pre-training")
    results = [
        _result(irrelevant_paper),
        _result(bert_paper),
        _result(somewhat_relevant_paper),
    ]

    out = await asyncio.wait_for(
        reranker((query, results)),
        timeout=_TIMEOUT_SECONDS,
    )

    assert len(out) == len(results)
    assert "BERT" in out[0].paper.title
