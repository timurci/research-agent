"""Tests for the Jaccard deduplication in `datagen.validator`."""

from __future__ import annotations

from datagen.validator import deduplicate
from research_agent.search.models import ResearchQuery


def test_deduplicate_drops_jaccard_above_threshold() -> None:
    """Near-identical queries (Jaccard > 0.8 on tokens) collapse to one."""
    a = ResearchQuery(text="graph neural networks for molecular property prediction")
    b = ResearchQuery(
        text="graph neural networks for molecular property prediction summary"
    )
    c = ResearchQuery(text="transformer scaling laws in language models")
    kept, removed = deduplicate([a, b, c])
    assert removed == 1
    assert len(kept) == 2
    assert c in kept


def test_deduplicate_keeps_unrelated_queries() -> None:
    """Queries that are not near-duplicates are all kept."""
    a = ResearchQuery(text="crispr gene editing in cancer therapy")
    b = ResearchQuery(text="monetary policy impacts on emerging markets")
    c = ResearchQuery(text="transformer architectures for low-resource languages")
    kept, removed = deduplicate([a, b, c])
    assert removed == 0
    assert kept == [a, b, c]


def test_deduplicate_keeps_short_queries_with_different_words() -> None:
    """Short queries that share only stop-words are not considered duplicates.

    The tokeniser drops tokens shorter than 3 characters, so "in" and
    "of" do not contribute to the Jaccard score.
    """
    a = ResearchQuery(text="graph neural networks")
    b = ResearchQuery(text="transformer architectures")
    kept, removed = deduplicate([a, b])
    assert removed == 0
    assert kept == [a, b]
