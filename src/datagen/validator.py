"""Jaccard deduplication for the generated query set."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from research_agent.search.models import ResearchQuery

_MIN_TOKEN_LENGTH: int = 2
_JACCARD_THRESHOLD: float = 0.8


def _token_set(text: str) -> set[str]:
    return {t for t in text.lower().split() if len(t) > _MIN_TOKEN_LENGTH}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def deduplicate(
    queries: list[ResearchQuery],
) -> tuple[list[ResearchQuery], int]:
    """Remove near-identical queries (Jaccard > 0.8 on tokens)."""
    kept: list[ResearchQuery] = []
    seen_tokens: list[set[str]] = []
    removed = 0
    for query in queries:
        tokens = _token_set(query.text)
        if any(_jaccard(tokens, prev) > _JACCARD_THRESHOLD for prev in seen_tokens):
            removed += 1
            continue
        kept.append(query)
        seen_tokens.append(tokens)
    return kept, removed
