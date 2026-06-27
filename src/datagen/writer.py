"""Writes generated queries to a JSONL training file."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from research_agent.search.models import ResearchQuery


def write(queries: list[ResearchQuery], out_dir: str) -> Path:
    """Write the query training file and return its path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    path = out / "queries_train.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for query in queries:
            f.write(query.model_dump_json() + "\n")

    return path
