"""Search-suite evaluation dataset loaders.

* **search-search** — Hugging Face ``tcakmako/research_queries`` **test**
  split; query-only rows (no gold papers or relevance labels).
* **search-suggest** — local Opik search-search I/O export (query + papers
  captured from a prior eval run). Default path:
  ``data/optimize/input/eval-search-search-io.json``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from datasets import load_dataset
from pydantic import TypeAdapter, ValidationError

from research_agent.search.models import PaperInfo, ResearchQuery
from research_agent.search.workflows import SUGGESTION_TOP_N

SEARCH_HF_PATH: str = "tcakmako/research_queries"
SEARCH_SPLIT: str = "test"

DEFAULT_SUGGEST_INPUTS_PATH: Path = Path(
    "data/optimize/input/eval-search-search-io.json",
)

_OPIK_QUERY_KEY: str = "dataset.query"
_OPIK_PAPERS_KEY: str = "output.papers"

_PAPER_LIST_ADAPTER: TypeAdapter[list[PaperInfo]] = TypeAdapter(list[PaperInfo])


class SuggestInputsError(Exception):
    """Raised when the suggestion-inputs export is missing or malformed."""


class MalformedSuggestRowError(Exception):
    """Raised when a row or payload violates the expected input schema."""


def research_query_from_row(row: Mapping[str, object]) -> ResearchQuery:
    """Map one HF row to a ``ResearchQuery``.

    Expects ``text: str`` and optional ``domains: list[str]``.

    Raises:
        MalformedSuggestRowError: If *row* fields have the wrong type.
    """
    text, domains = row["text"], row.get("domains")
    if not isinstance(text, str):
        msg = f"text must be str, got {type(text).__name__}"
        raise MalformedSuggestRowError(msg)
    if domains is not None and not (
        isinstance(domains, list) and all(isinstance(d, str) for d in domains)
    ):
        msg = "domains must be list[str] or None"
        raise MalformedSuggestRowError(msg)
    return ResearchQuery.model_validate(
        {"text": text, "domains": domains or None},
    )


def papers_from_payload(raw: object, *, limit: int | None = None) -> list[PaperInfo]:
    """Validate a paper-list payload as ``list[PaperInfo]``.

    Args:
        raw: JSON-decoded paper list.
        limit: Optional max length (e.g. ``SUGGESTION_TOP_N``).

    Raises:
        MalformedSuggestRowError: If *raw* is not a list of valid paper
            objects.
    """
    if not isinstance(raw, list):
        msg = f"papers must be list, got {type(raw).__name__}"
        raise MalformedSuggestRowError(msg)
    payload = raw if limit is None else raw[:limit]
    try:
        return _PAPER_LIST_ADAPTER.validate_python(payload)
    except ValidationError as exc:
        msg = f"papers must be list[PaperInfo]: {exc}"
        raise MalformedSuggestRowError(msg) from exc


def suggestion_pair_from_opik_row(
    row: Mapping[str, object],
    *,
    paper_limit: int = SUGGESTION_TOP_N,
) -> tuple[ResearchQuery, list[PaperInfo]]:
    """Map one Opik search-search I/O export row to domain inputs.

    Expects ``dataset.query`` (``ResearchQuery`` fields) and
    ``output.papers`` (``list[PaperInfo]`` fields). Papers are truncated
    to *paper_limit* to match the runtime suggestion step.

    Raises:
        MalformedSuggestRowError: If *row* does not match that shape.
    """
    query_raw = row[_OPIK_QUERY_KEY]
    if not isinstance(query_raw, dict):
        msg = f"{_OPIK_QUERY_KEY} must be a dict, got {type(query_raw).__name__}"
        raise MalformedSuggestRowError(msg)
    query_fields: dict[str, object] = {
        str(key): value for key, value in query_raw.items()
    }
    query = research_query_from_row(query_fields)
    papers = papers_from_payload(row[_OPIK_PAPERS_KEY], limit=paper_limit)
    return query, papers


def load_search_queries(
    *,
    path: str = SEARCH_HF_PATH,
    split: str = SEARCH_SPLIT,
) -> list[ResearchQuery]:
    """Load search-suite queries from Hugging Face.

    Args:
        path: Hugging Face dataset path.
        split: Dataset split name (default ``test``).

    Returns:
        Domain queries in dataset order.
    """
    dataset = load_dataset(path, split=split)
    return [research_query_from_row(row) for row in dataset]


def load_search_eval_data(
    *,
    path: str = SEARCH_HF_PATH,
    split: str = SEARCH_SPLIT,
) -> list[dict[str, object]]:
    """Load Opik ``evaluate`` rows for the search suite.

    Each row is query-only::

        {"query": ResearchQuery(...)}

    Args:
        path: Hugging Face dataset path.
        split: Dataset split name (default ``test``).

    Returns:
        Rows suitable for ``opik.evaluate(dataset=...)``.
    """
    queries = load_search_queries(path=path, split=split)
    return [{"query": q.model_dump()} for q in queries]


def load_suggest_pairs(
    *,
    path: Path = DEFAULT_SUGGEST_INPUTS_PATH,
    paper_limit: int = SUGGESTION_TOP_N,
) -> list[tuple[ResearchQuery, list[PaperInfo]]]:
    """Load query+papers pairs from a local Opik search I/O export.

    Rows whose ``output.papers`` is missing, not a list, empty, or not
    valid ``PaperInfo`` values are skipped (Opik exports use ``"-"`` for
    absent outputs).

    Args:
        path: JSON file path (list of export objects).
        paper_limit: Max papers kept per row (default runtime top-N).

    Returns:
        Domain ``(query, papers)`` pairs in file order.

    Raises:
        SuggestInputsError: If the file is missing or not a JSON list.
    """
    if not path.is_file():
        msg = (
            f"suggestion inputs file not found: {path}. "
            "Export a search-search Opik run (dataset.query + "
            "output.papers) into this path."
        )
        raise SuggestInputsError(msg)

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        msg = f"failed to read suggestion inputs from {path}: {exc}"
        raise SuggestInputsError(msg) from exc

    if not isinstance(raw, list):
        msg = (
            f"suggestion inputs must be a JSON list, got {type(raw).__name__} in {path}"
        )
        raise SuggestInputsError(msg)

    pairs: list[tuple[ResearchQuery, list[PaperInfo]]] = []
    for row in raw:
        if not isinstance(row, Mapping):
            continue
        try:
            query, papers = suggestion_pair_from_opik_row(
                row,
                paper_limit=paper_limit,
            )
        except KeyError, MalformedSuggestRowError, ValidationError:
            continue
        if papers:
            pairs.append((query, papers))
    return pairs


def load_suggest_eval_data(
    *,
    path: Path = DEFAULT_SUGGEST_INPUTS_PATH,
    paper_limit: int = SUGGESTION_TOP_N,
) -> list[dict[str, object]]:
    """Load Opik ``evaluate`` rows for the suggestion suite.

    Each row carries fixed paper inputs::

        {"query": ResearchQuery(...), "papers": list[PaperInfo]}

    Loaded from a local Opik search-search I/O export (not HF). No gold
    suggestion text.

    Args:
        path: Local JSON export path.
        paper_limit: Max papers kept per row.

    Returns:
        Rows suitable for ``opik.evaluate(dataset=...)``.
    """
    return [
        {
            "query": query.model_dump(mode="json"),
            "papers": [paper.model_dump(mode="json") for paper in papers],
        }
        for query, papers in load_suggest_pairs(path=path, paper_limit=paper_limit)
    ]
