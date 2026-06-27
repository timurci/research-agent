"""CLI entrypoint for the synthetic query generation pipeline.

Usage:
    python -m datagen.main --model MODEL --api-key KEY
                           [--out-dir DIR] [--limit N]
                           [--queries-per-stratum N]
                           [--reasoning-effort LEVEL]
                           [--extra-body JSON]

Generates synthetic ResearchQuery objects via per-domain LLM sessions
and writes them to queries_train.jsonl, one query per line. The output
is consumed by the optimization pipeline in `src/optimize/`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import TYPE_CHECKING, Any

from datagen.config import (
    DOMAINS,
    QUERIES_PER_STRATUM,
    GenerationConfig,
)
from datagen.llm_client import LLMClient
from datagen.query_generator import QueryGenerator
from datagen.validator import deduplicate
from datagen.writer import write

if TYPE_CHECKING:
    from research_agent.search.models import ResearchQuery


async def run(config: GenerationConfig) -> None:
    """Run the per-domain batch generation pipeline."""
    client = LLMClient(
        config.llm_model,
        config.api_key,
        reasoning_effort=config.reasoning_effort,
        extra_body=config.extra_body,
    )
    query_gen = QueryGenerator(client)

    items: list[tuple[str, ResearchQuery]] = []
    produced = 0

    for domain in DOMAINS:
        if config.limit is not None and produced >= config.limit:
            break
        try:
            batch = await query_gen.generate_batch(domain, n=config.queries_per_stratum)
        except Exception as exc:  # noqa: BLE001  # skip-and-log is the intended behavior
            print(  # noqa: T201  # CLI status output, not logging
                f"[skip] batch generation failed for {domain}: {exc}",
                file=sys.stderr,
            )
            continue
        for query in batch:
            items.append((domain, query))
            produced += 1
        print(f"[ok] {domain} +{len(batch)} (total {produced})")  # noqa: T201  # CLI status output, not logging

    queries = [q for _, q in items]
    queries, removed = deduplicate(queries)
    if removed:
        print(f"[dedup] removed {removed} near-duplicate queries")  # noqa: T201  # CLI status output, not logging

    path = write(queries, config.out_dir)
    print(f"[done] wrote {path}")  # noqa: T201  # CLI status output, not logging


def _parse_extra_body(raw: str | None) -> dict[str, Any] | None:
    """Parse the ``--extra-body`` JSON string into a dict.

    Returns ``None`` when no value was supplied. Raises ``SystemExit(2)``
    with a clear stderr message when the value is set but malformed or
    does not decode to a JSON object.
    """
    if raw is None:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(  # noqa: T201  # CLI status output, not logging
            f"[error] --extra-body must be valid JSON: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc
    if not isinstance(parsed, dict):
        print(  # noqa: T201  # CLI status output, not logging
            "[error] --extra-body must decode to a JSON object",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return parsed


def main() -> None:
    """Parse CLI args and run the per-domain generation pipeline."""
    parser = argparse.ArgumentParser(description="Generate synthetic search queries.")
    parser.add_argument("--out-dir", default="data/datagen/output")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap number of queries. Cuts mid-batch; useful only for smoke runs.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help='LLM model string in LiteLLM format (e.g. "openai/gpt-4o-mini").',
    )
    parser.add_argument(
        "--api-key",
        required=True,
        help="API key for the LLM provider.",
    )
    parser.add_argument(
        "--queries-per-stratum",
        type=int,
        default=QUERIES_PER_STRATUM,
        help="Queries per (intent, specificity) pair per domain.",
    )
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        help=(
            "Reasoning effort for reasoning-capable models "
            '(e.g. "low", "medium", "high", "minimal", "none"). '
            "Forwarded to LiteLLM as reasoning_effort=... "
            "and ignored by models that do not support it."
        ),
    )
    parser.add_argument(
        "--extra-body",
        default=None,
        help=(
            "JSON object merged into the LiteLLM request body. "
            "OpenRouter provider routing with fallbacks: "
            '\'{"provider": {"order": ["Anthropic", "OpenAI"], '
            '"allow_fallbacks": true}}\'. '
            "OpenRouter provider routing without fallbacks: "
            '\'{"provider": {"order": ["openai", "together"], '
            '"allow_fallbacks": false}}\'.'
        ),
    )
    args = parser.parse_args()

    config = GenerationConfig(
        llm_model=args.model,
        api_key=args.api_key,
        out_dir=args.out_dir,
        limit=args.limit,
        queries_per_stratum=args.queries_per_stratum,
        reasoning_effort=args.reasoning_effort,
        extra_body=_parse_extra_body(args.extra_body),
    )
    asyncio.run(run(config))


if __name__ == "__main__":
    main()
