"""Static configuration for the synthetic query generation pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from pydantic import TypeAdapter

from research_agent.search.models import ResearchQuery

INTENTS: dict[str, str] = {
    "literature review": "a survey-style request",
    "known-item lookup": "looking for a specific paper",
    "methodology search": "asking about methods/techniques",
    "recent advances survey": "what's new in the area",
}

SPECIFICITY_LEVELS: dict[str, str] = {
    "vague": "1-3 words, very broad",
    "moderate": "1-2 sentences with some scope constraints",
    "detailed": "2-4 sentences with explicit scope, time range, or method",
}

DOMAINS: list[str] = [
    "computer_science",
    "artificial_intelligence",
    "machine_learning",
    "biomedicine",
    "genetics",
    "neuroscience",
    "physics",
    "chemistry",
    "materials_science",
    "earth_science",
    "climate_science",
    "mathematics",
    "statistics",
    "economics",
    "finance",
    "psychology",
    "sociology",
    "political_science",
    "philosophy",
    "linguistics",
    "education",
    "public_health",
    "engineering",
    "interdisciplinary",
]

QUERIES_PER_STRATUM: int = 5


@dataclass
class GenerationConfig:
    """Runtime configuration for a query generation run."""

    llm_model: str
    api_key: str
    queries_per_stratum: int = QUERIES_PER_STRATUM
    limit: int | None = None
    out_dir: str = field(default="data/datagen/output")
    reasoning_effort: str | None = None
    extra_body: dict[str, Any] | None = None


_QUERY_SCHEMA: str = json.dumps(
    TypeAdapter(list[ResearchQuery]).json_schema(),
    indent=2,
)

_BATCH_PROMPT = """\
You are generating realistic research queries for a scientific literature \
search agent. The optimization pipeline runs the agent live and scores \
returned results against the query, so queries should be diverse and \
realistic.

Domain: {domain}

For EACH of the {n_strata} (intent, specificity) combinations below, generate \
exactly {n} distinct queries. Each query within a stratum must vary in \
phrasing, focus, or angle — not be a minor word swap.

{strata}

Return ONLY JSON matching this schema (no prose, no fences, no markdown):
{schema}

Use "domains" to specify the research domain(s) for the queries.
"""


def _build_strata_section() -> str:
    """Render the (intent, specificity) list for inclusion in the prompt."""
    lines = ["Intents:"]
    for name, desc in INTENTS.items():
        lines.append(f"  - {name}: {desc}")
    lines.append("")
    lines.append("Specificity:")
    for name, desc in SPECIFICITY_LEVELS.items():
        lines.append(f"  - {name}: {desc}")
    return "\n".join(lines)


def build_batch_prompt(domain: str, n: int) -> str:
    """Render the per-domain batch-generation prompt.

    The template, intent/specificity descriptions, and JSON schema are all
    defined in this module; this factory composes them with the runtime
    domain and per-stratum query count.

    Args:
        domain: Primary research domain for the batch.
        n: Number of distinct queries to request per (intent, specificity)
            pair.

    Returns:
        The fully rendered prompt string ready to send to the LLM.
    """
    n_strata = len(INTENTS) * len(SPECIFICITY_LEVELS)
    return _BATCH_PROMPT.format(
        domain=domain,
        n=n,
        n_strata=n_strata,
        strata=_build_strata_section(),
        schema=_QUERY_SCHEMA,
    )
