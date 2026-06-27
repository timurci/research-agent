"""Generates synthetic ResearchQuery objects in per-domain LLM sessions.

A session targets one domain and asks the model for a fixed number of
queries per (intent, specificity) stratum, returned as a single JSON
array. The full response is parsed and validated against the
`ResearchQuery` schema by Pydantic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import TypeAdapter, ValidationError

from datagen.config import build_batch_prompt
from datagen.errors import LLMContractError
from research_agent.search.models import ResearchQuery

if TYPE_CHECKING:
    from datagen.llm_client import LLMClient

_QueryList = TypeAdapter(list[ResearchQuery])


class QueryGenerator:
    """Generates batches of ResearchQuery objects using a per-domain LLM session."""

    def __init__(self, client: LLMClient) -> None:
        """Initialise the generator.

        Args:
            client: LLM client used for all batched generations.
        """
        self._client = client

    def generate_batch(self, domain: str, n: int) -> list[ResearchQuery]:
        """Generate one LLM session's worth of queries for a single domain.

        Args:
            domain: Primary research domain for every query in the batch.
            n: Number of distinct queries to request per (intent, specificity)
                pair.

        Returns:
            A list of validated `ResearchQuery` objects, in the order the
            model returned them.

        Raises:
            LLMContractError: If the LLM response cannot be parsed as a
                JSON array, or if it does not match the `ResearchQuery`
                schema.
        """
        prompt = build_batch_prompt(domain, n)
        text = self._client.complete(prompt)
        try:
            return _QueryList.validate_json(text)
        except ValidationError as exc:
            msg = f"LLM response did not match the expected query schema: {exc}"
            raise LLMContractError(msg) from exc
