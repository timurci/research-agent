"""Runtime composition root for paper search.

Layer: Composition root.

Wires the search slice into a long-lived application facade with Opik
tracing and optional user feedback.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

from opik import start_as_current_trace

from research_agent.search.agents import (
    Reranker,
    SearchAgent,
    SuggestionGenerator,
)
from research_agent.search.tools import LiteratureSearch
from research_agent.search.workflows import PaperSearchWorkflow
from research_agent.shared.config.instructions import (
    DEFAULT_INSTRUCTIONS_CONFIG_PATH,
    file_sha256,
    instructions_path,
    load_instructions_config,
)
from research_agent.shared.config.lm import (
    DEFAULT_LM_CONFIG_PATH,
    ROLE_SEARCH_RERANK,
    ROLE_SEARCH_SEARCH,
    ROLE_SEARCH_SUGGEST,
    UnknownLMConfigRoleError,
    load_lm_configs,
)
from research_agent.shared.executor import run_async
from research_agent.shared.observability import (
    configure_dspy_opik_callback,
    flush_opik_client,
    record_user_feedback,
)
from research_agent.shared.session import InMemorySession

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping
    from pathlib import Path

    from research_agent.search.models import PaperInfo, ResearchQuery
    from research_agent.shared.config.models import LMConfig

__all__ = [
    "PaperSearchApp",
    "SearchRun",
    "build_paper_search_app",
    "configure_runtime_observability",
]

type WorkflowRunner = Callable[
    [ResearchQuery],
    Awaitable[tuple[list[PaperInfo], str]],
]

type TraceMetadataFactory = Callable[[], Mapping[str, object]]


@dataclass(frozen=True, slots=True)
class SearchRun:
    """Result of one traced paper-search run.

    ``trace_id`` is the Opik trace handle for optional later feedback.
    """

    papers: list[PaperInfo]
    suggestion: str
    trace_id: str


class PaperSearchApp:
    """Traced paper-search facade with optional user feedback."""

    def __init__(
        self,
        *,
        run_workflow: WorkflowRunner,
        project_name: str | None = None,
        metadata: TraceMetadataFactory | None = None,
    ) -> None:
        """Initialize the facade.

        Args:
            run_workflow: Async callable that runs one search workflow.
            project_name: Optional Opik project for traces and feedback.
            metadata: Optional zero-arg factory for root-trace metadata
                (rebuilt each ``search``, e.g. fresh instruction hashes).
        """
        self._run_workflow = run_workflow
        self._project_name = project_name
        self._metadata = metadata

    async def search(self, query: ResearchQuery) -> SearchRun:
        """Run paper search under an Opik root trace.

        Ends the root trace on context exit, then flushes the Opik client
        off the event loop so ``trace_id`` is ready for immediate
        ``record_feedback``.

        Args:
            query: Research question to search for.

        Returns:
            Reranked papers, suggestion text, and the Opik ``trace_id``.
        """
        domains = list(query.domains) if query.domains is not None else None
        metadata = self._resolve_metadata()
        with start_as_current_trace(
            name="paper_search",
            input={"text": query.text, "domains": domains},
            project_name=self._project_name,
            metadata=metadata,
            flush=False,
        ) as trace:
            papers, suggestion = await self._run_workflow(query)
            trace.update(
                output={
                    "paper_count": len(papers),
                    "suggestion": suggestion,
                },
            )
            run = SearchRun(
                papers=papers,
                suggestion=suggestion,
                trace_id=trace.id,
            )
        await self._flush_opik()
        return run

    async def _flush_opik(self) -> None:
        """Flush the Opik client, never failing the search result.

        Observability lag or an unreachable Opik endpoint must not lose an
        already-computed ``SearchRun``; degrade to a warning instead.
        """
        try:
            await run_async(flush_opik_client)
        except Exception:  # observability health must not fail the search result
            logging.getLogger(__name__).warning(
                "opik flush failed; trace may be delayed",
                exc_info=True,
            )

    async def record_feedback(
        self,
        trace_id: str,
        *,
        useful: bool,
        comment: str | None = None,
    ) -> None:
        """Attach thumbs-up/down feedback to a completed search trace.

        Runs the blocking Opik enqueue+flush path off the event loop.

        Args:
            trace_id: Opik trace id from ``SearchRun.trace_id``.
            useful: Whether the user found the results useful.
            comment: Optional free-text reason.
        """
        await run_async(
            partial(
                record_user_feedback,
                trace_id,
                useful=useful,
                comment=comment,
                project_name=self._project_name,
            ),
        )

    def _resolve_metadata(self) -> dict[str, object] | None:
        """Return root-trace metadata for the current search call."""
        if self._metadata is None:
            return None
        return dict(self._metadata())


def configure_runtime_observability(*, project_name: str | None = None) -> None:
    """Configure process-wide DSPy → Opik span nesting.

    Replaces the DSPy callbacks list for the process. Intended once at
    process start; embedders that already configure DSPy should call this
    only when they own that configuration.

    Args:
        project_name: Optional Opik project for DSPy spans.
    """
    configure_dspy_opik_callback(project_name=project_name)


def build_paper_search_app(  # noqa: PLR0913  # composition-root wiring knobs
    *,
    lm_config_path: Path = DEFAULT_LM_CONFIG_PATH,
    instructions_config_path: Path = DEFAULT_INSTRUCTIONS_CONFIG_PATH,
    project_name: str | None = None,
    configure_observability: bool = True,
    pubmed_api_key: str | None = None,
    openalex_api_key: str | None = None,
) -> PaperSearchApp:
    """Wire LM configs, agents, and a traced paper-search app.

    Builds a long-lived facade. Each ``search`` call constructs a fresh
    ``SearchAgent``, ``SuggestionGenerator``, and ``InMemorySession`` so
    session bags and DSPy program/LM instances stay isolated under
    concurrent async calls. That rebuild also reconstructs the search and
    suggest ``dspy.LM`` graphs and reloads optimized program JSON when
    instructions are configured. Reranker and literature client are shared
    across calls.

    Args:
        lm_config_path: YAML LM config with search-search, search-rerank,
            and search-suggest roles.
        instructions_config_path: YAML map of module name → optimized
            program path.
        project_name: Optional Opik project for traces and feedback.
        configure_observability: When True, register the process-wide
            DSPy Opik callback (replaces any existing DSPy callbacks).
            Pass False when the host process already configures DSPy.
        pubmed_api_key: Optional NCBI API key for elevated PubMed limits.
        openalex_api_key: Optional OpenAlex API key for the OpenAlex
            handler.

    Returns:
        A ready ``PaperSearchApp``.
    """
    if configure_observability:
        configure_runtime_observability(project_name=project_name)

    configs = load_lm_configs(lm_config_path)
    try:
        search_cfg = configs[ROLE_SEARCH_SEARCH]
        rerank_cfg = configs[ROLE_SEARCH_RERANK]
        suggest_cfg = configs[ROLE_SEARCH_SUGGEST]
    except KeyError as exc:
        msg = (
            f"unknown LM config role {exc.args[0]!r} in {lm_config_path}; "
            f"known roles: {sorted(configs)!r}"
        )
        raise UnknownLMConfigRoleError(msg) from None
    instructions = load_instructions_config(instructions_config_path)

    search_instructions = instructions_path(instructions, ROLE_SEARCH_SEARCH)
    suggest_instructions = instructions_path(instructions, ROLE_SEARCH_SUGGEST)

    literature_search = LiteratureSearch(
        pubmed_api_key=pubmed_api_key,
        openalex_api_key=openalex_api_key,
    )
    reranker = Reranker(rerank_cfg)

    def metadata() -> dict[str, object]:
        return _trace_metadata(
            search_cfg=search_cfg,
            rerank_cfg=rerank_cfg,
            suggest_cfg=suggest_cfg,
            search_instructions=search_instructions,
            suggest_instructions=suggest_instructions,
        )

    async def run_workflow(
        query: ResearchQuery,
    ) -> tuple[list[PaperInfo], str]:
        search_agent = SearchAgent(
            search_cfg,
            InMemorySession(),
            literature_search,
            instructions_path=search_instructions,
        )
        suggestion_generator = SuggestionGenerator(
            suggest_cfg,
            instructions_path=suggest_instructions,
        )
        workflow = PaperSearchWorkflow(
            search_agent,
            reranker,
            suggestion_generator,
        )
        return await workflow(query)

    return PaperSearchApp(
        run_workflow=run_workflow,
        project_name=project_name,
        metadata=metadata,
    )


def _trace_metadata(
    *,
    search_cfg: LMConfig,
    rerank_cfg: LMConfig,
    suggest_cfg: LMConfig,
    search_instructions: Path | None,
    suggest_instructions: Path | None,
) -> dict[str, object]:
    """Build root-trace metadata from loaded runtime config."""
    metadata: dict[str, object] = {
        "search.model": search_cfg.model,
        "rerank.model": rerank_cfg.model,
        "suggest.model": suggest_cfg.model,
    }
    if search_instructions is not None:
        metadata["search.instructions.path"] = str(search_instructions)
        metadata["search.instructions.sha256"] = file_sha256(search_instructions)
    if suggest_instructions is not None:
        metadata["suggest.instructions.path"] = str(suggest_instructions)
        metadata["suggest.instructions.sha256"] = file_sha256(suggest_instructions)
    return metadata
