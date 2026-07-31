"""FastAPI application factory.

Layer: Presentation.

Wires HTTP routes to the runtime composition root (``PaperSearchApp``)
and registers the API exception handlers.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI

from research_agent.api.handlers import register_exception_handlers
from research_agent.api.routes import router
from research_agent.app import build_paper_search_app
from research_agent.shared.config.instructions import DEFAULT_INSTRUCTIONS_CONFIG_PATH
from research_agent.shared.config.lm import DEFAULT_LM_CONFIG_PATH
from research_agent.shared.observability import flush_opik_client

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from research_agent.app import PaperSearchApp

__all__ = ["app", "create_app"]


def create_app(*, paper_search_app: PaperSearchApp | None = None) -> FastAPI:
    """Build the FastAPI application.

    Args:
        paper_search_app: Optional pre-built facade for tests. When omitted,
            the app is constructed at startup via ``build_paper_search_app``
            using environment variables and default config paths.

    Returns:
        A configured ``FastAPI`` instance.
    """

    @asynccontextmanager
    async def lifespan(fastapi_app: FastAPI) -> AsyncIterator[None]:
        if paper_search_app is not None:
            fastapi_app.state.paper_search_app = paper_search_app
        else:
            fastapi_app.state.paper_search_app = _build_from_env()
        yield
        try:
            flush_opik_client()
        except Exception:  # shutdown flush must not fail app teardown
            logging.getLogger(__name__).warning(
                "opik flush failed on shutdown; pending traces may be lost",
                exc_info=True,
            )

    application = FastAPI(
        title="research-agent",
        version=_package_version(),
        lifespan=lifespan,
    )
    register_exception_handlers(application)
    application.include_router(router)
    return application


def _package_version() -> str:
    """Return the installed package version, or a local placeholder."""
    try:
        return version("research-agent")
    except PackageNotFoundError:
        return "0.0.0+local"


def _build_from_env() -> PaperSearchApp:
    """Construct ``PaperSearchApp`` from environment and default paths."""
    lm_path = Path(os.environ.get("RESEARCH_AGENT_LM_CONFIG", DEFAULT_LM_CONFIG_PATH))
    instructions_path = Path(
        os.environ.get(
            "RESEARCH_AGENT_INSTRUCTIONS_CONFIG",
            DEFAULT_INSTRUCTIONS_CONFIG_PATH,
        ),
    )
    return build_paper_search_app(
        lm_config_path=lm_path,
        instructions_config_path=instructions_path,
        project_name=os.environ.get("OPIK_PROJECT_NAME"),
        pubmed_api_key=os.environ.get("PUBMED_API_KEY"),
        openalex_api_key=os.environ.get("OPENALEX_API_KEY"),
    )


app = create_app()
