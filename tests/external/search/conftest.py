"""Shared fixtures for live tests in ``tests/external/search``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from research_agent.shared.lm_config import (
    ROLE_SEARCH_RERANK,
    ROLE_SEARCH_SEARCH,
    lm_config,
)

if TYPE_CHECKING:
    from research_agent.shared.agent import LMConfig


@pytest.fixture
def search_lm_config() -> LMConfig:
    """LMConfig for the live search model (``search-search`` role)."""
    return lm_config(ROLE_SEARCH_SEARCH)


@pytest.fixture
def reranker_lm_config() -> LMConfig:
    """LMConfig for the live rerank model (``search-rerank`` role)."""
    return lm_config(ROLE_SEARCH_RERANK)
