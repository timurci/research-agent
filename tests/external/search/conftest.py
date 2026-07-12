"""Shared fixtures for live tests in ``tests/external/search``."""

from __future__ import annotations

import pytest
from pydantic import HttpUrl

from research_agent.shared.agent import LMConfig

SEARCH_MODEL: str = "openai/LFM2.5-8B-A1B"
SEARCH_API_KEY: str = "ignored-auth-key"
SEARCH_BASE_URL: str = "http://localhost:8080/v1"

RERANK_MODEL: str = "infinity/LFM2.5-ColBERT-350M"
RERANK_API_KEY: str = "ignored-auth-key"
RERANK_BASE_URL: str = "http://localhost:8080/v1"


@pytest.fixture
def search_lm_config() -> LMConfig:
    """LMConfig pointing at the local search model."""
    return LMConfig(
        model=SEARCH_MODEL,
        api_key=SEARCH_API_KEY,
        base_url=HttpUrl(SEARCH_BASE_URL),
    )


@pytest.fixture
def reranker_lm_config() -> LMConfig:
    """LMConfig pointing at the local rerank model."""
    return LMConfig(
        model=RERANK_MODEL,
        api_key=RERANK_API_KEY,
        base_url=HttpUrl(RERANK_BASE_URL),
    )
