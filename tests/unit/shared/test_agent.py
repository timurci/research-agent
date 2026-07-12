"""Unit tests for ``research_agent.shared.agent``."""

from __future__ import annotations

from pydantic import HttpUrl

from research_agent.shared.agent import LMConfig


def test_lm_config_requires_model() -> None:
    config = LMConfig(model="gpt-4o")
    assert config.model == "gpt-4o"
    assert config.api_key is None
    assert config.base_url is None


def test_lm_config_optional_fields() -> None:
    config = LMConfig(
        model="claude-3-5-sonnet",
        api_key="secret",
        base_url=HttpUrl("https://api.example.com"),
    )
    assert config.model == "claude-3-5-sonnet"
    assert config.api_key == "secret"
    assert str(config.base_url) == "https://api.example.com/"


def test_lm_config_is_frozen() -> None:
    config = LMConfig(model="gpt-4o")
    assert hash(config) is not None
