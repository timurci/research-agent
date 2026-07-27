"""Unit tests for YAML LM config loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from research_agent.shared.config.lm import (
    DEFAULT_LM_CONFIG_PATH,
    ROLE_SEARCH_RERANK,
    ROLE_SEARCH_SEARCH,
    LMConfigFileError,
    UnknownLMConfigRoleError,
    lm_config,
    load_lm_configs,
)


def _write_yaml(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_default_path_constant() -> None:
    assert Path("config/lm.yaml") == DEFAULT_LM_CONFIG_PATH


def test_load_lm_configs_happy_path(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path / "lm.yaml",
        """
search-search:
  model: openai/search-model
  api_key: search-key
  base_url: http://search.example/v1
  provider_config:
    provider:
      order: [OpenAI]
search-rerank:
  model: infinity/rerank-model
  api_key: rerank-key
  base_url: http://rerank.example/v1
""",
    )
    configs = load_lm_configs(path)
    assert set(configs) == {ROLE_SEARCH_SEARCH, ROLE_SEARCH_RERANK}
    search = configs[ROLE_SEARCH_SEARCH]
    assert search.model == "openai/search-model"
    assert search.api_key == "search-key"
    assert str(search.base_url).rstrip("/") == "http://search.example/v1"
    assert search.provider_config == {"provider": {"order": ["OpenAI"]}}
    rerank = configs[ROLE_SEARCH_RERANK]
    assert rerank.model == "infinity/rerank-model"
    assert rerank.provider_config is None


def test_lm_config_selects_role(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path / "lm.yaml",
        """
search-search:
  model: openai/search-model
search-rerank:
  model: infinity/rerank-model
""",
    )
    config = lm_config(ROLE_SEARCH_SEARCH, path=path)
    assert config.model == "openai/search-model"


def test_load_lm_configs_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"
    with pytest.raises(LMConfigFileError, match="not found"):
        load_lm_configs(missing)


def test_load_lm_configs_empty_file(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path / "empty.yaml", "")
    with pytest.raises(LMConfigFileError, match="empty"):
        load_lm_configs(path)


def test_load_lm_configs_not_a_mapping(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path / "list.yaml", "- just\n- a\n- list\n")
    with pytest.raises(LMConfigFileError, match="mapping of roles"):
        load_lm_configs(path)


def test_load_lm_configs_invalid_role_entry(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path / "bad.yaml",
        """
search-search: not-a-mapping
""",
    )
    with pytest.raises(LMConfigFileError, match="must be a mapping"):
        load_lm_configs(path)


def test_load_lm_configs_invalid_lm_fields(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path / "bad-fields.yaml",
        """
search-search:
  base_url: http://example/v1
""",
    )
    with pytest.raises(LMConfigFileError, match="invalid LMConfig"):
        load_lm_configs(path)


def test_load_lm_configs_invalid_yaml(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path / "broken.yaml", "search-search: [\n")
    with pytest.raises(LMConfigFileError, match="invalid YAML"):
        load_lm_configs(path)


def test_lm_config_unknown_role(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path / "lm.yaml",
        """
search-search:
  model: openai/search-model
""",
    )
    with pytest.raises(UnknownLMConfigRoleError, match="unknown LM config role"):
        lm_config(ROLE_SEARCH_RERANK, path=path)
