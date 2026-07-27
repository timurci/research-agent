"""Unit tests for optimized instructions config loading."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from research_agent.shared.config.instructions import (
    DEFAULT_INSTRUCTIONS_CONFIG_PATH,
    InstructionsConfigError,
    file_sha256,
    instructions_path,
    load_instructions_config,
)


def _write_yaml(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_default_path_constant() -> None:
    assert Path("config/instructions.yaml") == DEFAULT_INSTRUCTIONS_CONFIG_PATH


def test_load_instructions_config_missing_file_returns_empty(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    missing = tmp_path / "missing.yaml"

    with caplog.at_level("INFO"):
        config = load_instructions_config(missing)

    assert config == {}
    assert "not found" in caplog.text


def test_load_instructions_config_missing_instructions_key(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path / "instructions.yaml", "other:\n  value: 1\n")

    assert load_instructions_config(path) == {}


def test_load_instructions_config_happy_path(tmp_path: Path) -> None:
    program = tmp_path / "search-search.json"
    program.write_text("{}", encoding="utf-8")
    path = _write_yaml(
        tmp_path / "instructions.yaml",
        f"""
instructions:
  search-search: {program}
""",
    )

    config = load_instructions_config(path)
    assert config == {"search-search": program}


def test_load_instructions_config_path_does_not_exist(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path / "instructions.yaml",
        """
instructions:
  search-search: /does/not/exist.json
""",
    )

    with pytest.raises(InstructionsConfigError, match="does not exist"):
        load_instructions_config(path)


def test_load_instructions_config_non_string_value(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path / "instructions.yaml",
        """
instructions:
  search-search: 42
""",
    )

    with pytest.raises(InstructionsConfigError, match="must be a string"):
        load_instructions_config(path)


def test_load_instructions_config_invalid_yaml(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path / "instructions.yaml", "instructions: [\n")

    with pytest.raises(InstructionsConfigError, match="invalid YAML"):
        load_instructions_config(path)


def test_load_instructions_config_top_level_not_mapping(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path / "instructions.yaml", "- just\n- a\n- list\n")

    with pytest.raises(InstructionsConfigError, match="must be a mapping"):
        load_instructions_config(path)


def test_load_instructions_config_instructions_not_mapping(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path / "instructions.yaml", "instructions: 42\n")

    with pytest.raises(InstructionsConfigError, match="must be a mapping"):
        load_instructions_config(path)


def test_load_instructions_config_invalid_key_type(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path / "instructions.yaml",
        """
instructions:
  1: /path/to/program.json
""",
    )

    with pytest.raises(InstructionsConfigError, match="must be strings"):
        load_instructions_config(path)


def test_instructions_path_returns_path_or_none(tmp_path: Path) -> None:
    program = tmp_path / "program.json"
    config = {"search-search": program}

    assert instructions_path(config, "search-search") == program
    assert instructions_path(config, "missing") is None
    assert instructions_path({}, "search-search") is None


def test_file_sha256_returns_hex_digest(tmp_path: Path) -> None:
    path = tmp_path / "program.json"
    content = b"optimized instructions content"
    path.write_bytes(content)

    assert file_sha256(path) == hashlib.sha256(content).hexdigest()
