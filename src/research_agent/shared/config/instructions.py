"""Load optimized program paths from a YAML file.

Layer: Infrastructure.

Default path is ``config/instructions.yaml`` (copy from
``config/instructions.example.yaml``). Missing files or missing entries are
not errors: callers fall back to the default prompt generation when no
optimized program is configured for a module.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)

DEFAULT_INSTRUCTIONS_CONFIG_PATH = Path("config/instructions.yaml")

InstructionsConfig = dict[str, Path]


class InstructionsConfigError(Exception):
    """Raised when an instructions config file is malformed or missing a path."""


def file_sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of *path* contents.

    Args:
        path: File to hash.

    Returns:
        Lowercase hex digest of the file bytes.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_yaml(path: Path) -> object:
    """Read and parse YAML from *path*.

    Raises:
        InstructionsConfigError: If the file cannot be read or parsed.
    """
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        msg = f"failed to read instructions config file {path}: {exc}"
        raise InstructionsConfigError(msg) from exc
    except yaml.YAMLError as exc:
        msg = f"invalid YAML in instructions config file {path}: {exc}"
        raise InstructionsConfigError(msg) from exc


def _parse_instructions(raw: object) -> dict[str, object]:
    """Validate raw YAML and return the ``instructions`` mapping.

    Returns an empty mapping when *raw* is ``None`` or lacks the
    ``instructions`` key.

    Raises:
        InstructionsConfigError: If the top-level shape is invalid.
    """
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        msg = f"instructions config file must be a mapping, got {type(raw).__name__}"
        raise InstructionsConfigError(msg)

    instructions = raw.get("instructions")
    if instructions is None:
        return {}
    if not isinstance(instructions, dict):
        msg = (
            f"instructions config 'instructions' must be a mapping, "
            f"got {type(instructions).__name__}"
        )
        raise InstructionsConfigError(msg)
    typed: dict[str, object] = {}
    for key, value in instructions.items():
        if not isinstance(key, str):
            msg = f"instructions config keys must be strings, got {key!r}"
            raise InstructionsConfigError(msg)
        typed[key] = value
    return typed


def _validate_program_path(name: str, value: object) -> Path:
    """Return *value* as a Path after validating it points to a file.

    Raises:
        InstructionsConfigError: If *value* is not a valid program path.
    """
    if not isinstance(value, str):
        msg = (
            f"optimized program path for {name!r} must be a string, "
            f"got {type(value).__name__}"
        )
        raise InstructionsConfigError(msg)
    program_path = Path(value)
    if not program_path.is_file():
        msg = f"optimized program path for {name!r} does not exist: {program_path}"
        raise InstructionsConfigError(msg)
    return program_path


def load_instructions_config(
    path: Path = DEFAULT_INSTRUCTIONS_CONFIG_PATH,
) -> InstructionsConfig:
    """Load module name → optimized program path mappings from YAML.

    The file is expected to contain a top-level ``instructions`` mapping
    whose values are paths to saved DSPy programs. If the file is missing
    or the ``instructions`` key is absent, an empty mapping is returned so
    callers can fall back to default prompts.

    Args:
        path: Path to the instructions YAML file.

    Returns:
        Mapping from module name to optimized program path.

    Raises:
        InstructionsConfigError: If the file exists but cannot be parsed
            or a configured program path does not exist.
    """
    if not path.is_file():
        logger.info("instructions config not found: %s; using default prompts", path)
        return {}

    instructions = _parse_instructions(_read_yaml(path))
    return {
        name: _validate_program_path(name, value)
        for name, value in instructions.items()
    }


def instructions_path(
    config: Mapping[str, Path],
    name: str,
) -> Path | None:
    """Return the optimized program path for *name* or ``None``.

    Args:
        config: Loaded instructions config.
        name: Module name to look up.

    Returns:
        The configured path, or ``None`` if the module is not present.
    """
    return config.get(name)
