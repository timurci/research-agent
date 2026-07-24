"""Load ``LMConfig`` roles from a YAML file.

Layer: Infrastructure.

Default path is ``config/lm.yaml`` (copy from ``config/lm.example.yaml``).
There are no in-code model/url/key defaults and no environment fallbacks.

Application LM roles for the search slice:

* ``search-search`` — search agent
* ``search-rerank`` — reranker / relevance labeler

Tooling roles (not runtime):

* ``optimize-teacher`` — GEPA reflection/teacher for optimize runs
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from research_agent.shared.agent import LMConfig

DEFAULT_LM_CONFIG_PATH = Path("config/lm.yaml")

ROLE_SEARCH_SEARCH = "search-search"
ROLE_SEARCH_RERANK = "search-rerank"
ROLE_OPTIMIZE_TEACHER = "optimize-teacher"


class LMConfigFileError(Exception):
    """Raised when an LM config file is missing or malformed."""


class UnknownLMConfigRoleError(Exception):
    """Raised when a requested LM config role is not in the file."""


def load_lm_configs(path: Path) -> dict[str, LMConfig]:
    """Load a mapping of role name → ``LMConfig`` from YAML.

    Args:
        path: Path to a YAML file whose top-level value is a mapping of
            role names to ``LMConfig`` field dicts.

    Returns:
        Validated role → config mapping.

    Raises:
        LMConfigFileError: If the file is missing, unreadable, not a
            mapping, or contains an invalid role entry.
    """
    if not path.is_file():
        msg = (
            f"LM config file not found: {path}. "
            f"Copy config/lm.example.yaml to config/lm.yaml and edit."
        )
        raise LMConfigFileError(msg)

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        msg = f"failed to read LM config file {path}: {exc}"
        raise LMConfigFileError(msg) from exc
    except yaml.YAMLError as exc:
        msg = f"invalid YAML in LM config file {path}: {exc}"
        raise LMConfigFileError(msg) from exc

    if raw is None:
        msg = f"LM config file is empty: {path}"
        raise LMConfigFileError(msg)
    if not isinstance(raw, dict):
        msg = f"LM config file must be a mapping of roles, got {type(raw).__name__}"
        raise LMConfigFileError(msg)

    configs: dict[str, LMConfig] = {}
    for role, entry in raw.items():
        if not isinstance(role, str):
            msg = f"LM config role keys must be strings, got {role!r}"
            raise LMConfigFileError(msg)
        if not isinstance(entry, dict):
            msg = (
                f"LM config role {role!r} must be a mapping of LMConfig "
                f"fields, got {type(entry).__name__}"
            )
            raise LMConfigFileError(msg)
        try:
            configs[role] = LMConfig.model_validate(entry)
        except ValidationError as exc:
            msg = f"invalid LMConfig for role {role!r} in {path}: {exc}"
            raise LMConfigFileError(msg) from exc
    return configs


def lm_config(
    role: str,
    *,
    path: Path = DEFAULT_LM_CONFIG_PATH,
) -> LMConfig:
    """Return the ``LMConfig`` for ``role`` from the YAML file.

    Args:
        role: Top-level key in the config file (e.g. ``search-search``).
        path: Config file path (default ``config/lm.yaml``).

    Raises:
        LMConfigFileError: If the file cannot be loaded.
        UnknownLMConfigRoleError: If ``role`` is absent from the file.
    """
    configs = load_lm_configs(path)
    try:
        return configs[role]
    except KeyError:
        msg = (
            f"unknown LM config role {role!r} in {path}; "
            f"known roles: {sorted(configs)!r}"
        )
        raise UnknownLMConfigRoleError(msg) from None
