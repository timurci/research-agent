"""CLI entrypoint for MLflow GenAI evaluation modules.

Usage::

    uv run -m evals.main --list
    uv run -m evals.main --experiment my-exp search-e2e search
    uv run -m evals.main --experiment my-exp --tracking-uri ./mlruns search-e2e
    uv run -m evals.main --experiment my-exp --config config/lm.yaml search
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import mlflow

from evals.search.modules import MODULE_NAMES, build_modules
from research_agent.shared.lm_config import (
    DEFAULT_LM_CONFIG_PATH,
    ROLE_SEARCH_RERANK,
    ROLE_SEARCH_SEARCH,
    UnknownLMConfigRoleError,
    load_lm_configs,
)

__all__ = ["MODULE_NAMES", "main"]

if TYPE_CHECKING:
    from collections.abc import Sequence

    from evals.harness import EvalModule
    from research_agent.shared.agent import LMConfig


def _build_parser() -> argparse.ArgumentParser:
    """Build the evaluation CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Run MLflow GenAI evaluation modules.",
    )
    parser.add_argument(
        "modules",
        nargs="*",
        choices=sorted(MODULE_NAMES),
        help="One or more evaluation modules to run (required unless --list).",
    )
    parser.add_argument(
        "--experiment",
        default=None,
        help="MLflow experiment name (required unless --list).",
    )
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="MLflow tracking URI (default: MLflow file-store default).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_LM_CONFIG_PATH,
        help=(
            "YAML LM config path with search-search and search-rerank "
            f"roles (default: {DEFAULT_LM_CONFIG_PATH})."
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_modules",
        help="Print registered module names and exit.",
    )
    return parser


def _require_role(configs: dict[str, LMConfig], role: str, path: Path) -> LMConfig:
    """Return ``configs[role]`` or raise with a clear message."""
    try:
        return configs[role]
    except KeyError:
        msg = (
            f"unknown LM config role {role!r} in {path}; "
            f"known roles: {sorted(configs)!r}"
        )
        raise UnknownLMConfigRoleError(msg) from None


def _run_module(module: EvalModule) -> None:
    """Load data, build predict_fn/scorers, and call ``mlflow.genai.evaluate``."""
    data = module.load_data()
    predict_fn = module.build_predict_fn()
    scorers = list(module.build_scorers())

    with mlflow.start_run(run_name=module.name):
        mlflow.set_tag("eval.module", module.name)
        result = mlflow.genai.evaluate(
            data=data,
            predict_fn=predict_fn,
            scorers=scorers,
        )

    print(  # noqa: T201  # CLI status output, not logging
        f"[done] {module.name}: passed={result.passed} reason={result.reason!r}",
        file=sys.stderr,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Parse CLI args and run selected evaluation modules."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.list_modules:
        for name in sorted(MODULE_NAMES):
            print(name)  # noqa: T201  # CLI status output, not logging
        return

    if args.experiment is None:
        parser.error("--experiment is required (or pass --list)")
    if not args.modules:
        parser.error("at least one module is required (or pass --list)")

    configs = load_lm_configs(args.config)
    modules = build_modules(
        search_lm_config=_require_role(configs, ROLE_SEARCH_SEARCH, args.config),
        rerank_lm_config=_require_role(configs, ROLE_SEARCH_RERANK, args.config),
    )

    if args.tracking_uri is not None:
        mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    mlflow.dspy.autolog()
    mlflow.litellm.autolog()

    for name in args.modules:
        _run_module(modules[name])


if __name__ == "__main__":
    main()
