"""CLI entrypoint for MLflow GenAI evaluation modules.

Usage::

    uv run -m evals.main --list
    uv run -m evals.main --experiment my-exp search-e2e search
    uv run -m evals.main --experiment my-exp --tracking-uri ./mlruns search-e2e
    uv run -m evals.main --experiment my-exp --config config/lm.yaml search
    uv run -m evals.main --experiment my-exp --limit 5 --seed 7 search
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import mlflow

from evals.harness import EVAL_SEED, sample_rows
from evals.search.modules import MODULE_NAMES, build_modules
from research_agent.shared.lm_config import (
    DEFAULT_LM_CONFIG_PATH,
    ROLE_SEARCH_RERANK,
    ROLE_SEARCH_SEARCH,
    lm_config,
)

__all__ = ["MODULE_NAMES", "main"]

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from evals.harness import EvalModule


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
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Override per-module sample limits for this run "
            "(e.g. --limit 5 for a smoke run)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=EVAL_SEED,
        help=f"Sampling seed (default: {EVAL_SEED}).",
    )
    return parser


def _run_module(module: EvalModule, *, seed: int) -> None:
    """Load data, subsample to the module limit, and run ``mlflow.genai.evaluate``."""
    data = module.load_data()
    logger.info("[%s] loaded %d rows", module.name, len(data))

    full_size = len(data)
    data = sample_rows(data, limit=module.sample_limit, seed=seed)
    if len(data) < full_size:
        logger.warning(
            "[%s] %d rows exceed sample limit %d; sampled down (seed %d)",
            module.name,
            full_size,
            module.sample_limit,
            seed,
        )

    predict_fn = module.build_predict_fn()
    scorers = list(module.build_scorers())

    params: dict[str, int] = {"eval.rows": len(data), "eval.seed": seed}
    if module.sample_limit is not None:
        params["eval.sample_limit"] = module.sample_limit

    with mlflow.start_run(run_name=module.name):
        mlflow.set_tag("eval.module", module.name)
        mlflow.log_params(params)
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

    modules = build_modules(
        search_lm_config=lm_config(ROLE_SEARCH_SEARCH, path=args.config),
        rerank_lm_config=lm_config(ROLE_SEARCH_RERANK, path=args.config),
    )
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be >= 1")
        modules = {
            name: replace(module, sample_limit=args.limit)
            for name, module in modules.items()
        }

    if args.tracking_uri is not None:
        mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    mlflow.dspy.autolog()
    mlflow.litellm.autolog()

    for name in args.modules:
        _run_module(modules[name], seed=args.seed)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
