"""CLI entrypoint for MLflow GenAI evaluation modules.

Usage::

    uv run -m evals.main --list
    uv run -m evals.main --experiment my-exp search-e2e search
    uv run -m evals.main --experiment my-exp --tracking-uri ./mlruns search-e2e
"""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING

import mlflow

from evals.search.modules import MODULES as SEARCH_MODULES

if TYPE_CHECKING:
    from collections.abc import Sequence

    from evals.harness import EvalModule

MODULES = {**SEARCH_MODULES}


def _build_parser() -> argparse.ArgumentParser:
    """Build the evaluation CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Run MLflow GenAI evaluation modules.",
    )
    parser.add_argument(
        "modules",
        nargs="*",
        choices=sorted(MODULES),
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
        "--list",
        action="store_true",
        dest="list_modules",
        help="Print registered module names and exit.",
    )
    return parser


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
        for name in sorted(MODULES):
            print(name)  # noqa: T201  # CLI status output, not logging
        return

    if args.experiment is None:
        parser.error("--experiment is required (or pass --list)")
    if not args.modules:
        parser.error("at least one module is required (or pass --list)")

    if args.tracking_uri is not None:
        mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    mlflow.dspy.autolog()
    mlflow.litellm.autolog()

    for name in args.modules:
        _run_module(MODULES[name])


if __name__ == "__main__":
    main()
