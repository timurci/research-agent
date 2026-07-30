"""CLI entrypoint for Opik evaluation modules.

Usage::

    uv run -m evals.main --list
    uv run -m evals.main --experiment my-exp search-search
    uv run -m evals.main --experiment my-exp search-suggest
    uv run -m evals.main --experiment my-exp --config config/lm.yaml search-search
    uv run -m evals.main --experiment my-exp --limit 5 --seed 7 search-suggest
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import dspy
import opik
from opik.integrations.dspy import OpikCallback

from evals.harness import EVAL_SEED, sample_rows
from evals.search.modules import MODULE_NAMES, build_modules
from research_agent.shared.config.instructions import (
    DEFAULT_INSTRUCTIONS_CONFIG_PATH,
    InstructionsConfig,
    file_sha256,
    load_instructions_config,
)
from research_agent.shared.config.lm import (
    DEFAULT_LM_CONFIG_PATH,
    ROLE_LLM_JUDGE,
    ROLE_SEARCH_RERANK,
    ROLE_SEARCH_SEARCH,
    ROLE_SEARCH_SUGGEST,
    lm_config,
)

__all__ = ["MODULE_NAMES", "main"]

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from evals.harness import EvalModule
    from research_agent.shared.config.models import LMConfig


def _build_parser() -> argparse.ArgumentParser:
    """Build the evaluation CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Run Opik evaluation modules.",
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
        help="Opik experiment name (required unless --list).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_LM_CONFIG_PATH,
        help=(
            "YAML LM config path with search-search, search-rerank, "
            "search-suggest, and llm-judge roles "
            f"(default: {DEFAULT_LM_CONFIG_PATH})."
        ),
    )
    parser.add_argument(
        "--instructions",
        type=Path,
        default=DEFAULT_INSTRUCTIONS_CONFIG_PATH,
        help=(
            "YAML instructions config path mapping module names to saved "
            f"DSPy programs (default: {DEFAULT_INSTRUCTIONS_CONFIG_PATH})."
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


def _dataset_name(module: EvalModule, *, seed: int) -> str:
    """Build a stable dataset name from module config.

    Seed is included so different seeds get different datasets. A
    non-None sample limit is included as a suffix so capped and
    uncapped runs use distinct datasets.
    """
    name = f"eval-{module.name}-s{seed}"
    if module.sample_limit is not None:
        name += f"-l{module.sample_limit}"
    return name


def _run_module(  # noqa: PLR0913  # orchestration function with distinct config args
    module: EvalModule,
    *,
    experiment_name: str,
    search_lm_config: LMConfig,
    rerank_lm_config: LMConfig,
    suggest_lm_config: LMConfig,
    judge_lm_config: LMConfig,
    instructions: InstructionsConfig,
    seed: int,
) -> None:
    """Load data, subsample to the module limit, and run ``opik.evaluate``."""
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

    task = module.build_task()
    scorers = list(module.build_scorers())

    experiment_config: dict[str, object] = {
        "eval.rows": len(data),
        "eval.seed": seed,
        "search.model": search_lm_config.model,
        "reranker.model": rerank_lm_config.model,
        "suggest.model": suggest_lm_config.model,
        "judge.model": judge_lm_config.model,
    }
    if module.sample_limit is not None:
        experiment_config["eval.sample_limit"] = module.sample_limit
    for module_key, config_prefix in (
        ("search-search", "search"),
        ("search-suggest", "suggest"),
    ):
        if module_key in instructions:
            instruction_path = instructions[module_key]
            experiment_config[f"{config_prefix}.instructions.path"] = str(
                instruction_path,
            )
            experiment_config[f"{config_prefix}.instructions.sha256"] = file_sha256(
                instruction_path,
            )

    client = opik.Opik()
    dataset_name = _dataset_name(module, seed=seed)
    dataset = client.get_or_create_dataset(name=dataset_name)
    if not dataset.dataset_items_count:
        dataset.insert(data)
        logger.info(
            "[%s] populated dataset %r with %d rows",
            module.name,
            dataset_name,
            len(data),
        )
    else:
        logger.info(
            "[%s] reusing existing dataset %r (%d items)",
            module.name,
            dataset_name,
            dataset.dataset_items_count,
        )

    result = opik.evaluate(
        dataset=dataset,
        task=task,
        scoring_metrics=scorers,
        experiment_name=experiment_name,
        experiment_config=experiment_config,
        verbose=0,
    )

    print(  # noqa: T201  # CLI status output, not logging
        f"[done] {module.name}: experiment={result.experiment_name} "
        f"url={result.experiment_url}",
        file=sys.stderr,
    )


def _disable_dspy_disk_cache() -> None:
    """Disable DSPy on-disk cache so each evals run re-evaluates responses.

    DSPy enables a process-global disk cache at import time
    (``dspy.cache`` with ``enable_disk_cache=True``). For evals this
    hides metric changes across runs because identical prompts return
    cached responses from a previous invocation. In-memory cache is
    left on so repeated prompts within the same run are still cheap.
    """
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=True)
    logger.info("Disabled DSPy on-disk cache for evals; in-memory cache retained")


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

    instructions = load_instructions_config(args.instructions)
    search_lm_config = lm_config(ROLE_SEARCH_SEARCH, path=args.config)
    rerank_lm_config = lm_config(ROLE_SEARCH_RERANK, path=args.config)
    suggest_lm_config = lm_config(ROLE_SEARCH_SUGGEST, path=args.config)
    judge_lm_config = lm_config(ROLE_LLM_JUDGE, path=args.config)
    modules = build_modules(
        search_lm_config=search_lm_config,
        rerank_lm_config=rerank_lm_config,
        suggest_lm_config=suggest_lm_config,
        judge_lm_config=judge_lm_config,
        instructions=instructions,
    )
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be >= 1")
        modules = {
            name: replace(module, sample_limit=args.limit)
            for name, module in modules.items()
        }

    dspy.configure(callbacks=[OpikCallback()])
    _disable_dspy_disk_cache()

    for name in args.modules:
        _run_module(
            modules[name],
            experiment_name=args.experiment,
            search_lm_config=search_lm_config,
            rerank_lm_config=rerank_lm_config,
            suggest_lm_config=suggest_lm_config,
            judge_lm_config=judge_lm_config,
            instructions=instructions,
            seed=args.seed,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
