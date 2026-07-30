"""CLI entrypoint for DSPy GEPA optimization modules.

Usage::

    uv run -m optimize.main --list
    uv run -m optimize.main --config config/lm.yaml search-search
    uv run -m optimize.main --limit 5 --auto medium search-search

Optimizes one student step at a time (currently: search agent only).
Does not optimize the reranker or the search→rerank e2e workflow.

This pipeline loads the Hugging Face **train** split (evals keeps
**test**), samples a pool, splits it into GEPA train (reflection) and
val (Pareto selection) using a pool-size-aware fraction, builds GEPA
metrics that adapt domain quality functions, and compiles the student
program (``optimize.search.agents.SearchProgram``) with ``dspy.GEPA``.
Live index calls (PubMed, CrossRef, OpenAlex) happen inside the search
student during optimization. The ``search-rerank`` role is used only
for relevance labeling in metrics; ``gepa-reflection`` is the GEPA
reflection LM.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import dspy

from optimize.search.modules import (
    MODULE_NAMES,
    build_modules,
    sample_examples,
    split_train_val,
    train_fraction_for_pool_size,
)
from research_agent.shared.config.lm import (
    DEFAULT_LM_CONFIG_PATH,
    ROLE_GEPA_REFLECTION,
    ROLE_SEARCH_RERANK,
    ROLE_SEARCH_SEARCH,
    lm_config,
)
from research_agent.shared.dspy import dspy_lm

__all__ = ["MODULE_NAMES", "main"]

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from optimize.search.modules import OptimizeModule

type GepaAuto = Literal["light", "medium", "heavy"]

DEFAULT_SEED: int = 0
DEFAULT_OUT_DIR: Path = Path("data/optimize/output")
DEFAULT_GEPA_AUTO: GepaAuto = "light"
GEPA_AUTO_PRESETS: tuple[GepaAuto, ...] = ("light", "medium", "heavy")


@dataclass(frozen=True)
class _RunOptions:
    """CLI-derived settings for one optimize module run."""

    seed: int
    limit: int | None
    out_dir: Path
    teacher_lm: dspy.LM
    auto: GepaAuto


def _build_parser() -> argparse.ArgumentParser:
    """Build the optimization CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Run DSPy GEPA optimization modules.",
    )
    parser.add_argument(
        "modules",
        nargs="*",
        choices=sorted(MODULE_NAMES),
        help="One or more optimization modules to run (required unless --list).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_LM_CONFIG_PATH,
        help=(
            "YAML LM config path: search-search (student), search-rerank "
            "(metric labeler only), and gepa-reflection (GEPA reflection; "
            f"default: {DEFAULT_LM_CONFIG_PATH})."
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
        help=("Override per-module pool size before the train/val split."),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Sampling seed (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Directory for compiled programs (default: {DEFAULT_OUT_DIR}).",
    )
    parser.add_argument(
        "--auto",
        choices=sorted(GEPA_AUTO_PRESETS),
        default=DEFAULT_GEPA_AUTO,
        help=f"GEPA budget preset (default: {DEFAULT_GEPA_AUTO}).",
    )
    return parser


def _run_module(module: OptimizeModule, options: _RunOptions) -> None:
    """Load data, split train/val, build student, and compile with GEPA."""
    loaded = module.load_trainset()
    logger.info("[%s] loaded %d examples", module.name, len(loaded))

    sample_limit = options.limit if options.limit is not None else module.sample_limit
    full_size = len(loaded)
    pool = sample_examples(loaded, limit=sample_limit, seed=options.seed)
    if len(pool) < full_size:
        logger.warning(
            "[%s] %d examples exceed sample limit %d; sampled down (seed %d)",
            module.name,
            full_size,
            sample_limit,
            options.seed,
        )

    train_fraction = train_fraction_for_pool_size(len(pool))
    trainset, valset = split_train_val(
        pool,
        train_fraction=train_fraction,
        seed=options.seed,
    )
    logger.info(
        "[%s] split pool=%d → train=%d val=%d (fraction=%.2f, seed=%d)",
        module.name,
        len(pool),
        len(trainset),
        len(valset),
        train_fraction,
        options.seed,
    )

    student = module.build_student()
    optimizer = dspy.GEPA(
        metric=module.metric,
        auto=options.auto,
        add_format_failure_as_feedback=True,
        reflection_lm=options.teacher_lm,
        log_dir=str(options.out_dir / module.name),
        track_stats=True,
        seed=options.seed,
    )
    logger.info("[%s] compiling with GEPA (auto=%s)", module.name, options.auto)
    optimized = optimizer.compile(student, trainset=trainset, valset=valset)

    options.out_dir.mkdir(parents=True, exist_ok=True)
    program_path = options.out_dir / f"{module.name}.json"
    optimized.save(str(program_path))
    logger.info("[%s] saved optimized program to %s", module.name, program_path)

    detailed = getattr(optimized, "detailed_results", None)
    if detailed is not None:
        best = detailed.highest_score_achieved_per_val_task
        if best:
            logger.info(
                "[%s] best average val score: %.3f",
                module.name,
                sum(best) / len(best),
            )


def main(argv: Sequence[str] | None = None) -> None:
    """Parse CLI arguments and run selected optimization modules."""
    logging.basicConfig(level=logging.INFO)
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.list_modules:
        for name in sorted(MODULE_NAMES):
            print(name)  # noqa: T201  # CLI list output
        return

    if not args.modules:
        parser.error("the following arguments are required: modules")

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")

    search_cfg = lm_config(ROLE_SEARCH_SEARCH, path=args.config)
    labeler_cfg = lm_config(ROLE_SEARCH_RERANK, path=args.config)
    teacher_lm = dspy_lm(lm_config(ROLE_GEPA_REFLECTION, path=args.config))
    modules = build_modules(
        search_lm_config=search_cfg,
        labeler_lm_config=labeler_cfg,
    )
    run_options = _RunOptions(
        seed=args.seed,
        limit=args.limit,
        out_dir=args.out_dir,
        teacher_lm=teacher_lm,
        auto=args.auto,
    )

    for name in args.modules:
        _run_module(modules[name], run_options)


if __name__ == "__main__":
    main()
