"""Search optimization module registry for GEPA.

Registers single-step students only:

* ``search-search`` — ``SearchProgram``; count + relevance metric
* ``search-suggest`` — ``SuggestionProgram``; length + quality metric

Default data plan: sample ``SEARCH_SAMPLE_LIMIT`` /
``SUGGEST_SAMPLE_LIMIT`` examples from the matching HF **train** split,
then a train/val split whose fraction depends on the resulting pool size
(``train_fraction_for_pool_size``): 50/50 below
``LOW_POOL_SPLIT_THRESHOLD`` (200), 80/20 at or above. HF **test** splits
are reserved for evals and are never used here.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from optimize.search.agents import SearchProgram, SuggestionProgram
from optimize.search.dataset import load_search_trainset, load_suggest_trainset
from optimize.search.metrics import search_query_metric, search_suggest_metric

if TYPE_CHECKING:
    from collections.abc import Callable

    import dspy
    from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback

    from research_agent.shared.config.models import LMConfig

MODULE_NAMES: frozenset[str] = frozenset({"search-search", "search-suggest"})

SEARCH_SAMPLE_LIMIT: int | None = 50
SUGGEST_SAMPLE_LIMIT: int | None = 50
SEARCH_TRAIN_FRACTION: float = 0.8
LOW_POOL_TRAIN_FRACTION: float = 0.5
LOW_POOL_SPLIT_THRESHOLD: int = 200
_MIN_SPLIT_SIZE: int = 2


@dataclass(frozen=True)
class OptimizeModule:
    """One named GEPA optimization configuration (single student step)."""

    name: str
    load_trainset: Callable[[], list[dspy.Example]]
    metric: Callable[..., ScoreWithFeedback]
    build_student: Callable[[], dspy.Module]
    sample_limit: int | None = None


def build_modules(
    *,
    search_lm_config: LMConfig,
    labeler_lm_config: LMConfig | None = None,
    suggest_lm_config: LMConfig,
    judge_lm_config: LMConfig | None = None,
) -> dict[str, OptimizeModule]:
    """Build search-slice optimize modules with LM config injection.

    Args:
        search_lm_config: Student LM settings (``search-search`` role)
            for the search student program.
        labeler_lm_config: Held-out relevance labeler LM (``search-rerank``
            role). Used only by the search metric, not as a student.
        suggest_lm_config: Student LM settings (``search-suggest`` role)
            for the suggestion student program.
        judge_lm_config: Held-out quality judge LM (``llm-judge`` role)
            for the suggestion metric.
    """
    return {
        "search-search": OptimizeModule(
            name="search-search",
            load_trainset=load_search_trainset,
            metric=search_query_metric(lm_config=labeler_lm_config),
            build_student=lambda: SearchProgram(lm_config=search_lm_config),
            sample_limit=SEARCH_SAMPLE_LIMIT,
        ),
        "search-suggest": OptimizeModule(
            name="search-suggest",
            load_trainset=load_suggest_trainset,
            metric=search_suggest_metric(lm_config=judge_lm_config),
            build_student=lambda: SuggestionProgram(lm_config=suggest_lm_config),
            sample_limit=SUGGEST_SAMPLE_LIMIT,
        ),
    }


def sample_examples(
    examples: list[Any],
    *,
    limit: int | None,
    seed: int,
) -> list[Any]:
    """Deterministically subsample train examples to at most *limit*.

    Args:
        examples: Loaded DSPy examples.
        limit: Maximum number of examples. ``None`` means no cap.
        seed: Sampling seed.

    Returns:
        *examples* unchanged when *limit* is ``None`` or not smaller than
        the length; otherwise *limit* examples in original order.
    """
    if limit is None or len(examples) <= limit:
        return examples
    indices = sorted(
        random.Random(seed).sample(range(len(examples)), k=limit),  # noqa: S311  # reproducible optimize subsampling, not security
    )
    return [examples[index] for index in indices]


def train_fraction_for_pool_size(pool_size: int) -> float:
    """Pick a train/val split fraction based on the sampled pool size.

    Pools smaller than ``LOW_POOL_SPLIT_THRESHOLD`` use a 50/50 split so
    GEPA's val (Pareto) set stays meaningful when total examples are
    scarce. At or above the threshold, falls back to the default
    ``SEARCH_TRAIN_FRACTION`` (80/20).

    Args:
        pool_size: Number of examples after ``sample_examples``.

    Returns:
        ``LOW_POOL_TRAIN_FRACTION`` when *pool_size* is below
        ``LOW_POOL_SPLIT_THRESHOLD``; otherwise ``SEARCH_TRAIN_FRACTION``.
    """
    if pool_size < LOW_POOL_SPLIT_THRESHOLD:
        return LOW_POOL_TRAIN_FRACTION
    return SEARCH_TRAIN_FRACTION


def split_train_val(
    examples: list[Any],
    *,
    train_fraction: float,
    seed: int,
) -> tuple[list[Any], list[Any]]:
    """Split examples into GEPA train (reflection) and val (Pareto).

    Uses an 80/20-style fraction of a sampled pool from the HF train split.
    When fewer than two examples are available, both sides receive the full
    list so GEPA still has non-empty train and val.

    Args:
        examples: Pool already sampled from the HF train split.
        train_fraction: Fraction assigned to train (remainder is val).
            Clamped so each side has at least one example when ``len >= 2``.
        seed: Shuffle seed for the partition.

    Returns:
        ``(train, val)`` lists.
    """
    if len(examples) < _MIN_SPLIT_SIZE:
        pool = list(examples)
        return pool, list(pool)

    n_train = round(len(examples) * train_fraction)
    n_train = min(max(n_train, 1), len(examples) - 1)

    order = list(range(len(examples)))
    random.Random(seed).shuffle(order)  # noqa: S311  # reproducible optimize split, not security
    train = [examples[i] for i in order[:n_train]]
    val = [examples[i] for i in order[n_train:]]
    return train, val
