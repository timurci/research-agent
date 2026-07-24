"""Search optimization module registry for GEPA.

Registers the **search agent** student only. GEPA optimizes one program
step at a time; there is no e2e (search→rerank) module and no reranker
student.

Query-only modules load HF research queries (train split) with no gold
paper lists. Metrics adapt domain quality functions to
``ScoreWithFeedback``. The student is ``SearchProgram``
(``research_agent.search.program``), a persistent ReAct module whose
predictor instructions GEPA optimizes.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from optimize.search.dataset import load_search_trainset
from optimize.search.metrics import search_query_metric
from optimize.search.program import SearchProgram

if TYPE_CHECKING:
    from collections.abc import Callable

    import dspy
    from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback

    from research_agent.shared.agent import LMConfig

MODULE_NAMES: frozenset[str] = frozenset({"search-search"})

SEARCH_SAMPLE_LIMIT: int | None = 30


@dataclass(frozen=True)
class OptimizeModule:
    """One named GEPA optimization configuration (single student step)."""

    name: str
    load_trainset: Callable[[], list[dspy.Example]]
    build_metric: Callable[[], Callable[..., ScoreWithFeedback]]
    build_student: Callable[[], dspy.Module]
    sample_limit: int | None = None


def build_modules(
    *,
    search_lm_config: LMConfig,
    labeler_lm_config: LMConfig | None = None,
) -> dict[str, OptimizeModule]:
    """Build search-agent optimize modules with LM config injection.

    Args:
        search_lm_config: Student LM settings (``search-search`` role)
            for the search student program.
        labeler_lm_config: Held-out relevance labeler LM (``search-rerank``
            role). Used only by metrics, not as a student.
    """

    def _build_metric() -> Callable[..., ScoreWithFeedback]:
        return search_query_metric(lm_config=labeler_lm_config)

    def _build_student() -> dspy.Module:
        return SearchProgram(lm_config=search_lm_config)

    return {
        "search-search": OptimizeModule(
            name="search-search",
            load_trainset=load_search_trainset,
            build_metric=_build_metric,
            build_student=_build_student,
            sample_limit=SEARCH_SAMPLE_LIMIT,
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
