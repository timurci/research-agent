"""Shared MLflow evaluate harness types and sampling.

Layer: Infrastructure (evaluation harness).

Capability-specific adapters and suite registries live next to each
capability package (e.g. ``evals.search.modules``). This module owns the
slice-agnostic pieces: the ``EvalModule`` configuration, deterministic
subsample sampling for dataset size caps, and the project-wide eval
seed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from mlflow.genai.scorers import Scorer

EVAL_SEED: int = 1


@dataclass(frozen=True)
class EvalModule:
    """One named ``mlflow.genai.evaluate`` configuration."""

    name: str
    load_data: Callable[[], list[dict[str, Any]]]
    build_predict_fn: Callable[[], Callable[..., Any]]
    build_scorers: Callable[[], Sequence[Scorer]]
    sample_limit: int | None = None


def sample_rows[RowT](rows: list[RowT], *, limit: int | None, seed: int) -> list[RowT]:
    """Deterministically subsample *rows* to at most *limit* entries.

    Args:
        rows: Loaded dataset rows.
        limit: Maximum number of rows. ``None`` means no cap.
        seed: Sampling seed; the same seed selects the same subsample
            for a given dataset.

    Returns:
        *rows* unchanged when *limit* is ``None`` or not smaller than
        the row count; otherwise *limit* rows selected with a seeded
        RNG, in original dataset order.
    """
    if limit is None or len(rows) <= limit:
        return rows
    indices = sorted(
        random.Random(seed).sample(range(len(rows)), k=limit),  # noqa: S311  # reproducible eval subsampling, not security
    )
    return [rows[index] for index in indices]
