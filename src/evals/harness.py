"""Shared MLflow evaluate harness types.

Capability-specific adapters and suite registries live next to each
capability package (e.g. ``evals.search.modules``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from mlflow.genai.scorers import Scorer


@dataclass(frozen=True)
class EvalModule:
    """One named ``mlflow.genai.evaluate`` configuration."""

    name: str
    load_data: Callable[[], list[dict[str, Any]]]
    build_predict_fn: Callable[[], Callable[..., Any]]
    build_scorers: Callable[[], Sequence[Scorer]]
