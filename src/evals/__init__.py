"""Opik evaluation tooling for domain quality metrics.

This package is tooling, not part of the runtime application. It adapts
domain metrics (pure functions over domain value objects) to Opik
scoring metrics for use with ``opik.evaluate()``. Metric definitions
stay in the domain; adapters only coerce eval I/O and map scores to
Opik ``ScoreResult``.

``research_agent`` never imports from this package.
"""
