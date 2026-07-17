"""MLflow evaluation tooling for domain quality metrics.

This package is tooling, not part of the runtime application. It adapts
domain metrics (pure functions over domain value objects) to MLflow
code-based scorers for use with ``mlflow.genai.evaluate()``. Metric
definitions stay in the domain; adapters only coerce eval I/O and map
scores to MLflow ``Feedback``.

``research_agent`` never imports from this package.
"""
