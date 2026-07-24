"""DSPy GEPA optimization tooling for domain quality metrics.

This package is tooling, not part of the runtime application. It adapts
domain metrics (pure functions over domain value objects) to DSPy/GEPA
metrics that return continuous floats plus textual feedback for
reflection. Metric definitions stay in the domain; adapters only coerce
optimizer I/O.

Capability suites live under subpackages (e.g. ``optimize.search``).
``research_agent`` never imports from this package.
"""
