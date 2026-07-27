"""DSPy GEPA optimization tooling for domain quality metrics.

This package is tooling, not part of the runtime application. It adapts
domain metrics (pure functions over domain value objects) to DSPy/GEPA
metrics that return continuous floats plus textual feedback for
reflection. Metric definitions stay in the domain; adapters only coerce
optimizer I/O.

The entrypoint is ``optimize.main``. It parses CLI arguments, loads LM
configs, builds the registered modules, and runs each module through the
GEPA compile loop. Module factories live in subpackages such as
``optimize.search``; the top-level package only orchestrates.

``research_agent`` never imports from this package.
"""
