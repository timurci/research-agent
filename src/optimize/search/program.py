"""GEPA student programs for search-slice optimization.

Layer: Infrastructure (optimization-only, not imported by runtime).

Students live in ``optimize.search.agents``. This module re-exports them
for a stable import path used by the module registry and docs.
"""

from __future__ import annotations

from optimize.search.agents import SearchProgram, SuggestionProgram

__all__ = ["SearchProgram", "SuggestionProgram"]
