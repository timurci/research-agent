"""GEPA student program for search-agent optimization.

Layer: Infrastructure (optimization-only, not imported by runtime).

``SearchProgram`` lives in ``optimize.search.agents`` (owned ReAct on
``self.react``, sync ``forward``). This module re-exports it for a
stable import path used by the module registry and docs.
"""

from __future__ import annotations

from optimize.search.agents import SearchProgram

__all__ = ["SearchProgram"]
