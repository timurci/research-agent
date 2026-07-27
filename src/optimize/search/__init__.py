"""DSPy/GEPA adapters for the search agent (one step).

This subpackage registers the search-agent ``OptimizeModule``. It
provides the HF train split loader, GEPA metric adapters, the
``SearchProgram`` student, and the module factory wiring used by
``optimize.main``.

Optimizes only the search student (not the reranker, not e2e
search→rerank). Relevance labels for metrics come from a held-out
relevance labeler at score time.
"""
