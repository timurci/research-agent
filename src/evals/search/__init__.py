"""Opik scorers, agents, and dataset loaders for the search capability.

* ``search-search`` — query-only HF test rows; relevance from ``reranker()``.
* ``search-suggest`` — query+papers from local Opik I/O export; length +
  LLM quality judge.
"""
