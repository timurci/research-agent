"""Opik scorers, agents, and dataset loaders for the search capability.

Query-only evalsets: ``task`` is the search agent; relevance labels
come from ``reranker()`` at score time. Rows come from
``load_search_eval_data()`` (HF ``tcakmako/research_queries`` test split).
"""
