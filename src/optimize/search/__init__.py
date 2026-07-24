"""DSPy/GEPA adapters for the search **agent** (one step).

Optimizes only the search student (not the reranker, not e2e
search→rerank). Query-only trainsets use ``research_query`` inputs and
agent-level ``list[PaperInfo]`` predictions. Relevance labels for metrics
come from a held-out ``relevance_labeler()`` at score time. Rows come
from ``load_search_trainset()`` (HF ``tcakmako/research_queries`` train
split).
"""
