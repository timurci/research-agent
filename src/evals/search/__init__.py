"""MLflow scorers, agents, and dataset loaders for the search capability.

Query-only evalsets: ``predict_fn`` is search agent or
``paper_search_workflow()``; relevance labels come from ``reranker()`` at
score time. Rows come from ``load_search_eval_data()`` (HF
``tcakmako/research_queries`` test split).
"""
