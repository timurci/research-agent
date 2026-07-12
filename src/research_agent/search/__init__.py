"""Search slice for the research agent.

Self-contained capability slice grouping the domain, application, and
infrastructure concerns for the search capability. The slice owns its
domain models, its LLM-backed adapters, and the application workflows
that compose them. It is portable: it depends only on
``research_agent.shared`` (cross-slice contracts) and external
libraries. It must not import from any other ``research_agent`` slice
or from the runtime shell.

Layers are virtual, expressed by module role and documented in each
module's docstring (not by subfolders):

- ``models.py`` — Layer: Domain.
- ``metrics.py`` — Layer: Domain (relevance, non-hallucination, non-duplicate metrics).
- ``tools.py`` — Layer: Infrastructure.
- ``agents.py`` — Layer: Infrastructure (DSPy search agent, LiteLLM-backed reranker).
- ``workflows.py`` — Layer: Application (search + rerank composition).
- ``program.py`` — Layer: Infrastructure (reserved; not yet present).
"""
