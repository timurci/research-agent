"""Search slice for the research agent.

Self-contained capability slice grouping the domain, application, and
infrastructure concerns for the `SearchSuggestionNode`. The slice is
portable: it depends only on `research_agent.shared` (cross-slice
contracts) and external libraries. It must not import from any other
`research_agent` slice or from the runtime shell.

Layers are virtual, expressed by module role and documented in each
file's docstring (not by subfolders):

- ``models.py`` — Layer: Domain.
- ``tools.py`` — Layer: Infrastructure.
- ``node.py`` — Layer: Domain service (reserved; not yet present).
- ``program.py`` — Layer: Infrastructure (reserved; not yet present).
"""
