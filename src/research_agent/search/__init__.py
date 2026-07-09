"""Search slice for the research agent.

Self-contained capability slice grouping the domain and infrastructure
concerns for the search capability. Application orchestration lives in
the runtime shell (``workflows.py``), which calls the ``Agent`` port
directly; the slice itself owns only its domain models and its
infrastructure. The slice is
portable: it depends only on `research_agent.shared` (cross-slice
contracts) and external libraries. It must not import from any other
`research_agent` slice or from the runtime shell.

Layers are virtual, expressed by module role and documented in each
file's docstring (not by subfolders):

- ``models.py`` — Layer: Domain.
- ``tools.py`` — Layer: Infrastructure.
- ``program.py`` — Layer: Infrastructure (reserved; not yet present).

The slice has no domain service: there is no domain logic that doesn't
fit on a single model, so none is warranted. Do not add a thin wrapper
that merely delegates to the ``Agent`` port — that is application
orchestration, and belongs in the runtime shell.
"""
