"""Research assistant runtime.

Portable capability slices (e.g. ``research_agent.search``) own domain,
application, and infrastructure for each capability. The composition root
(``research_agent.app``) wires the search workflow for runtime use with
Opik tracing and optional user feedback.
"""
