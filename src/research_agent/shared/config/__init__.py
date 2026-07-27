"""Configuration loaders for the research agent.

Layer: Infrastructure.

Reads project-local YAML files and returns plain Python objects or
Pydantic models. Callers in the application layer decide how to inject
the loaded values into agents and workflows.
"""
