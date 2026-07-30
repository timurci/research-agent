# research-agent

A research assistant optimized to run locally on small language models (SLMs).

The goal is a capable literature assistant you can run on your own hardware — without relying on large cloud models for every step.

## Current focus: enhanced search

We are building and improving an **enhanced search** capability for scientific literature.

Given a research question, the agent:

1. **Generates** better search queries from your question
2. **Searches** multiple scholarly indexes in parallel
3. **Refines** results by iterating — trying new queries, dropping dead ends, and improving what it finds
4. **Suggests** a practical research direction from the top results

That loop is the core idea: not a single-shot keyword lookup, but an agent that keeps searching until it has a strong set of papers, plus a concise suggestion for what to do next.

This is the feature under active development. More research capabilities will follow later.

## Running the search service

From the repository root, after `uv sync` and with config in place (`config/lm.yaml`, and optionally `config/instructions.yaml`):

```bash
uv run uvicorn research_agent.api.app:app --host 0.0.0.0 --port 8000
```

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/search` | Search for papers from a research question |
| `POST` | `/feedback` | Thumbs feedback on a previous search (can be updated by sending again) |

Optional keys for higher literature-API limits: `PUBMED_API_KEY`, `OPENALEX_API_KEY`.

## Quick start

```bash
uv sync
```

For design details, see [docs/architecture.md](docs/architecture.md). Developer tooling for synthetic queries and prompt optimization lives under `src/datagen` and `src/optimize`.
