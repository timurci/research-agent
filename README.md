# research-agent

A research assistant optimized to run locally on small language models (SLMs).

The goal is a capable literature assistant you can run on your own hardware — without relying on large cloud models for every step.

## Current focus: enhanced search

We are building and improving an **enhanced search** capability for scientific literature.

Given a research question, the agent:

1. **Generates** better search queries from your question
2. **Searches** multiple scholarly indexes in parallel
3. **Refines** results by iterating — trying new queries, dropping dead ends, and improving what it finds through a ReAct-style loop

That loop is the core idea: not a single-shot keyword lookup, but an agent that keeps searching until it has a strong set of papers.

This is the feature under active development. More research capabilities will follow later.

## Architecture

See [docs/architecture.md](docs/architecture.md) for design decisions and how the system is structured.

## Quick start

```bash
uv sync
uv run generate-queries --model openai/gpt-4o-mini --api-key $OPENAI_API_KEY
# → data/datagen/output/queries_train.jsonl
```

See details in the [datagen README](src/datagen/README.md).
