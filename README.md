# research-agent

A research assistant optimized to run locally on small language models (SLMs).

The goal is a capable literature assistant you can run on your own hardware — without relying on large cloud models for every step.

## Current focus: enhanced search

We are building and improving an **enhanced search** capability for scientific literature.

Given a research question, the agent:

1. **Generates** better search queries from your question
2. **Searches** multiple scholarly indexes in parallel
3. **Refines** results by iterating — trying new queries, dropping dead ends, and improving what it finds through a ReAct-style loop
4. **Suggests** a practical research direction from the top results

That loop is the core idea: not a single-shot keyword lookup, but an agent that keeps searching until it has a strong set of papers, plus a concise suggestion for what to do next.

This is the feature under active development. More research capabilities will follow later.

## Optimizing search

The search agent can also be tuned automatically. The `optimize` package uses DSPy GEPA to improve the agent's instructions against a Hugging Face query dataset.

```bash
uv run -m optimize.main --config config/lm.yaml search-search
```

This is developer tooling, not part of the runtime agent. See the [optimize README](src/optimize/README.md) for details.

## Generating training queries

Optimization needs example research questions. The `datagen` package synthesizes training queries from a language model.

```bash
uv run generate-queries --model openai/gpt-4o-mini --api-key $OPENAI_API_KEY
# → data/datagen/output/queries_train.jsonl
```

See the [datagen README](src/datagen/README.md) for details.

## Architecture

See [docs/architecture.md](docs/architecture.md) for design decisions and how the system is structured.

## Quick start

```bash
uv sync
```

Then see [Generating training queries](#generating-training-queries) to produce data and [Optimizing search](#optimizing-search) to tune the agent.
