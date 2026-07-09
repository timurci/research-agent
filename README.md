# research-agent

A research assistant for scientific literature discovery.

Currently the system focuses on a single capability — turning a research question
into a list of relevant papers — by building and optimizing an AI-powered search
capability. More capabilities will follow in later iterations.

See [docs/architecture.md](docs/architecture.md) for the architecture and design
decisions.

## Current focus

The iteration centers on the **search capability**: it takes a
`ResearchQuery` and returns a list of `SearchResult` by delegating to a
language model through the `Agent` port.

```
ResearchQuery
     │
     ▼
search capability   (LLM via the Agent port)
     │
     ▼
list[SearchResult]
```

A **query generation utility** is already in place and produces the
training set used to optimize the search prompt via DSPy: a varied set of
`ResearchQuery` objects spanning scientific domains, intents, and specificity levels.

The capability is then **optimized** by running it live against the
configured indexes (Semantic Scholar, arXiv, PubMed, CrossRef) and
scoring the returned `list[SearchResult]` against the original query with
an embedding-similarity metric that rewards both high average
relevance and a high relevance floor, so padding the list with
irrelevant abstracts is not a viable strategy.

## Quick start

```bash
uv sync
uv run generate-queries --model openai/gpt-4o-mini --api-key $OPENAI_API_KEY
# → data/datagen/output/queries_train.jsonl
```

See details in the [datagen README](src/datagen/README.md).
