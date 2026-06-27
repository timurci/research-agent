# research-agent

A research assistant for scientific literature discovery.

Currently the system focuses on a single capability — turning a research question
into a list of relevant papers — by building and optimizing a AI-powered search
node. More capabilities will follow in later iterations.

## Current focus

The iteration centers on the **search node**: `SearchSuggestionNode`
takes a `ResearchQuery` and returns `SearchResults` by delegating to a
language model.

```
ResearchQuery
     │
     ▼
SearchSuggestionNode
     │
     ▼
SearchResults
```

A **query generation utility** is already in place and produces the
training set to be used to optimize the node's prompt via DSPy: a varied set of
`ResearchQuery` objects spanning scientific domains, intents, and specificity levels.

The node is then **optimized** by running it live against the
configured indexes (Semantic Scholar, arXiv, PubMed, CrossRef) and
scoring the returned `SearchResults` against the original query with
an embedding-similarity metric that rewards both high average
relevance and a high relevance floor, so padding the list with
irrelevant abstracts is not a viable strategy.

## Quick start

```bash
uv sync
uv run generate-queries --model openai/gpt-4o-mini --api-key $OPENAI_API_KEY
# → data/datagen/output/queries_train.jsonl
```

See the details in [README](src/datagen/README.md).
