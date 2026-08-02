# research-agent

A research assistant for scientific literature, optimized to run locally on small language models (SLMs).

The goal is a capable literature assistant you can run on your own hardware — without relying on large cloud models for every step.

## How it works

Given a research question, the agent:

1. **Generates** better search queries from your question
2. **Searches** multiple scholarly indexes in parallel
3. **Refines** results by iterating — trying new queries, dropping dead ends, and improving what it finds
4. **Suggests** a practical research direction from the top results

That loop is the core idea: not a single-shot keyword lookup, but an agent that keeps searching until it has a strong set of papers, plus a concise suggestion for what to do next.

## Quick start

From the repository root:

```bash
uv sync
cp config/lm.example.yaml config/lm.yaml
cp .env.example .env
make serve
```

Edit `config/lm.yaml` with the language-model provider you want to use. Optionally configure Opik in `.env` (copied from `.env.example`) to record traces, and `PUBMED_API_KEY` / `OPENALEX_API_KEY` for better rate limits.

Without optimized instructions (`config/instructions.yaml`), the backend runs with default prompts and performance may not be at a desirable level — see [Optimizing prompts](#optimizing-prompts).

`make serve` runs the FastAPI backend on `0.0.0.0:8000` (override with `HOST=` / `PORT=`). It is equivalent to:

```bash
uv run --env-file .env uvicorn research_agent.api.app:app --host 0.0.0.0 --port 8000
```

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/search` | Search for papers from a research question |
| `POST` | `/feedback` | Thumbs feedback on a previous search (can be updated by sending again) |

## Optimizing prompts

Optimization tunes the prompts on the configured model. It takes `config/lm.yaml` plus a training set (for search: the Hugging Face `tcakmako/research_queries` train split; for suggestions: a local search I/O export), and runs DSPy GEPA instruction optimization over the `search-search` or `search-suggest` steps. The output is compiled program files referenced from `config/instructions.yaml`, which the backend loads at startup.

## Evaluating

Evaluation scores the search and suggestion steps against held-out data — the `tcakmako/research_queries` test split for search, the same local export for suggestions. It needs `config/lm.yaml` (including the `llm-judge` role) and an experiment name; results are logged as an Opik experiment for comparison. Run with:

```bash
make eval ARGS="--experiment my-exp search-search"
```

## Generating training data

Data generation creates the synthetic research queries that optimization trains on. It takes an LLM model string and an API key, and produces deduplicated `ResearchQuery` objects across scientific domains and query types, written to `data/datagen/output/queries_train.jsonl`.

For design details, see [docs/architecture.md](docs/architecture.md).
