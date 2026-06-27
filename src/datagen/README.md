# src/datagen

Synthetic search-query generation tooling for DSPy prompt optimization.

This package is tooling, not part of the runtime application. It shares
domain models from `research_agent.search.models` for validation, and its
output is consumed by the optimization pipeline in `src/optimize/`.

## Purpose

Generate varied `ResearchQuery` objects to use as the training set when
optimizing the `SearchSuggestionNode` prompt. The optimization pipeline
runs the node live and scores the returned `SearchResults` with an
embedding-similarity metric (defined in `src/optimize/`), so the datagen
package only needs to produce queries — not pre-baked result sets or labels.

## Pipeline

```
domain
   │
   ▼
 QueryGenerator.generate_batch(domain, n)  ──one LLM session──▶  list[ResearchQuery] (4 intents × 3 specificities × n)
   │
   ▼
 deduplicate  (Jaccard > 0.8 on tokens)
   │
   ▼
 write  ──▶  queries_train.jsonl
```

## Files

| File | Responsibility | Key exports |
|---|---|---|
| `main.py` | CLI entry point and pipeline orchestrator. | `run`, `main` |
| `config.py` | Static configuration: domains, intents, specificity levels, per-stratum repeat count, LLM model. | `INTENTS`, `SPECIFICITY_LEVELS`, `DOMAINS`, `QUERIES_PER_STRATUM`, `LLM_MODEL`, `GenerationConfig` |
| `errors.py` | Project-specific exceptions for the pipeline. | `LLMContractError`, `InvalidGeneratedQueryError` |
| `llm_client.py` | Thin DSPy/LiteLLM wrapper for raw text completions and JSON extraction. | `LLMClient` |
| `query_generator.py` | Generates a batch of `ResearchQuery` per domain in a single LLM session. | `QueryGenerator` |
| `validator.py` | Jaccard deduplication. | `deduplicate` |
| `writer.py` | Writes the generated queries to `queries_train.jsonl`. | `write` |

## Data flow

1. `main.py` iterates the configured `DOMAINS` list.
2. For each domain, `QueryGenerator.generate_batch(domain, n)` opens one
   LLM session that covers all 12 (intent, specificity) strata for that
   domain and asks for `n` queries per stratum (4 × 3 × n = 12·n total).
3. Failed batches are skipped (logged to stderr) and the run continues.
4. `validator.deduplicate` drops near-identical queries (Jaccard > 0.8
   over tokens longer than 2 characters) across the full set.
5. `writer.write` serializes each `ResearchQuery` to one JSON line in
   `queries_train.jsonl`.

## How to run

```bash
python -m datagen.main --model MODEL --api-key KEY
                       [--out-dir DIR] [--limit N]
                       [--queries-per-stratum N]
```

Flags:
- `--model` — **required**. LLM model string in LiteLLM format
  (e.g. `"openai/gpt-4o-mini"`).
- `--api-key` — **required**. API key for the LLM provider.
- `--out-dir` — where to write the JSONL file (default: `data/datagen/output`).

Generated artifacts live under `data/`, kept out of `src/` so the source
tree stays clean. The default `data/datagen/output/` directory is created
on first run; commit the `.gitkeep` placeholder but ignore the generated
`queries_train.jsonl` (see `.gitignore`).
- `--limit` — cap the number of generated queries. Cuts mid-batch; useful
  only for smoke runs, not for partial coverage.
- `--queries-per-stratum` — number of queries per (intent, specificity)
  pair per domain session (default: 5 → 1500 queries per full run).

## Output

- `queries_train.jsonl` — one `ResearchQuery` per line:

  ```json
  {"text": "...", "domains": ["machine_learning", "artificial_intelligence"]}
  ```

## Relationship to the runtime app

- Imports `research_agent.search.models` for type validation.
- The runtime `research_agent` package never imports from `datagen`.
- The downstream consumer is `src/optimize/`, which loads the JSONL file
  into `dspy.Example` objects and runs the search-node optimization loop.

## Configuration

Per-domain generation follows `docs/data_generation_plan.md`:

- **Domains** (~25): see `DOMAINS` in `config.py`.
- **Intents** (4): literature review, known-item lookup, methodology search, recent advances survey.
- **Specificity** (3): vague, moderate, detailed.
- **Queries per stratum** (default 5): each domain session returns
  4 × 3 × 5 = 60 queries, so the full run produces 25 × 60 = **1500
  queries**. The LLM is asked to vary phrasing, focus, or angle within
  each stratum.
