# src/datagen

Synthetic search-query generation tooling for DSPy prompt optimization.

This package is tooling, not part of the runtime application. It shares
domain models from `research_agent.search.models` for validation, and its
output is consumed by the optimization pipeline in `src/optimize/`.

## Purpose

Generate varied `ResearchQuery` objects to use as the training set when
optimizing the search prompt. The optimization pipeline
runs the search capability live and scores the returned `list[SearchResult]` with an
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
| `llm_client.py` | Thin async `litellm` wrapper for raw text completions and JSON extraction. | `LLMClient` |
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
                       [--reasoning-effort LEVEL]
                       [--extra-body JSON]
```

Flags:
- `--model` — **required**. LLM model string in LiteLLM format
  (e.g. `"openai/gpt-4o-mini"`).
- `--api-key` — **required**. API key for the LLM provider.
- `--out-dir` — where to write the JSONL file (default: `data/datagen/output`).
- `--reasoning-effort` — reasoning effort for reasoning-capable models
  (e.g. `low`, `medium`, `high`, `minimal`, `none`). Forwarded to
  LiteLLM as `reasoning_effort=...` and ignored by models that do not
  support it.
- `--extra-body` — JSON object merged into the LiteLLM request body.
  Common use case is OpenRouter provider routing; see below.

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

## LLMClient options

`LLMClient` (in `llm_client.py`) is a thin async wrapper around
`litellm.acompletion`. It accepts two optional kwargs beyond `model` and
`api_key`:

| Kwarg | Forwarded as | Purpose |
|---|---|---|
| `reasoning_effort` | `reasoning_effort=...` | Reasoning effort for reasoning-capable models. |
| `extra_body` | `extra_body=...` | Arbitrary extra body fields. Common case: OpenRouter provider routing. |

### Reasoning effort

Pass `--reasoning-effort` (or `reasoning_effort=...` to `LLMClient`)
to control reasoning on supported models. Examples:

- OpenAI o-series / gpt-5: `low`, `medium`, `high` (`minimal` on gpt-5).
- Google Gemini: `low`, `medium`, `high` (use `none` to disable).
- OpenRouter: pass through; depends on the underlying model.

Models that do not expose reasoning ignore the parameter.

### OpenRouter provider routing

Pass `--extra-body` with a JSON object that sets the `provider` key.
OpenRouter uses the `order` list to pick upstream providers and
`allow_fallbacks` (snake-case) to control whether later providers are
tried when an earlier one fails.

With fallbacks (try Anthropic first, fall back to OpenAI):

```bash
--extra-body '{"provider": {"order": ["Anthropic", "OpenAI"], "allow_fallbacks": true}}'
```

Without fallbacks (pin to the first provider, fail if it errors):

```bash
--extra-body '{"provider": {"order": ["openai", "together"], "allow_fallbacks": false}}'
```

## Relationship to the runtime app

- Imports `research_agent.search.models` for type validation.
- The runtime `research_agent` package never imports from `datagen`.
- The downstream consumer is `src/optimize/`, which loads the JSONL file
  into `dspy.Example` objects and runs the search optimization loop.

## Configuration

Per-domain generation follows `docs/data_generation_plan.md`:

- **Domains** (~25): see `DOMAINS` in `config.py`.
- **Intents** (4): literature review, known-item lookup, methodology search, recent advances survey.
- **Specificity** (3): vague, moderate, detailed.
- **Queries per stratum** (default 5): each domain session returns
  4 × 3 × 5 = 60 queries, so the full run produces 25 × 60 = **1500
  queries**. The LLM is asked to vary phrasing, focus, or angle within
  each stratum.
