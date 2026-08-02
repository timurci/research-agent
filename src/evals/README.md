# src/evals

Opik evaluation tooling for research-agent capabilities.

This package is tooling, not part of the runtime application. It mirrors
`src/optimize` structurally: capability slices under `evals/<slice>/`,
held-out datasets, and adapters that **consume domain metrics** without
redefining "good." Optimize trains on the **train** split; evals scores
against the **test** split or a local export.

## Scope

| Evaluated | Dataset | Scorers |
|---|---|---|
| Search agent (`search-search`) | HF `tcakmako/research_queries` **test** split, query-only | Count + relevance (labeled at score time by `search-rerank`) |
| Suggestion generator (`search-suggest`) | Local Opik search-search I/O export | Length (code) + quality (`llm-judge` rubric) |

## Layout

```text
evals/
  feedback.py           # EvaluationScore → Opik ScoreResult
  harness.py            # EvalModule config, seeded subsampling, eval seed
  main.py               # CLI entrypoint: load configs, wire modules, opik.evaluate
  search/               # Search-slice evaluation
    agents.py           # DSPy/LiteLLM-backed agents for eval tasks
    dataset.py          # HF test split + local export → eval rows
    modules.py          # EvalModule registry
    scorers.py          # Opik BaseMetric adapters over domain metrics
```

## Main entrypoint (`main.py`)

`evals.main` is the command-line harness for `opik.evaluate` on a
registered `EvalModule`. At a high level it:

1. Parses CLI arguments (`--experiment`, `--config`, `--instructions`,
   `--limit`, `--seed`, module names, or `--list`).
2. Loads LM configurations from `config/lm.yaml` for students and metric
   labelers/judges.
3. Loads the module's dataset, seeded-subsamples to the module limit, and
   registers it as an Opik dataset under a stable name
   `eval-<module>-s<seed>[-l<limit>]`, reused across runs.
4. Runs `opik.evaluate` with the module's task and scorers and prints the
   experiment name and URL.

Search evaluation runs the search student live against the query-only
test rows and labels relevance at score time with the held-out reranker
LM. Suggestion evaluation uses fixed papers from the local export (no
live search).

## How to run

```bash
make eval-list
make eval ARGS="--experiment my-exp search-search"
make eval ARGS="--experiment my-exp search-suggest"
make eval ARGS="--experiment my-exp --limit 5 --seed 7 search-suggest"
```

or directly:

```bash
uv run -m evals.main --experiment my-exp search-search
```

`make eval` requires `.env` (e.g. `cp .env.example .env`); the direct
`uv run` form does not.

Requires `config/lm.yaml` roles:

| Role | Used by |
|------|---------|
| `search-search` | Search student |
| `search-suggest` | Suggestion student |
| `search-rerank` | Search relevance labeler (scorer) |
| `llm-judge` | Suggestion quality judge (scorer) |

`--instructions` (default `config/instructions.yaml`) optionally maps
module names to saved DSPy programs so optimized instructions are
evaluated against the baseline. Results are logged as an Opik experiment
and viewed in the Opik UI.

## Datasets

### `search-search`

- Path: Hugging Face `tcakmako/research_queries`
- Source split: **`test`** only (optimize uses **`train`**)
- Shape: query-only (`query`)

### `search-suggest`

- Path: local Opik export (gitignored; export a search-search Opik run's
  `dataset.query` + `output.papers` to regenerate) at
  `data/optimize/input/eval-search-search-io.json` (same file for evals
  and optimize until a dedicated train/test split exists)
- Shape: query + papers rows; papers truncated to `SUGGESTION_TOP_N`.

### Shared sampling

- Default caps: 30 rows per module (seeded subsample; seed `1`).
- `--limit N` overrides the cap; the seed is included in the dataset name
  so different seeds get different datasets.

## Notes

- DSPy's on-disk cache is disabled for evals so identical prompts are
  re-evaluated every run; the in-memory cache stays on.
- Search relevance is labeled by a held-out reranker LM at score time;
  suggestion quality is labeled by a held-out `llm-judge` rubric judge.
- See `docs/troubleshooting-evals.md` for known eval issues.

## Relationship to runtime and optimize

- Imports `research_agent` models, metrics, agents, and LM config only.
- Does **not** import `optimize` (sibling tooling; independent adapters).
- Runtime `research_agent` never imports `evals`.
- Same domain metrics as `optimize`; different harness (Opik evaluate vs
  GEPA).
