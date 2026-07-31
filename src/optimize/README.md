# src/optimize

DSPy **GEPA** optimization tooling for research-agent capabilities.

This package is tooling, not part of the runtime application. It mirrors
`src/evals` structurally: capability slices under `optimize/<slice>/`,
Hugging Face datasets, and adapters that **consume domain metrics**
without redefining "good."

## Scope

| Optimized | Not optimized |
|-----------|---------------|
| Search agent (`search-search`) | Reranker |
| Suggestion generator (`search-suggest`) | Multi-step e2e workflow |

GEPA optimizes one program step at a time. Multi-step workflows are not
optimization targets here.

## Layout

```text
optimize/
  feedback.py           # EvaluationScore → GEPA ScoreWithFeedback
  main.py               # CLI entrypoint: load configs, wire modules, run GEPA, save
  search/               # Search-slice optimization
    dataset.py          # HF train splits → dspy.Example
    metrics.py          # GEPA metric adapters over domain metrics
    agents.py           # SearchProgram / SuggestionProgram students
    program.py          # re-exports of students
    modules.py          # OptimizeModule registry
```

## Main entrypoint (`main.py`)

`optimize.main` is the command-line harness for running GEPA on a registered
`OptimizeModule`. At a high level it:

1. Parses CLI arguments (`--config`, `--limit`, `--seed`, `--out-dir`,
   `--budget`, module names, or `--list`).
2. Loads LM configurations from `config/lm.yaml` for students, metric
   labelers/judges, and the GEPA reflection teacher.
3. Builds the requested modules via `build_modules(...)`, injecting the
   loaded LM configs into each module's factories.
4. For each module:
   - Loads the HF **train** split.
   - Optionally subsamples to `--limit` examples.
   - Splits the pool into train (reflection) and val (Pareto) sets.
   - Builds the student and compiles it with `dspy.GEPA`.
   - Saves the compiled program to `data/optimize/output/<module>.json`.

Live index calls (PubMed, CrossRef, OpenAlex) happen inside the search
student during optimization. The suggestion student uses fixed papers from
the suggestion-inputs dataset (no live search).

## Datasets

### `search-search`

- Path: Hugging Face `tcakmako/research_queries`
- Source split: **`train`** only (evals uses **`test`**)
- Shape: query-only (`research_query`)

### `search-suggest`

- Path: local Opik export (gitignored; export a search-search Opik run's
  `dataset.query` + `output.papers` to regenerate) at
  `data/optimize/input/eval-search-search-io.json` (same file for evals
  and optimize until a dedicated train/test split exists)
- Shape (Opik search-search I/O export rows):

  ```text
  dataset.query   → { text, domains? }
  output.papers   → list[PaperInfo fields]  (or "-" when absent)
  ```

  Extra export columns (tokens, feedback scores, …) are ignored. Rows
  with missing/invalid papers are skipped. Papers are truncated to
  `SUGGESTION_TOP_N` (runtime suggestion top-N).

### Shared sampling

- Default pool: sample **50** examples
- Split: pool-size-aware — **50/50** below **200** examples, **80/20**
  at or above
- `--limit N` overrides the pool size before the train/val split

## How to run

```bash
uv run -m optimize.main --list
uv run -m optimize.main --config config/lm.yaml search-search
uv run -m optimize.main --config config/lm.yaml search-suggest
uv run -m optimize.main --config config/lm.yaml --limit 5 --budget light search-suggest
uv run -m optimize.main --config config/lm.yaml --budget 20 search-search
```

Requires `config/lm.yaml` roles:

| Role | Used by |
|------|---------|
| `search-search` | Search student |
| `search-suggest` | Suggestion student |
| `search-rerank` | Search metric relevance labeler only |
| `llm-judge` | Suggestion metric quality judge (all modules) |
| `gepa-reflection` | GEPA reflection teacher |

Compiled programs are written to `data/optimize/output/<module>.json`.

## Relationship to runtime and evals

- Imports `research_agent` models, metrics, agents, and LM config only.
- Does **not** import `evals` (sibling tooling; independent adapters).
- Runtime `research_agent` never imports `optimize`.
- Same domain metrics as `evals`; different harness (GEPA vs Opik evaluate).
- Evals registers `search-search` and `search-suggest`; optimize does not
  register an e2e module.

## Notes

- Live search optimization calls PubMed, CrossRef, and OpenAlex inside the
  search student — slow and costly; use `--limit` for smoke runs.
- Suggestion optimization does not call literature indexes; it reads the
  local Opik search-search I/O export
  `data/optimize/input/eval-search-search-io.json` (gitignored; export
  from a search-search Opik run).
- GEPA invokes the metric once when scoring candidates and again when
  building the reflective dataset, so labeler/judge LMs run roughly twice
  per example per iteration.
- Search relevance is labeled by a held-out reranker LM; suggestion quality
  is labeled by a held-out `llm-judge` rubric judge.
