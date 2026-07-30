# src/optimize

DSPy **GEPA** optimization tooling for research-agent capabilities.

This package is tooling, not part of the runtime application. It mirrors
`src/evals` structurally: capability slices under `optimize/<slice>/`,
Hugging Face query datasets, and adapters that **consume domain metrics**
without redefining "good."

## Scope

| Optimized | Not optimized |
|-----------|---------------|
| Search agent (one GEPA student step) | Reranker |
| | Search→rerank e2e workflow |

GEPA optimizes one program step at a time. Multi-step workflows are not
optimization targets here.

## Layout

```text
optimize/
  feedback.py           # EvaluationScore → GEPA ScoreWithFeedback
  main.py               # CLI entrypoint: load configs, wire modules, run GEPA, save
  search/               # Search-agent optimization slice
    dataset.py          # HF train split → dspy.Example
    metrics.py          # GEPA metric adapters over domain metrics
    agents.py           # SearchProgram student
    program.py          # re-export of SearchProgram
    modules.py          # OptimizeModule registry
```

## Main entrypoint (`main.py`)

`optimize.main` is the command-line harness for running GEPA on a registered
`OptimizeModule`. At a high level it:

1. Parses CLI arguments (`--config`, `--limit`, `--seed`, `--out-dir`,
   `--auto`, module names, or `--list`).
2. Loads LM configurations from `config/lm.yaml` for the student,
   the metric labeler, and the GEPA reflection teacher.
3. Builds the requested modules via `build_modules(...)`, injecting the
   loaded LM configs into each module's factories.
4. For each module:
   - Loads the HF **train** split.
   - Optionally subsamples to `--limit` examples.
   - Splits the pool into train (reflection) and val (Pareto) sets.
   - Builds the student and compiles it with `dspy.GEPA`.
   - Saves the compiled program to `data/optimize/output/<module>.json`.

Live index calls (PubMed, CrossRef, OpenAlex) happen inside the search
student during optimization.

## Dataset

- Path: Hugging Face `tcakmako/research_queries`
- Source split: **`train`** only (evals uses **`test`**; never mixed into GEPA)
- Default pool: sample **50** examples
- Split: pool-size-aware — **50/50** below **200** examples, **80/20**
  at or above. The threshold keeps GEPA's val (Pareto) set meaningful
  when total examples are scarce.
- `--limit N` overrides the pool size before the train/val split

## How to run

```bash
uv run -m optimize.main --list
uv run -m optimize.main --config config/lm.yaml search-search
uv run -m optimize.main --config config/lm.yaml --limit 5 --auto light search-search
```

Requires `config/lm.yaml` roles: `search-search` (student), `search-rerank`
(metric labeler only), and `gepa-reflection` (GEPA reflection). Compiled
programs are written to `data/optimize/output/<module>.json`.

## Relationship to runtime and evals

- Imports `research_agent` models, metrics, agents, and LM config only.
- Does **not** import `evals` (sibling tooling; independent adapters).
- Runtime `research_agent` never imports `optimize`.
- Same domain metrics as `evals`; different harness (GEPA vs Opik evaluate).
- Evals runs `search-search` only; optimize does **not** register an e2e module.

## Notes

- Live optimization will call PubMed, CrossRef, and OpenAlex inside the
  search student — slow and costly; use `--limit` for smoke runs.
- GEPA invokes the metric once when scoring candidates and again when
  building the reflective dataset, so the labeler LM runs roughly twice
  per example per iteration.
- Relevance is labeled by a held-out reranker LM (bootstrap signal), not
  by optimizing the reranker itself.
