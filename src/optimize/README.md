# src/optimize

DSPy **GEPA** optimization pipeline for research-agent capabilities.

This package is tooling, not part of the runtime application. It mirrors
`src/evals` structurally: capability slices under `optimize/<slice>/`,
Hugging Face query datasets, and adapters that **consume domain metrics**
without redefining “good.”

## Scope

| Optimized | Not optimized |
|-----------|----------------|
| Search agent (one GEPA student step) | Reranker |
| | Search→rerank e2e workflow |

GEPA optimizes one program step at a time. Multi-step workflows are not
optimization targets here.

## Layout

```text
optimize/
  feedback.py           # EvaluationScore → GEPA ScoreWithFeedback
  main.py               # CLI: load trainset, compile SearchProgram with GEPA, save
  search/
    dataset.py          # HF train split → dspy.Example(research_query=…)
    metrics.py          # DSPy/GEPA metric adapters over domain metrics
    agents.py           # SearchProgram student (owned ReAct); relevance labeler
    program.py          # re-export of SearchProgram
    modules.py          # named OptimizeModule registry (`search-search` only)
```

Student: `optimize.search.agents.SearchProgram` — `dspy.Module` with
`self.react` (`SearchAgentSignature` ReAct), sync `forward` (GEPA path),
per-call session isolation, prediction field `search_results`.

## Metric contract (GEPA)

Domain metrics return `EvaluationScore` (`passing`, `reason`, `score` in
`[0, 1]`). Adapters map them to `ScoreWithFeedback`:

- **score** — continuous domain `.score` (GEPA objective)
- **feedback** — text built from pass/fail + `.reason` (GEPA reflection)

Search-agent optimization uses one GEPA metric, `search_query_metric`,
which averages three domain metrics and concatenates their feedback:

- `search_result_count`
- `search_result_relevance` (labels from a held-out **relevance labeler**,
  not from the student)

GEPA callables accept five arguments:
`(gold, pred, trace, pred_name, pred_trace)`.

Predictions must expose agent-level `search_results: list[PaperInfo]`
(session bag after tool use). Signature `status` is diagnostic only and
is not scored here.

## Dataset

- Path: Hugging Face `tcakmako/research_queries`
- Source split: **`train`** only (evals uses **`test`**; never mixed into GEPA)
- Default pool: sample **50** examples, then **80/20** → **40 train** (reflection) / **10 val** (Pareto)
- Shape: `dspy.Example(research_query=ResearchQuery).with_inputs("research_query")`
- `--limit N` overrides the pool size before the train/val split

## How to run

```bash
uv run -m optimize.main --list
uv run -m optimize.main --config config/lm.yaml search-search
uv run -m optimize.main --config config/lm.yaml --limit 5 --auto light search-search
```

Requires `config/lm.yaml` roles: `search-search` (student), `search-rerank`
(metric labeler only), and `optimize-teacher` (GEPA reflection). Compiled
programs are written to `data/optimize/output/<module>.json`.

## Relationship to runtime and evals

- Imports `research_agent` models, metrics, agents, and LM config only.
- Does **not** import `evals` (sibling tooling; independent adapters).
- Runtime `research_agent` never imports `optimize`.
- Same domain metrics as `evals`; different harness (GEPA vs MLflow).
- Evals may run `search-e2e`; optimize does **not** register an e2e module.

## Notes

- Live optimization will call PubMed, CrossRef, and OpenAlex inside the
  search student — slow and costly; use `--limit` for smoke runs.
- GEPA invokes the metric once when scoring candidates and again when
  building the reflective dataset, so the relevance labeler LM runs
  roughly twice per example per iteration (the two runs can disagree;
  DSPy keeps the module-level score).
- Relevance is labeled by a held-out reranker LM (bootstrap signal), not
  by optimizing the reranker itself.
