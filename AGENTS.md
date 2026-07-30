# AGENTS.md

Operating manual for AI coding agents. Exact commands and file-level rules; architectural _why_ is in [`docs/architecture.md`](docs/architecture.md).

## Project overview

A research assistant for scientific literature discovery. The current capability is the **search slice**: `ResearchQuery` → `tuple[list[PaperInfo], str]` via LLM-backed `Agent` ports (search + rerank + suggest), composed by `PaperSearchWorkflow`. Sibling packages `src/datagen`, `src/optimize`, and `src/evals` are tooling only — never imported by the runtime.

## Tech stack

- package management: uv
- quality gates: prek, ruff, ty, pytest
- HTTP API: fastapi, uvicorn
- AI evaluation: opik
- runtime observability: opik
- prompt optimization: dspy
- inference: dspy, litellm
- literature search tools: biopython, habanero

## Environment setup

```bash
uv sync                     # install deps into .venv
uv run <command>            # run inside the project env
```

Manage dependencies through `uv` only; do not use `pip` directly.

## Quality gates

A change is complete when all of the following pass:

```bash
uv run ruff check --fix src/ tests/
uv run ruff format src/ tests/
uv run ty check src/ tests/
uv run pytest
```

`prek` runs ruff and ty on the staging area, plus conventional commits on the message:

```bash
prek run                     # staging area
prek run --all-files         # all tracked files
prek run <hook-id>           # ruff-check, ruff-format, or ty
```

## Project structure

Portable capability slices under `src/research_agent/` (not physical layer folders). Layer roles are declared in each file’s docstring. See [`docs/architecture.md`](docs/architecture.md).

```text
src/research_agent/          runtime package
  app.py                     composition root (PaperSearchApp, tracing, feedback)
  api/                       FastAPI presentation layer (HTTP only)
  shared/                    cross-slice kernel (Agent, Session, observability, …)
  search/                    search slice (models, metrics, tools, agents, workflows)
src/datagen/                 synthetic query generation (NOT runtime)
src/optimize/                DSPy GEPA optimization harness (NOT runtime)
src/evals/                   Opik scorers + eval agent wiring (NOT runtime)
tests/unit/                  deterministic tests (mirror package layout)
tests/external/              live tests (`live` marker; skipped by default)
data/                        generated artifacts (output/ gitignored except .gitkeep)
```

Do not stub unused reserved paths. Datagen entrypoint: `uv run generate-queries ...` → `data/datagen/output/queries_train.jsonl`.

## Architecture rules

- **Domain** — pure; owns meaning, invariants, ports, quality metrics. No DSPy/Opik/LLM clients.
- **Application** — orchestrates; calls ports, never adapters. No business logic.
- **Infrastructure** — implements ports (DSPy, tools, clients). No use-case orchestration.
- No Domain or Application module imports Infrastructure. No DSPy, prompts, model names, or LLM clients in Domain/Application.
- A slice depends only on `research_agent.shared` and external libraries — never on a sibling slice.
- `Agent[InputT, OutputT]` (`shared.agent`) is the LM seam: `async def __call__(data: InputT) -> OutputT`. Type vars are unbounded (outputs may be containers).
- Application methods and ports they call are `async`. Blocking SDK work stays in infrastructure (`shared.executor.run_async`).
- `research_agent` never imports `datagen`, `optimize`, or `evals`.

## Conventions

- **Ubiquitous language.** Names say what the object is; no empty `Request`/`Result` wrappers around existing models.
- **Pydantic invariants at construction.** Domain objects are never patched into validity after create.
- **Workflow output is `tuple[list[PaperInfo], str]`.** Papers are the reranked search results; the string is a free-text suggestion for manual research. No search-index provenance on results; `SearchIndex` is a tool-dispatch key only.
- **Runtime facade is `PaperSearchApp`.** `build_paper_search_app` wires agents and optional literature API keys; `search` returns `SearchRun` (papers, suggestion, Opik `trace_id` after a flushed root trace); `await record_feedback(trace_id, useful=..., comment=...)` attaches optional thumbs feedback (also flushes).
- **HTTP presentation is `research_agent.api`.** FastAPI only; routes call `PaperSearchApp`. Serve from the repo root with `uv run uvicorn research_agent.api.app:app` (or set absolute `RESEARCH_AGENT_LM_CONFIG` / `RESEARCH_AGENT_INSTRUCTIONS_CONFIG`).
- **Metrics live in the domain** as pure functions; `optimize` and `evals` consume them and must not reimplement “good.”
- **No code comments** (inline/trailing). Google docstrings required (`ruff`).
- **Local docstrings only.** Each docstring documents only what is local to the object. The definition is the single source of truth for what something *is* and does internally; the call site documents how it *composes* or *uses* it, and only when extra context is needed there. Do not restate an object's behavior, lifecycle, or invariants at a second site. Example: `SearchAgent` is defined in `src/research_agent/search/agents.py`; how a workflow or eval suite wires it belongs in that workflow/eval file, not duplicated on `SearchAgent` or in its module docstring.
- **State what is true, not what is absent.** A docstring describes what the object is and what it does. Do not document behavior the object does *not* have, or relationships it does *not* have, with other objects. Naming an unrelated concept (e.g., a tool with no session interaction saying "does not affect session") only adds noise. The absence is the default.
- **Let errors propagate.** Catch only to recover, translate, or re-raise with context — not to return `None`/empty defaults.
- **Project-specific exceptions** for rule violations: `class <Name>Error(Exception)`; raise directly.
- **No `cast`.** Prefer better types or control flow.
- **No bare lint suppressions.** `# noqa` / `# type: ignore` / `# ty: ignore` need an explanatory comment.

## Tooling boundaries

- `src/datagen` — synthetic queries → `data/datagen/output/queries_train.jsonl` via `uv run generate-queries`.
- `src/optimize` — DSPy GEPA optimization harness, currently optimizes the **search agent only** (not reranker, not e2e).
- `src/evals` — Opik scoring metrics + eval agent factories; run suites with `make eval ARGS="--experiment NAME search-search"` or `make eval-list`.
- LM endpoints: copy `config/lm.example.yaml` → `config/lm.yaml` (gitignored). Loader is `research_agent.shared.config.lm` (Infrastructure). Roles `search-search`, `search-rerank`, and `search-suggest` are `LMConfig` field maps.
- Optimized instructions: copy `config/instructions.example.yaml` → `config/instructions.yaml` (gitignored). Loader is `research_agent.shared.config.instructions` (Infrastructure). Maps module names (e.g. `search-search`) to saved DSPy program paths.
- Runtime observability: Opik is a main dependency. Configure via `OPIK_URL_OVERRIDE` / `OPIK_API_KEY` / `OPIK_PROJECT_NAME` (same as evals). Helpers live in `research_agent.shared.observability`; composition root is `research_agent.app`.
- Do not commit under `data/**/output/` except `.gitkeep`.

## What NOT to do

- Do not import `datagen`, `optimize`, or `evals` from `research_agent`.
- Do not import DSPy, LiteLLM, Opik, or other LLM libraries into Domain or Application.
- Do not import between slices.
- Do not reimplement domain metric logic in MLflow scorers or GEPA metrics (adapt I/O only; map `EvaluationScore` → Opik `ScoreResult` or GEPA `ScoreWithFeedback`).
- Do not bound `Agent` type variables to `BaseModel`.
- Do not re-validate structure Pydantic already enforces.
- Do not duplicate a definition's behavior, lifecycle, or invariants in another module's docstring or comments. The call site documents only what is local to *its* composition.
- Do not document absent behavior or non-relationships. If an object has no interaction with X, do not write "does not affect X" — say nothing about X.

## Known gotchas

- `uv run pytest` runs unit tests + coverage on `src/research_agent` only; live tests need `uv run pytest -m live`.
- `ruff` uses `select = ["ALL"]`; do not add new backlog violations.
- Optimize data is HF `tcakmako/research_queries` **train** only (sample 50, 80/20 → 40 train / 10 val for GEPA); evals uses **test**. Compiled artifacts go under `data/optimize/output/`.
- Live optimize runs hit PubMed/CrossRef/OpenAlex inside the student and a relevance labeler LM in the metric — use `--limit` for smoke tests.
- CLI defaults in `datagen` must stay aligned with the `data/` layout.
