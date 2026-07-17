# AGENTS.md

Operating manual for AI coding agents. Exact commands and file-level rules; architectural _why_ is in [`docs/architecture.md`](docs/architecture.md).

## Project overview

A research assistant for scientific literature discovery. The current capability is the **search slice**: `ResearchQuery` → `list[PaperInfo]` via LLM-backed `Agent` ports (search + rerank), composed by `PaperSearchWorkflow`. Sibling packages `src/datagen`, `src/optimize`, and `src/evals` are tooling only — never imported by the runtime.

## Tech stack

- Python ≥ 3.14
- `pydantic` — domain models and invariants
- `dspy` — LLM infrastructure (signatures, programs, optimizers, LiteLLM embeddings)
- `mlflow` — AI evaluation (code-based scorers); dev dependency
- `arxiv`, `biopython`, `habanero` — search-index clients
- `uv` — package manager, runner, build backend
- `ruff` — lint + format
- `ty` — type checker
- `pytest` + `pytest-cov` + `pytest-asyncio` — tests; 85% coverage on `src/research_agent`
- `prek` — git hooks

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
src/research_agent/          runtime; imports slices, wires workflows
  shared/                    cross-slice kernel (Agent, Session, executor, …)
  search/                    search slice (models, metrics, tools, agents, workflows)
src/datagen/                 synthetic query generation (NOT runtime)
src/optimize/                DSPy optimization pipeline (NOT runtime)
src/evals/                   MLflow scorers + eval agent wiring (NOT runtime)
tests/unit/                  deterministic tests (mirror package layout)
tests/external/              live tests (`live` marker; skipped by default)
data/                        generated artifacts (output/ gitignored except .gitkeep)
```

Do not stub unused reserved paths. Datagen entrypoint: `uv run generate-queries ...` → `data/datagen/output/queries_train.jsonl`.

## Architecture rules

- **Domain** — pure; owns meaning, invariants, ports, quality metrics. No DSPy/MLflow/LLM clients.
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
- **Workflow output is bare `list[PaperInfo]`.** No search-index provenance on results; `SearchIndexType` is a tool-dispatch key only.
- **Metrics live in the domain** as pure functions; `optimize` and `evals` consume them and must not reimplement “good.”
- **No code comments** (inline/trailing). Google docstrings required (`ruff`).
- **Let errors propagate.** Catch only to recover, translate, or re-raise with context — not to return `None`/empty defaults.
- **Project-specific exceptions** for rule violations: `class <Name>Error(Exception)`; raise directly.
- **No `cast`.** Prefer better types or control flow.
- **No bare lint suppressions.** `# noqa` / `# type: ignore` / `# ty: ignore` need an explanatory comment.

## Tooling boundaries

- `src/datagen` — synthetic queries → `data/datagen/output/queries_train.jsonl` via `uv run generate-queries`.
- `src/optimize` — DSPy optimization against live indexes; unfinished until `research_agent.search.program` exists (`NotImplementedError` is intentional).
- `src/evals` — MLflow scorer adapters + eval agent factories; run suites with `uv run -m evals.main --list` / `uv run -m evals.main --experiment NAME search-e2e search`.
- LM endpoints: copy `config/lm.example.yaml` → `config/lm.yaml` (gitignored). Loader is `research_agent.shared.lm_config` (Infrastructure). Roles `search-search` and `search-rerank` are `LMConfig` field maps.
- Do not commit under `data/**/output/` except `.gitkeep`.

## What NOT to do

- Do not import `datagen`, `optimize`, or `evals` from `research_agent`.
- Do not import DSPy, LiteLLM, MLflow, or other LLM libraries into Domain or Application.
- Do not import between slices.
- Do not reimplement domain metric logic in MLflow scorers (adapt I/O only; map `EvaluationScore` → `Feedback` with a metrics-module `source_id`).
- Do not bound `Agent` type variables to `BaseModel`.
- Do not re-validate structure Pydantic already enforces.

## Known gotchas

- `uv run pytest` runs unit tests + coverage on `src/research_agent` only; live tests need `uv run pytest -m live`.
- `ruff` uses `select = ["ALL"]`; do not add new backlog violations.
- `optimize/main.py` raises `NotImplementedError` until the student program exists — do not wire around it.
- CLI defaults in `datagen`/`optimize` must stay aligned with the `data/` layout.
