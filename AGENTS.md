# AGENTS.md

Operating manual for AI coding agents working in this repository. File-level rules and exact commands; the *why* behind each architectural decision is in [`docs/architecture.md`](docs/architecture.md).

## Project overview

A research assistant for scientific literature discovery. The current capability is the **search slice**: it takes a `ResearchQuery` and returns a `list[SearchResult]` via an LLM-backed `Agent`. The domain has no scoring or ranking stage — relevance is enforced upstream by prompt optimization. Two sibling tooling packages generate synthetic training queries (`src/datagen`) and run the DSPy optimization pipeline (`src/optimize`); neither is imported at runtime.

## Tech stack

- Python ≥ 3.14
- `pydantic` ≥ 2.13.4 — domain models, I/O contract validation, and invariant enforcement
- `dspy` ≥ 3.2.1 — LLM infrastructure (signatures, programs, optimizers, embeddings via LiteLLM)
- `arxiv`, `biopython`, `habanero`, `semanticscholar` — search-index clients
- `uv` — package manager, runner, and build backend (`uv_build`)
- `ruff` — lint + format (`ruff.toml`: `select = ["ALL"]`, Google docstrings, ignore `COM812`)
- `ty` — type checker
- `pytest` + `pytest-cov` + `pytest-asyncio` — tests, coverage gate 85% on `src/research_agent`
- `prek` — git hook runner

## Environment setup

```bash
uv sync                     # install deps into .venv
uv run <command>            # run anything inside the project env
```

Manage all dependencies through `uv`; do not use `pip` directly.

## Quality gates

A change is complete when all of the following pass:

```bash
uv run ruff check --fix src/
uv run ruff format src/
uv run ty check src/
uv run pytest
```

`prek.toml` runs `ruff-check --fix`, `ruff-format`, and `ty check` automatically on the staging area, plus `conventional-pre-commit` on the commit message. Iterate via prek:

```bash
prek run                     # hooks on the git staging area
prek run --all-files         # all tracked files
prek run <hook-id>           # single hook (ruff-check, ruff-format, ty)
```

## Project structure

Organized as **portable capability slices** under `src/research_agent/`, not physical layer folders. A slice is a self-contained folder grouping one capability's domain, application, and infrastructure files; layer membership is *virtual*, declared by each file's docstring role, not by subfolder. A slice may depend only on `research_agent.shared` and external libraries — never on a sibling slice.

```text
src/research_agent/              runtime shell; imports slices, wires workflows
  __init__.py                    runtime shell docstring (no logic yet)
  shared/                        cross-slice kernel (contracts shared by >1 slice)
    agent.py                     [Infrastructure] Agent[InputT, OutputT] protocol (async __call__)
    executor.py                  [Infrastructure] run_async — offloads sync SDKs to a thread pool
  search/                        THE search slice; portable
    __init__.py                  slice docstring: layer roles, dependency direction
    models.py                    [Layer: Domain] ResearchQuery, SearchResult, PaperInfo, PaperSource, SearchIndexReference, SearchIndexType
    tools.py                     [Layer: Infrastructure] LiteratureSearch + per-index handlers
    program.py                   [Layer: Infrastructure] DSPy student program (reserved)
  workflows.py                   [Layer: Application] workflow orchestration (reserved)
  api/                           [Layer: Presentation] runtime entrypoints (reserved)
src/datagen/                     synthetic query generation tooling (NOT runtime)
src/optimize/                    DSPy optimization pipeline for the search slice (NOT runtime)
tests/unit/                      deterministic tests (mirror slice layout)
tests/external/                  live tests (tagged `live`, skipped by default)
conftest.py                      registers `live` marker, skips live tests unless `-m live`
```

Do not stub reserved files (YAGNI). Run datagen via its entrypoint: `uv run generate-queries ...` → `data/datagen/output/queries_train.jsonl`.

## Architecture rules

Layered architecture (Ports and Adapters + DDD) as portable slices. Dependency arrows point inward; slices are isolated from each other. See [`docs/architecture.md`](docs/architecture.md) for rationale.

- A **Domain** module depends on nothing project-specific. It owns business meaning, invariants (Pydantic validators), ports, and quality metrics. It may contain pure domain services — logic that doesn't fit on a single model — added only when such logic exists; the current slice has none.
- An **Application** module depends on its slice's Domain; it orchestrates, calling ports (and domain services, when present), never adapters. No business logic.
- An **Infrastructure** module depends on its slice's Domain + DSPy; it implements ports with concrete adapters. No business logic, no use-case orchestration.
- No Domain or Application module imports Infrastructure. DSPy, prompts, tools, model names, and LLM clients never leak into Domain or Application files.
- A slice depends only on `research_agent.shared` and external libraries. Slices never import from each other; a contract needed by two slices lives in `shared`.

The `Agent[InputT, OutputT]` protocol in `research_agent.shared.agent` is the only bridge between a slice's domain and its LLM-backed infrastructure — `async def __call__(data: InputT) -> OutputT`. The type variables are deliberately unbounded: DSPy signatures wrap domain models in containers (`list[SearchResult]`, `dict[str, Model]`), so the port's output type is often a container, not a `BaseModel`. Boundary validation comes from typing a signature's output field as a domain model, not from bounding the port.

Application methods and the ports they call are `async`. Synchronous, blocking work (e.g. a synchronous search-index SDK) stays private to infrastructure — offload it via `shared.executor.run_async` — and never surface `async`/threading in a port signature.

`src/datagen` and `src/optimize` are tooling, not runtime. `research_agent` never imports from them. Optimization output is loaded into an `Agent` adapter at the composition root; the domain does not know optimization happened.

## Domain modeling conventions

- **Ubiquitous language.** A class name must say what the object *is* in domain terms; vague `Request`/`Result` pairings are forbidden. (`ResultIdentifier` was renamed to `SearchIndexReference` + `PaperSource` because "result of what, identifier of what?" had no answer.)
- **Pydantic encodes domain invariants.** A domain object must never exist in an inconsistent state; enforce invariants with Pydantic validators at construction, not by patching after. (e.g. `ResearchQuery.text = Field(min_length=5)`.)
- **Group tightly-coupled fields into their own model.** A native ID is meaningless without its index, so `SearchIndexReference(index, id)` is a unit; `SearchResult` bundles `paper: PaperInfo` (with `source: PaperSource`) plus the cross-index `search_reference`.
- **Quality metrics and LLM-judge rubrics are domain knowledge.** Define them in the domain layer as pure functions over domain value objects; the optimizer in `src/optimize` consumes them. This is a *definition*, not a runtime stage.
- **One capability in the current workflow.** The search slice takes `ResearchQuery`, returns `list[SearchResult]`. Do not add a scoring/ranking stage, `ScoreRequest`, `ScoredResults`, or categorical relevance labels — relevance is enforced by optimization, not a downstream domain stage.

## Code conventions

- **No code comments.** No inline or trailing comments. Docstrings are required (Google convention, enforced via `ruff.toml`).
- **YAGNI.** Do not add abstractions, configuration hooks, plugin systems, or extra layers unless a current requirement justifies them.
- **KISS.** Prefer the simplest implementation that satisfies the requirement — plain functions and direct data structures before classes, frameworks, or indirection.
- **Plain strings over enums when values are pure labels with no behavior.** (`datagen/config.py` uses `INTENTS: list[str]` rather than a `StrEnum`.)
- **Let errors propagate.** Do not catch `Exception` to return `None`, empty collections, or silent fallbacks. Catch only when the handler does something meaningful (recover, translate, re-raise with context). (`datagen/main.py`'s `except Exception as exc:  # noqa: BLE001` is intentional — skip-and-log is documented behavior.)
- **Raise project-specific exceptions for violations** — a project rule, domain rule, invariant, required behavior, unsupported configuration, forbidden input shape, or impossible branch. Name them `class <PascalCaseName>Error(Exception)`; raise directly. Do not return `None`, `False`, empty collections, or default objects. Raising is separate from catching.
- **No `cast`.** Improve type definitions or narrow control flow instead.
- **Do not suppress lint without a comment.** `# noqa`, `# type: ignore`, and `# ty: ignore` require an explanatory comment (e.g. `# noqa: BLE001  # skip-and-log is the intended behavior`). Prefer fixing annotations, control flow, or library typing boundaries over suppressing.

## Tooling boundaries

- `src/datagen` — generates `queries_train.jsonl` only. Strata → `QueryGenerator` → Jaccard dedup → domain coverage check → write. Run via `uv run generate-queries`. Output: `data/datagen/output/queries_train.jsonl`.
- `src/optimize` — DSPy optimization. Loads queries, runs the search capability live against real indexes, scores `list[SearchResult]` with the embedding-similarity metric `mean(sims) + 10 * n_results * min(sims)`. `main.py` raises `NotImplementedError` at the optimizer-wiring step until `research_agent.search.program` (the DSPy student program) is built.
- `data/` — generated artifacts. `.gitkeep` files are committable; everything else under `data/**/output/` is gitignored.

## What NOT to do

- Do not import `datagen` or `optimize` from anywhere in `research_agent`.
- Do not import DSPy, LiteLLM, or any LLM library into any Domain or Application module (`search/models.py`, the future `workflows.py`).
- Do not import between slices. A slice depends only on `research_agent.shared` and external libraries.
- Do not commit files under `data/**/output/` other than `.gitkeep`.
- Do not add `Request`/`Result`-style domain models that bundle existing domain objects without adding meaning. The domain takes inputs separately; infrastructure bundles them in DSPy signatures.
- Do not add a scoring/ranking stage to the search workflow without an explicit design decision — the current design omits it on purpose.
- Do not add code comments.
- Do not bound the `Agent` port's type variables; the output type may be a container of models.

## Known gotchas

- `uv run pytest` runs `tests/unit` plus coverage; `tests/external` is deselected by default via the `live` marker (root `conftest.py`). Run live tests with `uv run pytest -m live`. The 85% coverage gate is scoped to `src/research_agent` only (`--cov src/research_agent`); `datagen` and `optimize` coverage is not measured. `raise NotImplementedError` and `...` Protocol placeholders are excluded from coverage.
- `ruff.toml` uses `select = ["ALL"]`. New code must not add violations to the existing backlog.
- `optimize/main.py` raises `NotImplementedError` pending `research_agent/search/program.py`. Do not wire the optimizer until that file exists; the error is intentional.
- Generated output defaults to `data/...`; CLI argparse defaults in `datagen/main.py` and `optimize/main.py` must stay aligned with the `data/` convention.
