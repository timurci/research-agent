# AGENTS.md

Operating manual for AI coding agents working in this repository.

## Project overview

Research agent for scientific literature discovery. A single domain node (`SearchSuggestionNode`) takes a `ResearchQuery` and returns `SearchResults` via an LLM-backed agent. Relevance is enforced upstream through prompt optimization — there is no scoring or ranking stage in the domain. Two sibling tooling packages generate synthetic training queries (`src/datagen`) and run the DSPy optimization pipeline (`src/optimize`).

## Tech stack

- Python ≥ 3.14
- `pydantic` ≥ 2.13 — domain model contract validation
- `dspy` ≥ 3.2.1 — LLM infrastructure (signatures, programs, optimizers, embeddings via LiteLLM)
- `semanticscholar` ≥ 0.12 — Semantic Scholar index client
- `uv` — package manager and runner
- `hatchling` — build backend
- `ruff` — lint + format (`ruff.toml`: `select = ["ALL"]`, Google docstrings, ignore `COM812`)
- `ty` — type checker
- `pytest` + `pytest-cov` — tests (coverage gate 85% on `src/research_agent`); `pytest-asyncio` for async tests
- `prek` — git hook runner (`prek.toml`); runs `ruff-check --fix`, `ruff-format`, `ty check`, and `conventional-pre-commit`

## Environment setup

```bash
uv sync                     # install deps into .venv
uv run <command>            # run anything inside the project env
```

Do not use `pip` directly. Manage all dependencies through `uv`.

## Quality gates

Manual order for iterating outside the hooks:

```bash
uv run ruff check --fix src/
uv run ruff format src/
uv run ty check src/
uv run pytest
```

`prek.toml` runs `ruff-check --fix`, `ruff-format`, and `ty check` automatically (plus `conventional-pre-commit` on the commit message). Iterate via `prek`:

```bash
prek run                       # run configured hooks on the git staging area
prek run --all-files           # run on all tracked files
prek run <hook-id>             # run a single hook (e.g. ruff-check, ruff-format, ty)
prek run --files <path>        # target specific files
```

Current state (honest, as of last update):
- `ruff check` is configured strict (`select = ["ALL"]`); a backlog of violations exists across the touched packages and is being worked down. Do not introduce new violations — run `ruff check` and fix what your change adds.
- `ruff format` has not been applied to all files; run it before committing.
- `ty check` passes.
- Tests live under `tests/unit` (deterministic) and `tests/external` (live — hits the real Semantic Scholar API). Live tests are tagged `live` and skipped by default via the root `conftest.py`; run them explicitly with `uv run pytest -m live`. Coverage gate 85% on `src/research_agent` (`--cov src/research_agent -ra`) runs clean offline.
- `research_agent.search` is partially scaffolded: `search/tools.py` (the Semantic Scholar adapter + DSPy `build_search_tools()` factory) exists; the DSPy student program `search.program` is still pending (the `optimize/main.py` `NotImplementedError` gates on it), as is the domain service `search.node`.

## Project structure

The codebase is organized as **portable capability slices** under `src/research_agent/`, not as physical layer folders. Each slice is a self-contained folder grouping that capability's domain, application, and infrastructure concerns; layers are *virtual*, declared by the role documented in each file's docstring (e.g. `models.py` = Domain, `tools.py` = Infrastructure). A slice must be copyable into another project and work without relying on sibling slices — it may depend only on `research_agent.shared` (cross-slice contracts) and external libraries.

```
src/
  research_agent/                runtime shell; imports slices, wires workflows
    __init__.py
    shared/                       cross-slice kernel (contracts shared by >1 slice)
      __init__.py
      ports.py                    LanguageModel[InputT, OutputT] protocol (not yet present)
    search/                       THE search slice (SearchSuggestionNode); portable
      __init__.py                 slice docstring: layer-tag rules, dependency direction
      models.py                   [Layer: Domain] ResearchQuery, SearchResult, SearchResults, PaperReference, SearchIndexId, SearchIndexType
      tools.py                    [Layer: Infrastructure] SemanticScholarSearch + build_search_tools() (dspy-aware)
      node.py                     [Layer: Domain service] SearchSuggestionNode (not yet present)
      program.py                  [Layer: Infrastructure] SearchSuggestionProgram, DSPy student (not yet present)
    workflows.py                  [Layer: Application] workflow orchestration (not yet present)
    api/                          [Layer: Presentation] runtime entrypoints (not yet present)
  datagen/                        synthetic query generation tooling (NOT runtime)
  optimize/                       DSPy optimization pipeline for the search node (NOT runtime)
tests/
  unit/                           deterministic tests (mirror slice layout)
  external/                       live tests (tagged `live`, skipped by default)
conftest.py                       registers `live` marker, skips live tests unless `-m live`
prek.toml                         git hook config (ruff-check, ruff-format, ty, conventional-pre-commit)
ruff.toml                         lint + format config
data/                             generated artifacts (gitignored except .gitkeep)
  datagen/output/queries_train.jsonl
  optimize/output/compiled.json
```

Reserved-but-not-present files are documented for forward planning; do not stub them (YAGNI). The single present slice is `research_agent/search/`; `shared/` is reserved for the `LanguageModel` port, and `workflows.py`/`api/` are reserved for the application/presentation shell.

## Architecture rules

Layered architecture (Ports and Adapters + DDD), implemented as **portable capability slices** rather than physical layer folders. A slice groups one capability's domain, application, and infrastructure files into a single self-contained folder; layer membership is virtual — declared in each file's docstring via a `[Layer: …]` tag, not by directory.

Dependency arrows point inward, and slices are isolated from each other:

- A `Domain` module depends on nothing project-specific. Owns business meaning, invariants, and behavior.
- An `Application` module depends on its slice's `Domain`; contains no business logic, only orchestration.
- An `Infrastructure` module depends on its slice's `Domain` + DSPy; implements the `LanguageModel` port.
- No `Domain` or `Application` module depends on `Infrastructure`. DSPy, prompts, tools, and optimizers never leak into `Domain` or `Application` files.
- A slice may depend only on `research_agent.shared` (cross-slice contracts) and external libraries. Slices never import from each other. If two slices need the same contract, it belongs in `shared`.

The `LanguageModel[InputT, OutputT]` protocol lives in `research_agent.shared.ports` and is the only bridge between a slice's domain node and its LLM infrastructure. It is generic over Pydantic-bound input/output types. Each slice's node declares its concrete `InputT`/`OutputT`; the slice's infrastructure supplies a DSPy-backed implementation.

The runtime shell (`research_agent/__init__.py`, `workflows.py`, `api/`) imports slices and wires them into workflows and entrypoints. It owns no domain or infrastructure logic.

`src/datagen` and `src/optimize` are tooling, not runtime. They import from `research_agent.search.models` for validation; `research_agent` never imports from them. Optimization output is configured into a `LanguageModel` instance; the domain does not know optimization happened.

## Domain modeling conventions

- **Ubiquitous language.** A class name must tell the reader what the object *is* in domain terms. Vague `Request`/`Result` pairings are forbidden. Example: `ResultIdentifier` was renamed to `PaperReference` (+ `SearchIndexId`) because "result of what, identifier of what?" had no answer.
- **Pydantic validates I/O contracts; it does not enforce domain invariants.** A domain object should never exist in an inconsistent state — invariants are guaranteed at construction, not patched after.
- **Group tightly-coupled fields into their own model.** A native ID is meaningless without its index, so `SearchIndexId(index, id)` is a unit; `PaperReference` bundles `source: SearchIndexId` plus the cross-index `doi`.
- **One node in the current workflow.** `SearchSuggestionNode` takes `ResearchQuery`, returns `SearchResults`. Do not add a scoring/ranking node, `ScoreRequest`, `ScoredResults`, or categorical relevance labels. Relevance is enforced by prompt optimization, not a downstream domain stage.

## Code conventions

- **No code comments.** Do not add inline comments or trailing comments. Docstrings are required (Google convention, enforced via `ruff.toml`).
- **YAGNI.** Do not add abstractions, configuration hooks, plugin systems, or extra layers unless a current requirement justifies them.
- **KISS.** Prefer the simplest implementation that satisfies the requirement. Use plain functions and direct data structures before introducing classes, frameworks, or indirection.
- **Plain strings over enums when values are pure labels with no behavior.** `datagen/config.py` uses `INTENTS: list[str]` rather than a `StrEnum` because the values are only formatted into a prompt.
- **Let errors propagate.** Do not catch `Exception` to return `None`, empty collections, or silent fallbacks. Catch only when the handler does something meaningful (recover, translate, re-raise with context). `datagen/main.py`'s `except Exception as exc:  # noqa: BLE001` is intentional — it skips a failed stratum and logs, which is documented behavior.
- **Raise project-specific exceptions for violations.** When code detects a condition that violates a project rule, domain rule, invariant, required behavior, unsupported configuration, forbidden input shape, or impossible branch, raise a project-specific exception named `class <PascalCaseName>Error(Exception)` directly. Raise; do not return `None`, `False`, empty collections, or default objects. Raising exceptions is separate from catching them.
- **No `cast`.** Improve type definitions or narrow control flow instead.
- **Do not suppress lint without a comment.** `# noqa`, `# type: ignore`, and `# ty: ignore` are allowed only with an explanatory comment (e.g. `# noqa: BLE001  # skip-and-log is the intended behavior`). Prefer fixing annotations, control flow, or library typing boundaries over suppressing diagnostics.

## Tooling boundaries

- `src/datagen` — generates `queries_train.jsonl` only. Strata → `QueryGenerator` → Jaccard dedup → domain coverage check → write. Output: `data/datagen/output/queries_train.jsonl`.
- `src/optimize` — DSPy optimization. Loads queries, runs the search node live, scores `SearchResults` with the embedding-similarity metric `mean(sims) + 10 * n_results * min(sims)`. Metric, embedder, and dataset loader are usable today; `main.py` raises `NotImplementedError` at the optimizer-wiring step until `research_agent.search.program` (the DSPy student program) is built.
- `data/` — generated artifacts. `.gitkeep` files are committable; everything else under `data/**/output/` is gitignored.

## What NOT to do

- Do not import `datagen` or `optimize` from anywhere in `research_agent`.
- Do not import DSPy, LiteLLM, or any LLM library into any `[Layer: Domain]` or `[Layer: Application]` module (e.g. `search/models.py`, the future `search/node.py`, `workflows.py`).
- Do not import between slices. A slice depends only on `research_agent.shared` and external libraries.
- Do not commit files under `data/**/output/` other than `.gitkeep`.
- Do not add `Request`/`Result`-style domain models that bundle existing domain objects without adding meaning. The domain takes inputs separately; infrastructure bundles them in DSPy signatures.
- Do not add a second (scoring/ranking) node to the paper-discovery workflow without an explicit design decision — the current design omits it on purpose.
- Do not add code comments.

## Known gotchas

- `uv run pytest` runs `tests/unit` plus coverage; `tests/external` is deselected by default via the `live` marker (root `conftest.py`). Run live tests with `uv run pytest -m live`. Coverage gate 85% is scoped to `src/research_agent` only (`--cov src/research_agent`); `datagen` and `optimize` coverage is not measured.
- `ruff.toml` uses `select = ["ALL"]`. New code must not add violations to the existing backlog.
- `optimize/main.py` and the optimization pipeline are pending `research_agent.search.program`. Do not wire the optimizer until that file exists; the `NotImplementedError` is intentional.
- Generated output defaults to `data/...`; CLI argparse defaults in `datagen/main.py` and `optimize/main.py` must stay aligned with the `data/` convention.
