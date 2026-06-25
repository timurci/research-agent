# AGENTS.md

Operating manual for AI coding agents working in this repository.

## Project overview

Research agent for scientific literature discovery. A single domain node (`SearchSuggestionNode`) takes a `ResearchQuery` and returns `SearchResults` via an LLM-backed agent. Relevance is enforced upstream through prompt optimization — there is no scoring or ranking stage in the domain. Two sibling tooling packages generate synthetic training queries (`src/datagen`) and run the DSPy optimization pipeline (`src/optimize`).

## Tech stack

- Python ≥ 3.14
- `pydantic` ≥ 2.13 — domain model contract validation
- `dspy` ≥ 3.2.1 — LLM infrastructure (signatures, programs, optimizers, embeddings via LiteLLM)
- `uv` — package manager and runner
- `hatchling` — build backend
- `ruff` — lint + format (`ruff.toml`: `select = ["ALL"]`, Google docstrings, ignore `COM812`)
- `ty` — type checker
- `pytest` + `pytest-cov` — tests (coverage gate 85% on `src/research_agent`)

## Environment setup

```bash
uv sync                     # install deps into .venv
uv run <command>            # run anything inside the project env
```

Do not use `pip` directly. Manage all dependencies through `uv`.

## Quality gates

Run in this order before considering work done:

```bash
uv run ruff check --fix src/
uv run ruff format src/
uv run ty check src/
uv run pytest
```

Current state (honest, as of last update):
- `ruff check` is configured strict (`select = ["ALL"]`); a backlog of violations exists across the touched packages and is being worked down. Do not introduce new violations — run `ruff check` and fix what your change adds.
- `ruff format` has not been applied to all files; run it before committing.
- `ty check` passes.
- `pytest` has no tests yet AND the `pyproject.toml` `addopts` force `--cov src/research_agent -ra` with `fail_under = 85`. With no tests, `uv run pytest` fails on coverage. To iterate without the gate, run `uv run pytest --no-cov -p no:cacheprovider` until tests are added; restore the full command once tests exist.

## Project structure

```
src/
  research_agent/           runtime application (the actual product)
    domain/
      models.py             Pydantic domain models (meaning + contracts)
      ports.py              LanguageModel protocol (not yet present)
      nodes/                domain services (not yet present)
    application/            workflow orchestration (not yet present)
    infrastructure/        DSPy adapter, signatures, programs, tools (not yet present)
  datagen/                  synthetic query generation tooling (NOT runtime)
  optimize/                 DSPy optimization pipeline for the search node (NOT runtime)
docs/
  architecture.md            layered architecture, ports, optimization location
  paper_discovery_workflow.md  single-node workflow design
  data_generation_plan.md   datagen plan
  research_api_comparison.md  index API survey
data/                       generated artifacts (gitignored except .gitkeep)
  datagen/output/queries_train.jsonl
  optimize/output/compiled.json
```

`research_agent/application`, `research_agent/infrastructure`, and `research_agent/domain/{ports,nodes}/` do not exist yet — they are scaffolded in `docs/architecture.md`. Follow that doc when adding them.

## Architecture rules

Layered architecture (Ports and Adapters + DDD). Dependency arrows point inward:

- `domain` depends on nothing project-specific. Owns business meaning, invariants, and behavior.
- `application` depends on `domain`; contains no business logic, only orchestration.
- `infrastructure` depends on `domain` + DSPy; implements the `LanguageModel` port.
- No layer above depends on `infrastructure`. DSPy, prompts, tools, and optimizers never leak into `domain` or `application`.

The `LanguageModel[InputT, OutputT]` protocol is the only bridge between the domain and LLM infrastructure. It is generic over Pydantic-bound input/output types. New nodes declare their concrete `InputT`/`OutputT`; the infrastructure supplies a DSPy-backed implementation.

`src/datagen` and `src/optimize` are tooling, not runtime. They import from `research_agent.domain.models` for validation; `research_agent` never imports from them. Optimization output is configured into a `LanguageModel` instance; the domain does not know optimization happened.

## Domain modeling conventions

- **Ubiquitous language.** A class name must tell the reader what the object *is* in domain terms. Vague `Request`/`Result` pairings are forbidden. Example: `ResultIdentifier` was renamed to `PaperReference` (+ `SearchIndexId`) because "result of what, identifier of what?" had no answer.
- **Pydantic validates I/O contracts; it does not enforce domain invariants.** A domain object should never exist in an inconsistent state — invariants are guaranteed at construction, not patched after.
- **Group tightly-coupled fields into their own model.** A native ID is meaningless without its index, so `SearchIndexId(index, id)` is a unit; `PaperReference` bundles `source: SearchIndexId` plus the cross-index `doi`.
- **One node in the current workflow.** `SearchSuggestionNode` takes `ResearchQuery`, returns `SearchResults`. Do not add a scoring/ranking node, `ScoreRequest`, `ScoredResults`, or categorical relevance labels. Relevance is enforced by prompt optimization, not a downstream domain stage.

## Code conventions

- **No code comments.** Do not add inline comments or trailing comments. Docstrings are required (Google convention, enforced via `ruff.toml`).
- **Plain strings over enums when values are pure labels with no behavior.** `datagen/config.py` uses `INTENTS: list[str]` rather than a `StrEnum` because the values are only formatted into a prompt.
- **Let errors propagate.** Do not catch `Exception` to return `None`, empty collections, or silent fallbacks. Catch only when the handler does something meaningful (recover, translate, re-raise with context). `datagen/main.py`'s `except Exception as exc:  # noqa: BLE001` is intentional — it skips a failed stratum and logs, which is documented behavior.
- **No `cast`.** Improve type definitions or narrow control flow instead.
- **Do not suppress lint without a comment.** `# noqa` is allowed only with an explanatory comment (e.g. `# noqa: BLE001  # skip-and-log is the intended behavior`).

## Tooling boundaries

- `src/datagen` — generates `queries_train.jsonl` only. Strata → `QueryGenerator` → Jaccard dedup → domain coverage check → write. Output: `data/datagen/output/queries_train.jsonl`.
- `src/optimize` — DSPy optimization. Loads queries, runs the search node live, scores `SearchResults` with the embedding-similarity metric `mean(sims) + 10 * n_results * min(sims)`. Metric, embedder, and dataset loader are usable today; `main.py` raises `NotImplementedError` at the optimizer-wiring step until `research_agent.infrastructure.llm` (the DSPy student program) is built.
- `data/` — generated artifacts. `.gitkeep` files are committable; everything else under `data/**/output/` is gitignored.

## What NOT to do

- Do not import `datagen` or `optimize` from anywhere in `research_agent`.
- Do not import DSPy, LiteLLM, or any LLM library into `research_agent.domain` or `research_agent.application`.
- Do not commit files under `data/**/output/` other than `.gitkeep`.
- Do not add `Request`/`Result`-style domain models that bundle existing domain objects without adding meaning. The domain takes inputs separately; infrastructure bundles them in DSPy signatures.
- Do not add a second (scoring/ranking) node to the paper-discovery workflow without an explicit design decision — the current design omits it on purpose.
- Do not add code comments.

## Known gotchas

- `uv run pytest` fails with no tests because of the 85% coverage gate in `pyproject.toml`. Coverage is scoped to `src/research_agent` only (`--cov src/research_agent`); `datagen` and `optimize` coverage is not measured. Add tests under `tests/` (not yet created) following `pytest` defaults.
- `ruff.toml` uses `select = ["ALL"]`. New code must not add violations to the existing backlog.
- `optimize/main.py` and the optimization pipeline are pending `research_agent.infrastructure.llm`. Do not wire the optimizer until that package exists; the `NotImplementedError` is intentional.
- Generated output defaults to `data/...`; CLI argparse defaults in `datagen/main.py` and `optimize/main.py` must stay aligned with the `data/` convention.