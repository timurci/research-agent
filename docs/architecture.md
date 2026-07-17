# Architecture

High-level decisions and design principles — *what* and *why*. File-level structure, commands, and coding rules live in [`AGENTS.md`](../AGENTS.md). Implementation detail lives in the code.

## Goals

- Keep the domain free of LLM-framework concerns.
- Treat DSPy as replaceable infrastructure.
- Preserve DSPy optimization without leaking prompts, tools, or control flow into the domain.
- Make domain logic testable without invoking a language model.
- Keep each capability **portable**: a slice can be copied into another project without sibling slices.

## Portable slices and layers

The project uses Ports and Adapters + DDD, organized as **portable capability slices** rather than physical layer folders. A slice is a self-contained folder for one capability’s domain, application, and infrastructure. Layer membership is *virtual* — declared in each file’s docstring, not by subfolder.

Slices are isolated: a slice depends only on `research_agent.shared` and external libraries, never on a sibling slice. Contracts needed by more than one slice live in `shared`. The first slice, `search`, owns literature search (`ResearchQuery` in, papers out).

```
application  ──depends on──▶  domain
infrastructure ──depends on──▶  domain
slices depend only on shared + external libs
```

**Domain** — business meaning, invariants, ports, and quality metrics. Pure Python + Pydantic. No DSPy, prompts, model names, tools, or LLM clients. Domain services (pure logic that does not fit a single model) exist only when that logic appears.

**Application** — use-case orchestration. Calls ports (and domain services when present), never adapters. No business logic.

**Infrastructure** — DSPy signatures, programs, tools, optimizers, LM clients. Signatures use domain vocabulary and domain models directly. Tools hold no business rules.

## Seams

**`Agent[InputT, OutputT]`** — the single generic LM port (`async def __call__(data: InputT) -> OutputT`). Type variables are unbounded so outputs may be containers (`list[PaperInfo]`). Boundary validation comes from typing signature fields as domain models, not from bounding the port. The seam is where provider swap, offline fakes, and unit tests attach.

**`Session`** — application port for session-scoped key/value working memory (not a domain repository). Default adapter: `InMemorySession` in the same module.

Application methods and the ports they call are `async`. Synchronous blocking work stays private to infrastructure (e.g. offloaded via a thread pool) and never appears on a port signature.

## Tooling packages

DSPy optimization (`src/optimize`) and offline evaluation (`src/evals`) are build-time tooling, not request-handling. They are never imported by `research_agent` at runtime. They stay pinned to the runtime by:

1. **Shared signatures** — optimization compiles against the same signature definitions the runtime adapters use.
2. **Shared domain metrics** — quality definitions live in the domain as pure functions over domain value objects. The optimizer and MLflow scorers consume them; scorers adapt I/O and do not redefine “good.”

`src/datagen` is the same class of sibling tooling (synthetic training queries).

Do not pre-build observability middleware, a DI container, long-term memory, or a second capability slice until real pressure exists.

## Design heuristics

1. **Verb-noun workflow** — an ordered verb-noun pipeline in infrastructure is a use case hiding there; promote it.
2. **Swap** — can you replace DSPy with a hand-written adapter without touching application or domain? If not, the port is wrong.
3. **Fake** — can you unit-test the use case with in-memory fakes and no network? If not, infrastructure has leaked.
4. **Narrative** — would a domain expert name this step? Yes → application or domain. No → infrastructure.
