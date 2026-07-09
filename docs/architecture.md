# Architecture

This document records the architectural decisions and design principles for the project, at a high level. It states *what* and *why*, not the file-level *how* — implementation detail lives in the code and is expected to change, so it is deliberately not enumerated here. File-level structure and coding conventions live in [`AGENTS.md`](../AGENTS.md).

## Goals

- Keep the domain layer free of LLM-framework concerns.
- Treat DSPy as replaceable infrastructure.
- Preserve DSPy's optimization capabilities without leaking prompts, tools, or control flow into the domain.
- Make domain logic testable without invoking a language model.
- Keep domain logic in the domain — invariants on models, pure logic in domain services when it exists — rather than leaking it into infrastructure. The domain may be thin when the core capability is LLM-powered; that is not anemic, since the LLM call lives behind a port by design.
- Keep each capability **portable**: a slice can be copied into another project and work without relying on sibling slices.

## Layered architecture as portable slices

The project follows a layered architecture inspired by Ports and Adapters (Hexagonal Architecture) and Domain-Driven Design, organized as **portable capability slices** rather than physical layer folders.

A slice is a self-contained folder grouping one capability's domain, application, and infrastructure concerns. Layer membership is *virtual* — declared in each file's docstring, not by subfolder — so a slice mirrors the layer model internally rather than spanning three top-level layer packages. The runtime shell (workflows, entrypoints) supplies the application and presentation layers on top.

Slices are isolated from each other: a slice may depend only on the shared kernel and external libraries, never on a sibling slice. Contracts shared by more than one slice live in `research_agent.shared`, the cross-slice kernel. The first slice, `search`, owns the search-suggestion capability (a `ResearchQuery` in, search results out).

```
application  ──depends on──▶  domain
infrastructure ──depends on──▶  domain
slices depend only on shared + external libs
```

The virtual-layer choice is a deliberate deviation from physical layering; the dependency-direction rule and the placement tests below still govern what belongs where within it.

## The three layers

**Domain** — business meaning, invariants, and behavior. Pure Python + Pydantic. Holds value objects and entities, capability ports, and quality metrics. Domain services — pure logic that doesn't fit on a single model — are an optional home for such logic, added only when it appears (e.g. deduplicating results across indexes, reconciling conflicting metadata); the current slice has none, so none is expected. No DSPy, prompts, model names, tools, or LLM clients appear here. Pydantic is the mechanism for both I/O contract validation *and* domain invariants — a domain object should never exist in an inconsistent state, and invariants are enforced at construction (Pydantic validators), not patched afterwards. Typing a DSPy signature's output field as a domain model (or a container of them) gives validation at the LLM boundary: an invalid value fails loudly at the seam, not silently downstream.

**Application** — use-case orchestration. Thin services that coordinate capabilities and translate between presentation and domain. No business logic; it calls ports (and domain services, when present), not adapters.

**Infrastructure** — the "how." DSPy signatures, programs, tools, optimizers, and LM clients. Signatures are authored in domain vocabulary and reference domain models directly; a wrapper model for a collection only earns its keep if there's a collection-level invariant, otherwise plain `list[Model]` is honest. Tools hold no business rules — schema, binding, and invocation are infra; a tool may delegate to a domain service when it needs logic.

## The `Agent` seam

The port contract the application uses to invoke a language-model-based capability is the **`Agent` seam**: a generic `Agent[InputT, OutputT]` protocol with `async def __call__(data: InputT) -> OutputT`. This is the project's answer to the LM-seam question — the *single generic port* variant (one protocol parameterized by input/output types) rather than a per-capability typed port for each capability.

The type variables are deliberately **unbounded**. DSPy signatures routinely wrap domain models in containers (`list[SearchResult]`, `dict[str, Model]`), so the port's output type is often a container, not a `BaseModel` itself; bounding to `BaseModel` would forbid that, and there is no reason to constrain the port. The boundary-validation payoff does not come from bounding the port — it comes from typing the *signature's* output field as a domain model, where Pydantic validates.

The seam is the only place a model "shows up" above the adapter boundary. It determines provider swap (the adapter behind the port changes; the application and domain don't), offline replay (a fake `Agent` returning canned outputs), and the fakes that make the whole thing unit-testable.

## Async by default

Application-layer methods and the ports they call are `async` — a Python concurrency convention, not an infrastructure dependency. Synchronous, blocking work (e.g. wrapping a synchronous SDK) stays private to infrastructure, offloaded to a thread pool inside the adapter; it never surfaces in a port signature. The domain stays pure.

## Optimization — separate, pinned by contracts

DSPy optimization is build-time, not request-handling. It lives in a sibling tooling package (`src/optimize`), not in the runtime application, and is never imported by `research_agent` at runtime. It is not forced into clean architecture — no ports, fakes, or unit-of-work; it is a straightforward pipeline that loads examples, runs the student program, scores, and saves the compiled artifact.

It stays pinned to the runtime by two shared contracts, so build-time and run-time cannot silently drift:

1. **The same signatures** — optimization compiles against the exact signature definitions the runtime adapter loads.
2. **The same domain metrics** — quality metrics and LLM-judge rubrics are domain knowledge; their *definitions* live in the domain layer as pure functions over domain value objects. The optimizer that consumes them lives in the optimization package. Sharing the definition prevents metric drift.

A pure metric function in the domain is a *shared definition*, not a runtime stage — it is distinct from a scoring/ranking stage in the runtime workflow, which the project deliberately omits (relevance is enforced by optimization, not a downstream domain stage).

## Deferred — do not pre-build

Parked for when the pressure is real. The intended pattern is stated so a future contributor doesn't invent the wrong shape.

- **Observability** — out of scope for v1. When added: use-case-level tracing via a middleware wrapper applied at the composition root; LLM-call-level tracing via DSPy's native integration; span nesting via `contextvars` so span handles never flow through domain signatures. No `Tracer` port in the domain.
- **Composition root** — the runtime shell will hand-wire objects at startup: per-use-case factory functions, no DI container (the graph is small), secrets and config injected at the seam via a settings object. Artifact loading follows Style A — the root loads the compiled DSPy program and hands a ready module to the adapter; version selection is a deployment knob. No domain repository for artifacts (an artifact is not a domain aggregate).
- **Memory** — out of scope for v1. When added: short-term (session) memory travels as part of the capability input; long-term memory is exposed as a tool or injected into the prompt as metadata. Neither changes the domain port.
- **A second capability slice** — the slice/kernel structure is in place; do not pre-build a second slice or abstract the kernel before a second capability exists.

## Testing strategy

- **Domain tests** mock the `Agent` port — no DSPy, no LM, no network.
- **Application tests** mock the ports and services they orchestrate.
- **Infrastructure tests** exercise signatures, programs, and adapters, optionally against a real or cached LM. Live tests that hit a real external index live under `tests/external/`, tagged `live`, skipped by default.

## Design heuristics

Four tests decide whether code sits in the right layer.

1. **Verb-noun workflow test** — a component with an ordered verb-noun workflow is a use case hiding in infra. Promote it.
2. **Swap test** — can you replace DSPy with a hand-written adapter without touching the application or domain? If not, the port boundary is wrong.
3. **Fake test** — can you unit-test the use case with in-memory fakes and no network? If not, infra has leaked.
4. **Narrative test** — would a domain expert name this step? Yes → application or domain. No → infra.

## Principles summary

1. The domain owns meaning, invariants, and behavior; Pydantic encodes the invariants.
2. The `Agent` port is the only bridge between a slice's domain and its LLM-backed infrastructure.
3. DSPy signatures, programs, tools, and optimizers live in infrastructure; signatures reference domain models directly.
4. Quality metrics and LLM-judge rubrics are domain knowledge; their definitions live in the domain.
5. Optimization is a separate, build-time package pinned to the runtime by shared signatures and shared metrics.
6. Workflows orchestrate; models enforce invariants, and LLM-powered reasoning lives behind the `Agent` port.
7. Dependencies point inward; infrastructure is replaceable; slices are portable and isolated.
