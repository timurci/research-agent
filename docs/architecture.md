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

**`Agent[InputT, OutputT]`** — the single generic LM port (`async def __call__(data: InputT) -> OutputT`). Type variables are unbounded so outputs may be containers or tuples (`list[PaperInfo]`, `tuple[list[PaperInfo], str]`). Boundary validation comes from typing signature fields as domain models, not from bounding the port. The seam is where provider swap, offline fakes, and unit tests attach.

**`Session`** — application port for session-scoped key/value working memory (not a domain repository). Default adapter: `InMemorySession` in the same module.

Application methods and the ports they call are `async`. Synchronous blocking work stays private to infrastructure (e.g. offloaded via a thread pool) and never appears on a port signature.

## Tooling packages

DSPy optimization (`src/optimize`) and offline evaluation (`src/evals`) are build-time tooling, not request-handling. They are never imported by `research_agent` at runtime. They stay pinned to the runtime by:

1. **Shared signatures** — optimization compiles against the same signature definitions the runtime adapters use.
2. **Shared domain metrics** — quality definitions live in the domain as pure functions over domain value objects. The optimizer and Opik scorers consume them; scorers adapt I/O and do not redefine “good.”

`src/datagen` is the same class of sibling tooling (synthetic training queries).

## Runtime composition and observability

`research_agent.app` is the composition root for serving paper search as a
library API. `research_agent.api` is the FastAPI presentation layer on top of
that facade (HTTP only; no domain or workflow logic):

- `build_paper_search_app(...)` loads LM and instructions config, constructs
  agents, and returns a `PaperSearchApp`. Optional PubMed/OpenAlex API keys
  are injected there and forwarded to `LiteratureSearch`.
- Each `search` call builds a fresh `InMemorySession`, `SearchAgent`, and
  `SuggestionGenerator` so session bags and DSPy program/LM instances stay
  isolated under concurrent async calls. That also rebuilds the search and
  suggest graphs and reloads optimized program JSON when instructions are
  configured. Reranker and literature client are shared for the life of the
  app.
- Root Opik traces wrap the whole workflow; the client flushes after the
  trace context exits (off the event loop) so `SearchRun.trace_id` is ready
  for immediate `record_feedback`. Feedback logging also flushes. Instruction
  file hashes in root-trace metadata are recomputed per `search`.
- `configure_observability=True` (default) registers process-wide
  `OpikCallback` via `dspy.configure`, replacing any existing DSPy callbacks.
  Hosts that already configure DSPy should pass `False` and own that setup.
- Nested DSPy/`OpikCallback` spans cover search and suggest LM work. Rerank
  uses LiteLLM outside DSPy, so it appears only as wall time under the root
  trace (no nested rerank span today).
- `SearchRun.trace_id` is the handle for optional later feedback
  (thumbs + comment → Opik `user_useful` score via async `record_feedback`).
- HTTP surface (`POST /search`, `POST /feedback`, `GET /health`) lives under
  `research_agent.api` and depends only on `PaperSearchApp`. The API lifespan
  flushes Opik on shutdown.

Observability is middleware applied at the composition root — not a domain
port. `PaperSearchWorkflow` stays free of Opik. Optimized instructions still
load from files; Opik catalogs and correlates only.

Do not pre-build a DI container, long-term memory, or a second capability
slice until real pressure exists.

## Design heuristics

1. **Verb-noun workflow** — an ordered verb-noun pipeline in infrastructure is a use case hiding there; promote it.
2. **Swap** — can you replace DSPy with a hand-written adapter without touching application or domain? If not, the port is wrong.
3. **Fake** — can you unit-test the use case with in-memory fakes and no network? If not, infrastructure has leaked.
4. **Narrative** — would a domain expert name this step? Yes → application or domain. No → infrastructure.
