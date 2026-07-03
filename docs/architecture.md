# Architecture

This document records the architectural decisions and design principles for the project.

## Goals

- Keep the domain layer free of LLM-framework concerns.
- Treat DSPy as replaceable infrastructure.
- Preserve DSPy's optimization capabilities without leaking prompts, tools, or control flow into the domain.
- Make domain logic testable without invoking a language model.
- Maintain a rich domain model rather than an anemic one.
- Keep each capability **portable**: a slice can be copied into another project and work without relying on sibling slices.

## Layered architecture as portable slices

The project follows a layered architecture inspired by Ports and Adapters (Hexagonal Architecture) and Domain-Driven Design, but implemented as **portable capability slices** rather than physical layer folders.

A slice is a self-contained folder grouping one capability's domain, application, and infrastructure concerns. Layer membership is *virtual* — declared in each file's docstring via a `[Layer: …]` tag, not by subfolder. The single present slice is `research_agent/search/`, which owns the `SearchSuggestionNode` capability end to end.

```text
┌─────────────────────────────────────────┐
│  Application Layer                      │
│  Workflows orchestrate nodes            │
├─────────────────────────────────────────┤
│  Domain Layer                           │
│  Models, domain nodes (services), ports │
├─────────────────────────────────────────┤
│  Infrastructure Layer                   │
│  DSPy adapter, signatures, programs,    │
│  tools, optimizers, LM clients          │
└─────────────────────────────────────────┘
```

Within a slice, layers are files, not folders: `models.py` is Domain, `node.py` is a Domain service, `tools.py`/`program.py` are Infrastructure. A slice mirrors the layer model internally; the runtime shell (`workflows.py`, `api/`) supplies the Application and Presentation layers on top.

## Cross-slice kernel: `research_agent.shared`

Contracts shared by more than one slice live in `research_agent.shared`, the only package a slice may depend on besides external libraries. Slices never import from each other; if two slices need the same contract, it belongs in `shared`.

The central shared contract is the `LanguageModel` port.

## Domain layer

The domain layer owns business meaning, invariants, and behavior. Within a slice it lives in `models.py` (value objects/entities) and `node.py` (domain services).

### Domain models

Domain models are implemented as Pydantic models. Pydantic is used for **input/output contract validation**, not for encoding domain invariants. A domain object should never exist in an inconsistent state; invariants are guaranteed at construction time, not patched afterwards.

In the search slice these are `ResearchQuery`, `SearchResult`, `PaperInfo`, `PaperSource`, and `SearchIndexReference`:

```python
from pydantic import BaseModel

class ResearchQuery(BaseModel):
    text: str
    domains: list[str] | None = None

class SearchResult(BaseModel):
    paper: PaperInfo
    search_reference: SearchIndexReference
```

### Domain nodes

A **node** is a domain service that represents a bounded capability. It contains business rules and collaborates with a `LanguageModel` port to perform language-model-powered reasoning. A node does not know about DSPy, prompts, tools, or optimizers.

The search slice exposes `SearchSuggestionNode`, which takes a `ResearchQuery` and returns `list[SearchResult]`:

```python
from research_agent.search.models import ResearchQuery, SearchResult
from research_agent.shared.ports import LanguageModel

class SearchSuggestionNode:
    def __init__(self, llm: LanguageModel[ResearchQuery, list[SearchResult]]) -> None:
        self._llm = llm

    def suggest(self, query: ResearchQuery) -> list[SearchResult]:
        if not query.text.strip():
            raise ValueError("Cannot suggest with an empty query")
        return self._llm.generate(query)
```

### Domain ports

Ports are protocols defined in the shared kernel. They declare what capabilities the domain needs from the outside world without prescribing an implementation.

The central port is `LanguageModel`, in `research_agent.shared.ports`:

```python
from typing import Protocol, TypeVar
from pydantic import BaseModel

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)

class LanguageModel(Protocol[InputT, OutputT]):
    """Driven port: send a domain input, receive a domain output."""
    def generate(self, input: InputT) -> OutputT: ...
```

The protocol is generic so each node can declare its concrete input and output types. The `bound=BaseModel` constraint enforces that only structured Pydantic models pass through the port.

## Application layer

The application layer orchestrates nodes to fulfill use cases. Workflows are plain Python services in the runtime shell (`research_agent/workflows.py`). They contain **no business logic**; they only coordinate slices' domain nodes and translate between presentation concerns and domain inputs.

```python
from research_agent.search.models import ResearchQuery, SearchResult
from research_agent.search.node import SearchSuggestionNode

class DiscoveryWorkflow:
    def __init__(self, search_node: SearchSuggestionNode) -> None:
        self._search = search_node

    def run(self, query: ResearchQuery) -> list[SearchResult]:
        return self._search.suggest(query)
```

## Infrastructure layer

The infrastructure layer owns every LLM-specific concern: DSPy signatures, programs, tools, optimizers, and LM clients. Within a slice these live in `tools.py` and `program.py`.

### DSPy adapter

The adapter implements the `LanguageModel` port using a DSPy program. It will live in the search slice alongside the program it wraps.

```python
import dspy
from research_agent.shared.ports import LanguageModel, InputT, OutputT

class DSPyLanguageModel(LanguageModel[InputT, OutputT]):
    def __init__(self, program: dspy.Module) -> None:
        self._program = program

    def generate(self, input: InputT) -> OutputT:
        prediction = self._program(input=input)
        return prediction.output
```

### DSPy signatures

Signatures import a slice's domain models directly and add prompt-level instructions. They live in infrastructure because they are part of the LLM contract, not the domain contract.

```python
import dspy
from research_agent.search.models import ResearchQuery, SearchResult

class SuggestSearch(dspy.Signature):
    """Suggest a set of relevant papers for a research query."""
    input: ResearchQuery = dspy.InputField()
    output: list[SearchResult] = dspy.OutputField()
```

### DSPy programs

Programs wire signatures, tools, and predictors together. The search slice's student program will live in `research_agent.search.program` as `SearchSuggestionProgram`.

```python
import dspy
from research_agent.search.tools import LiteratureSearch

class SearchSuggestionProgram(dspy.Module):
    def __init__(self) -> None:
        self.re_act = dspy.ReAct(
            SuggestSearch,
            tools=[dspy.Tool(LiteratureSearch(), name="literature_search")],
        )

    def forward(self, input: ResearchQuery) -> dspy.Prediction:
        return self.re_act(input=input)
```

### Tools

Tool definitions and tool-calling mechanics belong to infrastructure. A tool may delegate to a domain service if it needs business logic, but the tool schema, binding, and invocation are infrastructure concerns. In the search slice they live in `search/tools.py`:

```python
import dspy
from research_agent.search.tools import LiteratureSearch

def build_literature_tool() -> dspy.Tool:
    return dspy.Tool(LiteratureSearch(), name="literature_search")
```

No business rule should be encoded inside a tool itself.

## Dependency direction

Dependency arrows point inward, and slices are isolated from each other:

- A `Domain` module depends on nothing project-specific.
- An `Application` module depends on its slice's `Domain`.
- An `Infrastructure` module depends on its slice's `Domain` + DSPy.
- No `Domain` or `Application` module depends on `Infrastructure`.
- A slice depends only on `research_agent.shared` and external libraries; slices never import from each other.

This means you can replace DSPy, change the prompt strategy, or swap the LM client without touching domain or application code.

## DSPy optimization

DSPy optimizes what it can see: signatures, programs, tools, and metrics. All of these live in the infrastructure layer, so optimization remains an infrastructure concern.

The optimization metric should still be defined in domain terms:

```python
def search_relevance_score(query: ResearchQuery, results: list[SearchResult]) -> float:
    return domain_evaluator.score(query, results)
```

The domain does not know that optimization happened; it only receives a configured `LanguageModel` instance.

### Optimization tooling location

The optimization pipeline is a sibling to datagen, not part of the runtime application:

- `src/datagen/` — generates the synthetic query training set (`queries_train.jsonl`).
- `src/optimize/` — DSPy optimization pipeline for the `SearchSuggestionNode`. It loads `queries_train.jsonl`, runs the node live against real indexes, and scores the returned `list[SearchResult]` with an embedding-similarity metric defined in domain terms. The metric, embedder, and dataset loader live here; the DSPy student program, signatures, and tools live in `research_agent.search.program` / `search.tools` once built.

Neither package is imported by `research_agent` at runtime.

## Memory

Memory is intentionally out of scope for the first version. When it is added:

- **Short-term memory** (session context) is passed as part of the node input.
- **Long-term memory** is exposed as a tool and/or injected into the prompt as metadata.

Neither case changes the domain port.

## Testing strategy

- **Domain tests** mock the `LanguageModel` port. They require no DSPy, no LM, and no network.
- **Application tests** mock the nodes they orchestrate.
- **Infrastructure tests** exercise DSPy signatures, programs, and adapters, optionally against a real or cached LM. Live tests that hit a real external index (e.g. the Semantic Scholar API) live under `tests/external/`, are tagged `live`, and are skipped by default.

## File structure

```text
src/research_agent/
├── __init__.py                 runtime shell; imports slices, wires workflows
├── shared/                     cross-slice kernel (contracts shared by >1 slice)
│   ├── __init__.py
│   └── ports.py                LanguageModel[InputT, OutputT] protocol (reserved)
├── search/                     THE search slice (SearchSuggestionNode); portable
│   ├── __init__.py             slice docstring: layer-tag rules, dependency direction
│   ├── models.py               [Layer: Domain] ResearchQuery, SearchResult, PaperInfo, PaperSource, SearchIndexReference, ...
│   ├── node.py                 [Layer: Domain service] SearchSuggestionNode (reserved)
│   ├── tools.py                [Layer: Infrastructure] LiteratureSearch (public) + private _SemanticScholarSearch / _ArXivSearch / _PubMedSearch / _CrossRefSearch
│   └── program.py              [Layer: Infrastructure] SearchSuggestionProgram (reserved)
├── workflows.py                [Layer: Application] workflow orchestration (reserved)
└── api/                        [Layer: Presentation] runtime entrypoints (reserved)

src/
├── datagen/                    synthetic query generation tooling (NOT runtime)
└── optimize/                   DSPy optimization pipeline for the search node (NOT runtime)

tests/
├── unit/                       deterministic tests (mirror slice layout)
└── external/                   live tests (tagged `live`, skipped by default)
```

Reserved-but-not-present files are documented for forward planning; do not stub them (YAGNI).

## Principles summary

1. The domain owns meaning and business rules.
2. The `LanguageModel` port is the only bridge between a slice's domain node and its DSPy-backed infrastructure.
3. Pydantic validates I/O contracts; it does not enforce domain invariants.
4. DSPy signatures, programs, tools, and optimizers live in infrastructure.
5. Workflows orchestrate; nodes reason and enforce rules.
6. Dependencies point inward; infrastructure is replaceable.
7. Slices are portable and isolated; cross-slice contracts live in `shared`.
