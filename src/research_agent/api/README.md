# src/research_agent/api

FastAPI presentation layer for the research-agent backend (HTTP only).

Routes call the runtime composition root (`PaperSearchApp` from
`research_agent.app`); no business logic lives here. The server runs the
module-level `app` instance created by `research_agent.api.app`.

## Layout

| File | Responsibility | Key exports |
|---|---|---|
| `app.py` | Application factory: lifespan, env-driven config, CORS, exception handlers. | `app`, `create_app` |
| `routes.py` | HTTP route handlers. | `router` |
| `deps.py` | FastAPI dependency exposing the `PaperSearchApp` facade from app state. | `get_paper_search_app`, `PaperSearchFacade` |
| `schemas.py` | HTTP request and response models. | `SearchResponse`, `FeedbackBody`, `HealthResponse` |
| `handlers.py` | Exception handlers: upstream rate limits (HTTP 429) to JSON 429 with `Retry-After`; everything else re-raised (default 500). | `register_exception_handlers` |

## How to run

From the repository root (requires `config/lm.yaml`; see the top-level
README):

```bash
make serve
```

or equivalently:

```bash
uv run uvicorn research_agent.api.app:app --host 0.0.0.0 --port 8000
```

Configuration comes from the environment at startup:

| Env var | Purpose | Default |
|---|---|---|
| `RESEARCH_AGENT_LM_CONFIG` | Path to the YAML LM config. | `config/lm.yaml` |
| `RESEARCH_AGENT_INSTRUCTIONS_CONFIG` | Path to the YAML optimized-instructions map. | `config/instructions.yaml` |
| `OPIK_PROJECT_NAME` | Opik project for traces and feedback. | — |
| `PUBMED_API_KEY` | PubMed API key for elevated rate limits. | — |
| `OPENALEX_API_KEY` | OpenAlex API key. | — |

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness probe. |
| `POST` | `/search` | Search for papers from a research question. |
| `POST` | `/feedback` | Thumbs feedback on a previous search (can be updated by sending again). |

## Notes

- CORS is limited to `https://timurci.github.io`.
- Upstream rate-limit failures surface as HTTP 429 with the upstream
  `Retry-After` header when present.
- FastAPI serves the OpenAPI schema at `/docs` and `/openapi.json`.

## Relationship to the runtime

- Presentation only: imports the composition root (`research_agent.app`)
  and search-slice models for request/response bodies.
- The runtime never imports `research_agent.api`; `research_agent.app`
  does not import the presentation layer.
