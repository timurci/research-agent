.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help message
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / { printf "  %-15s %s\n", $$1, $$2 }' $(MAKEFILE_LIST) | sort

.PHONY: eval eval-list serve

HOST ?= 0.0.0.0
PORT ?= 8000

eval: ## Run the evals harness
	@uv run --env-file .env -m evals.main $(ARGS)

eval-list: ## List available eval suites
	@uv run -m evals.main --list

serve: ## Serve the FastAPI backend (HOST and PORT overridable)
	@uv run --env-file .env uvicorn research_agent.api.app:app --host $(HOST) --port $(PORT)
