.PHONY: eval eval-list

eval:
	@(set -a && source ./.env && uv run -m evals.main $(ARGS))

eval-list:
	@uv run -m evals.main --list
