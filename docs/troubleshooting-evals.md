# Troubleshooting Eval Deadlocks

## Problem

`uv run -m evals.main` freezes after a few traces complete. Traces are stuck in
"pending" state with no requests reaching the LLM server. The symptom is
intermittent: 5-6 traces finish, then the remaining 10+ freeze.

**Root cause.** `litellm`'s `GLOBAL_LOGGING_WORKER` singleton binds to whichever
thread's event loop it first sees. MLflow's eval harness runs `predict_fn` in a
`ThreadPoolExecutor` where each worker calls `asyncio.run()`, creating a new
temporary event loop per call. When a worker's loop exits, orphaned logging
coroutines trigger the `RuntimeWarning: coroutine ... was never awaited` message
and freeze subsequent traces.

`mlflow.dspy.autolog()` compounds the issue: it registers a single
`MlflowCallback` instance into `dspy.settings.callbacks`, which is shared across
all worker threads. Under concurrent `asyncio.run()`, the callback's span
tracking (`_call_id_to_span`, `_call_id_to_module`) and the
`InMemoryTraceManager._lock` cause lock contention that stalls trace creation.

## Workarounds

### Option A: Disable `mlflow.dspy.autolog()` (fast, fewer spans)

Comment out the `mlflow.dspy.autolog()` call in `src/evals/main.py`. The
`@mlflow.trace` decorator on `predict_fn` still produces root-level traces; only
the per-DSPy-call spans (LM calls, Tool calls, ReAct steps) are lost.

```python
# src/evals/main.py, inside main()
# mlflow.dspy.autolog()  # disabled — see docs/troubleshooting-evals.md
```

**Trade-off:** Full eval speed; no per-DSPy-call span hierarchy in the MLflow
UI.

### Option B: Reduce max workers (slow, full tracing)

Keep `mlflow.dspy.autolog()` and set a low concurrency ceiling:

```bash
MLFLOW_GENAI_EVAL_MAX_WORKERS=2 uv run -m evals.main --experiment my-exp search-search
```

Or permanently in `src/evals/main.py`, change the default from `10` to `2` in
`_apply_mlflow_eval_env_defaults()`.

**Trade-off:** Full dspy autologging with per-call spans; eval runs 3-5x
slower.

## Recommendation

Use Option A for routine CI runs where speed matters. Switch to Option B when
debugging agent behaviour and you need the full DSPy call-tree in MLflow traces.
