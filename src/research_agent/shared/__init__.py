"""Shared kernel for the research agent.

Home for cross-slice contracts used by more than one capability slice.
Nothing here is slice-specific; slices depend on this package, never the
reverse.

- ``rerank.py`` — the provider-dispatch rerank client seam
  (litellm + OpenRouter SDK).

Reserved locations (not yet present):

- ``ports.py`` — the generic ``LanguageModel[InputT, OutputT]`` protocol,
  the only bridge between a slice's domain node and its DSPy-backed
  infrastructure.
"""
