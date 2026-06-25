"""Infrastructure layer for the research agent.

Owns DSPy adapter, signatures, programs, tools, and LM clients. Nothing
outside this layer may import from it; the dependency arrow points inward
toward the domain.
"""
