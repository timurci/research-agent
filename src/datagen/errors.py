"""Project-specific exceptions for the datagen pipeline."""


class LLMContractError(Exception):
    """Raised when the LLM returns data violating its expected contract."""
