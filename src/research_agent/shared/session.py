"""Session-scoped working memory.

Layer: Application (port) + Infrastructure (default adapter).

Holds key/value session state for adapter-local data (e.g. search hit
lists). Multi-turn message history is reserved for a later iteration
and is not implemented here (YAGNI). Distinct from a domain repository:
not aggregate persistence and not workflow-orchestrated.

The ``Session`` protocol is an application-layer port. ``InMemorySession``
is the default infrastructure adapter. Thread-isolated adapters
(``ScopedSession``) live in ``research_agent.shared.scoped_session``.
"""

from __future__ import annotations

from typing import Protocol


class MissingSessionKeyError(Exception):
    """Raised when a session key is not present."""

    def __init__(self, key: str) -> None:
        """Initialize with the missing key."""
        self.key = key
        super().__init__(f"missing session key: {key!r}")


class InvalidSessionStateError(Exception):
    """Raised when a session value has an unexpected shape or type."""


class Session(Protocol):
    """Session keeps track of conversation state."""

    def get(self, key: str, default: object | None = None) -> object | None:
        """Return the value for *key* or None if the key is not present."""
        # Note: Use a sentinel value for missing keys if you switch to Python 3.15+
        ...

    def set(self, key: str, value: object) -> None:
        """Store *value* under *key*."""
        ...

    def delete(self, key: str) -> None:
        """Remove *key* if present; no-op when missing."""
        ...


class InMemorySession(Session):
    """Session implementation backed by an in-memory dict."""

    def __init__(self) -> None:
        """Initialize empty session state."""
        self._state: dict[str, object] = {}

    def get(self, key: str, default: object | None = None) -> object | None:
        """Return the value for *key*."""
        return self._state.get(key, default)

    def set(self, key: str, value: object) -> None:
        """Store *value* under *key*."""
        self._state[key] = value

    def delete(self, key: str) -> None:
        """Remove *key* if present; no-op when missing."""
        self._state.pop(key, None)
