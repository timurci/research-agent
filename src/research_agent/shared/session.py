"""Session-scoped working memory.

Holds key/value session state for adapter-local data (e.g. search hit
lists). Multi-turn message history is reserved for a later iteration
and is not implemented here (YAGNI). Distinct from a domain repository:
not aggregate persistence and not workflow-orchestrated.

The ``Session`` protocol is an application-layer port. ``InMemorySession``
is the default infrastructure adapter. ``ScopedSession`` delegates to a
per-thread override when active so concurrent callers (GEPA eval
threads, parallel tool runs) sharing one long-lived session-backed
object each see their own state without rebuilding it. All three live
in this shared module because the kernel is small and a second file
would not earn its keep.
"""

from __future__ import annotations

import copy
import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterator


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


class ScopedSession(Session):
    """Session that delegates to a per-thread override when active.

    Wraps a base session and forwards every call to it, unless the
    current thread has activated an override via ``use``. The override is
    a thread-local: each thread that calls ``use`` gets its own isolated
    session, and other threads continue to see the base. This lets one
    long-lived session-backed object (e.g. a ``dspy.ReAct`` tool) serve
    concurrent calls without rebuilding it or sharing state.

    ``ContextVar`` would not propagate to worker threads spawned by
    ``ThreadPoolExecutor`` (GEPA's evaluation model), so a
    ``threading.local`` is the minimal correct primitive here.
    """

    def __init__(self, base: Session) -> None:
        """Wrap *base*; per-thread overrides become active only via ``use``.

        Args:
            base: Session seen by threads without an active override.
        """
        self._base = base
        self._local = threading.local()

    def __deepcopy__(self, memo: dict[int, object]) -> ScopedSession:
        """Deep-copy with a deep-copied base and a fresh thread-local slot.

        The base session is deep-copied so the copy starts independent of
        the original's state. The per-thread ``threading.local`` is
        replaced with a fresh one because per-thread overrides are
        meaningless across copies and ``threading.local`` itself cannot
        be deep-copied.
        """
        new = ScopedSession(copy.deepcopy(self._base, memo))
        memo[id(self)] = new
        new.__dict__["_local"] = threading.local()
        return new

    @property
    def _override(self) -> Session | None:
        return getattr(self._local, "session", None)

    @contextmanager
    def use(self, session: Session) -> Iterator[None]:
        """Run a block with *session* as the per-thread active session.

        Restores the prior thread-local state on exit so pooled worker
        threads do not carry stale overrides into later tasks.

        Args:
            session: Session to use for the duration of the block.
        """
        prev = self._override
        self._local.session = session
        try:
            yield
        finally:
            if prev is None:
                del self._local.session
            else:
                self._local.session = prev

    @property
    def active_session(self) -> Session:
        """Per-thread override if set, else the base session."""
        override = self._override
        return override if override is not None else self._base

    def get(self, key: str, default: object | None = None) -> object | None:
        """Return the value for *key* from the active session."""
        return self.active_session.get(key, default)

    def set(self, key: str, value: object) -> None:
        """Store *value* under *key* in the active session."""
        self.active_session.set(key, value)

    def delete(self, key: str) -> None:
        """Remove *key* in the active session; no-op when missing."""
        self.active_session.delete(key)
