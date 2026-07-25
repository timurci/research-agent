"""Unit tests for Session / InMemorySession / ScopedSession."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from research_agent.shared.session import InMemorySession, ScopedSession


def test_set_get_round_trip() -> None:
    session = InMemorySession()
    session.set("k", [1, 2])
    assert session.get("k") == [1, 2]


def test_get_missing_returns_none() -> None:
    session = InMemorySession()
    assert session.get("missing") is None


def test_get_missing_with_default() -> None:
    session = InMemorySession()
    assert session.get("missing", []) == []
    assert session.get("missing", "fallback") == "fallback"


def test_delete_existing_and_missing() -> None:
    session = InMemorySession()
    session.set("k", "v")
    session.delete("k")
    assert session.get("k") is None
    session.delete("k")


def test_scoped_defaults_to_base() -> None:
    base = InMemorySession()
    base.set("k", "base")
    scoped = ScopedSession(base)
    assert scoped.get("k") == "base"
    scoped.set("k", "scoped")
    assert base.get("k") == "scoped"


def test_scoped_use_swaps_per_thread_and_restores() -> None:
    base = InMemorySession()
    base.set("k", "base")
    scoped = ScopedSession(base)

    fresh = InMemorySession()
    with scoped.use(fresh):
        fresh.set("k", "override")
        assert scoped.get("k") == "override"
        assert base.get("k") == "base"
    assert scoped.get("k") == "base"


def test_scoped_use_does_not_leak_to_other_threads() -> None:
    base = InMemorySession()
    base.set("k", "base")
    scoped = ScopedSession(base)

    def worker() -> str:
        return str(scoped.get("k"))

    with scoped.use(InMemorySession()):
        scoped.set("k", "override")
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(worker)
            assert future.result() == "base"


def test_scoped_isolates_across_threads() -> None:
    base = InMemorySession()
    scoped = ScopedSession(base)

    def worker(value: str) -> str:
        with scoped.use(InMemorySession()):
            scoped.set("k", value)
            return str(scoped.get("k"))

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(worker, v) for v in ["a", "b", "c", "d"]]
        results = [f.result() for f in futures]
    assert results == ["a", "b", "c", "d"]
    assert base.get("k") is None


def test_scoped_nested_use_restores_outer() -> None:
    base = InMemorySession()
    base.set("k", "base")
    scoped = ScopedSession(base)

    outer = InMemorySession()
    outer.set("k", "outer")
    inner = InMemorySession()
    inner.set("k", "inner")

    with scoped.use(outer):
        assert scoped.get("k") == "outer"
        with scoped.use(inner):
            assert scoped.get("k") == "inner"
        assert scoped.get("k") == "outer"
    assert scoped.get("k") == "base"
