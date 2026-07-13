"""Root pytest configuration.

Registers the ``live`` marker and skips live-tagged tests by default so
``uv run pytest`` never attempts network calls. Run live tests explicitly:

    uv run pytest -m live

Also disables DSPy on-disk LM caching so independent test runs do not reuse
responses from ``~/.dspy_cache``.
"""

from __future__ import annotations

import dspy
import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register the ``live`` marker."""
    config.addinivalue_line(
        "markers",
        "live: hits an external network service; skipped unless `-m live` is passed",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Skip ``live`` tests unless the ``-m`` expression selects them."""
    markexpr = config.getoption("-m", default="") or ""
    if "live" in markexpr:
        return
    skip_live = pytest.mark.skip(reason="live test; run with `pytest -m live`")
    for item in items:
        if item.get_closest_marker("live"):
            item.add_marker(skip_live)


@pytest.fixture(scope="session", autouse=True)
def _disable_dspy_disk_cache() -> None:
    """Prevent LM responses from persisting across independent test runs."""
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=True)
