"""Unit tests for the FastAPI presentation layer."""

from __future__ import annotations

from importlib.metadata import version
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from pydantic import HttpUrl

from research_agent.api.app import _package_version, create_app
from research_agent.app import SearchRun
from research_agent.search.models import PaperInfo

if TYPE_CHECKING:
    from collections.abc import Iterator

_ABSTRACT = (
    "A sufficiently long abstract describing the research methodology, "
    "experimental setup, results, and conclusions of this work in detail "
    "to satisfy the PaperInfo min_length=200 invariant enforced by Pydantic."
)


def _paper() -> PaperInfo:
    return PaperInfo(
        title="Paper Alpha On Quantum Computing Advances",
        abstract=_ABSTRACT,
        authors=("Alice",),
        url=HttpUrl("https://example.com/p"),
        open_access=False,
    )


@pytest.fixture
def paper_search_app() -> MagicMock:
    app = MagicMock()
    app.search = AsyncMock(
        return_value=SearchRun(
            papers=[_paper()],
            suggestion="try surveys next",
            trace_id="trace-42",
        ),
    )
    app.record_feedback = AsyncMock()
    return app


@pytest.fixture
def client(paper_search_app: MagicMock) -> Iterator[TestClient]:
    with TestClient(create_app(paper_search_app=paper_search_app)) as test_client:
        yield test_client


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_search_returns_papers_suggestion_and_trace_id(
    client: TestClient,
    paper_search_app: MagicMock,
) -> None:
    response = client.post(
        "/search",
        json={"text": "quantum computing", "domains": ["cs"]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["suggestion"] == "try surveys next"
    assert body["trace_id"] == "trace-42"
    assert len(body["papers"]) == 1
    assert body["papers"][0]["title"] == "Paper Alpha On Quantum Computing Advances"

    paper_search_app.search.assert_awaited_once()
    query = paper_search_app.search.await_args.args[0]
    assert query.text == "quantum computing"
    assert query.domains == ("cs",)


def test_search_rejects_short_text(client: TestClient) -> None:
    response = client.post("/search", json={"text": "ab"})
    assert response.status_code == 422


def test_feedback_records_thumbs(
    client: TestClient,
    paper_search_app: MagicMock,
) -> None:
    response = client.post(
        "/feedback",
        json={
            "trace_id": "trace-42",
            "useful": True,
            "comment": "strong first page",
        },
    )
    assert response.status_code == 204
    paper_search_app.record_feedback.assert_awaited_once_with(
        "trace-42",
        useful=True,
        comment="strong first page",
    )


def test_feedback_allows_omitted_comment(
    client: TestClient,
    paper_search_app: MagicMock,
) -> None:
    response = client.post(
        "/feedback",
        json={"trace_id": "trace-7", "useful": False},
    )
    assert response.status_code == 204
    paper_search_app.record_feedback.assert_awaited_once_with(
        "trace-7",
        useful=False,
        comment=None,
    )


def test_lifespan_flushes_opik_on_shutdown(paper_search_app: MagicMock) -> None:
    with (
        patch("research_agent.api.app.flush_opik_client") as flush,
        TestClient(create_app(paper_search_app=paper_search_app)),
    ):
        pass
    flush.assert_called_once_with()


def test_lifespan_tolerates_opik_flush_failure(paper_search_app: MagicMock) -> None:
    with (
        patch(
            "research_agent.api.app.flush_opik_client",
            side_effect=RuntimeError("opik down"),
        ) as flush,
        TestClient(create_app(paper_search_app=paper_search_app)),
    ):
        pass
    flush.assert_called_once_with()


def test_package_version_matches_installed() -> None:
    assert _package_version() == version("research-agent")


def test_create_app_uses_package_version() -> None:
    application = create_app(paper_search_app=MagicMock())
    assert application.version == _package_version()
