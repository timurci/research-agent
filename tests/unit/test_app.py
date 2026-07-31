"""Unit tests for the runtime composition root."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from pydantic import HttpUrl

from research_agent.app import PaperSearchApp, SearchRun, build_paper_search_app
from research_agent.search.models import PaperInfo, ResearchQuery
from research_agent.shared.config.lm import UnknownLMConfigRoleError

if TYPE_CHECKING:
    from pathlib import Path

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


def _query() -> ResearchQuery:
    return ResearchQuery(text="quantum computing", domains=("cs",))


def _trace_cm(*, trace_id: str = "trace-app-1") -> MagicMock:
    trace = MagicMock()
    trace.id = trace_id
    cm = MagicMock()
    cm.__enter__.return_value = trace
    cm.__exit__.return_value = None
    return cm


@pytest.mark.asyncio
async def test_search_returns_papers_suggestion_and_trace_id() -> None:
    paper = _paper()

    async def run_workflow(
        query: ResearchQuery,
    ) -> tuple[list[PaperInfo], str]:
        assert query.text == "quantum computing"
        return [paper], "try surveys next"

    app = PaperSearchApp(run_workflow=run_workflow, project_name="proj")
    with (
        patch(
            "research_agent.app.start_as_current_trace",
            return_value=_trace_cm(trace_id="tid-42"),
        ) as start_trace,
        patch("research_agent.app.run_async", new_callable=AsyncMock) as run_async,
    ):
        run = await app.search(_query())

    assert isinstance(run, SearchRun)
    assert run.papers == [paper]
    assert run.suggestion == "try surveys next"
    assert run.trace_id == "tid-42"
    start_trace.assert_called_once()
    kwargs = start_trace.call_args.kwargs
    assert kwargs["name"] == "paper_search"
    assert kwargs["project_name"] == "proj"
    assert kwargs["flush"] is False
    assert kwargs["input"] == {"text": "quantum computing", "domains": ["cs"]}
    run_async.assert_awaited()


@pytest.mark.asyncio
async def test_search_sets_trace_output_with_counts() -> None:
    paper = _paper()

    async def run_workflow(
        _query: ResearchQuery,
    ) -> tuple[list[PaperInfo], str]:
        return [paper, paper], "more work"

    app = PaperSearchApp(run_workflow=run_workflow)
    cm = _trace_cm()
    with (
        patch("research_agent.app.start_as_current_trace", return_value=cm),
        patch("research_agent.app.run_async", new_callable=AsyncMock),
    ):
        await app.search(_query())

    cm.__enter__.return_value.update.assert_called_once_with(
        output={"paper_count": 2, "suggestion": "more work"},
    )


@pytest.mark.asyncio
async def test_search_flushes_opik_after_trace_context() -> None:
    async def run_workflow(
        _query: ResearchQuery,
    ) -> tuple[list[PaperInfo], str]:
        return [], ""

    app = PaperSearchApp(run_workflow=run_workflow)
    with (
        patch(
            "research_agent.app.start_as_current_trace",
            return_value=_trace_cm(),
        ),
        patch("research_agent.app.run_async", new_callable=AsyncMock) as run_async,
        patch("research_agent.app.flush_opik_client") as flush,
    ):
        await app.search(_query())

    run_async.assert_awaited_once()
    assert run_async.await_args is not None
    assert run_async.await_args.args[0] is flush


@pytest.mark.asyncio
async def test_search_returns_run_when_opik_flush_fails() -> None:
    paper = _paper()

    async def run_workflow(
        _query: ResearchQuery,
    ) -> tuple[list[PaperInfo], str]:
        return [paper], "more work"

    app = PaperSearchApp(run_workflow=run_workflow)
    with (
        patch(
            "research_agent.app.start_as_current_trace",
            return_value=_trace_cm(trace_id="tid-99"),
        ),
        patch(
            "research_agent.app.run_async",
            new_callable=AsyncMock,
            side_effect=RuntimeError("opik down"),
        ) as run_async,
    ):
        run = await app.search(_query())

    run_async.assert_awaited_once()
    assert isinstance(run, SearchRun)
    assert run.papers == [paper]
    assert run.trace_id == "tid-99"


@pytest.mark.asyncio
async def test_record_feedback_offloads_to_observability() -> None:
    app = PaperSearchApp(
        run_workflow=MagicMock(),
        project_name="proj",
    )
    with (
        patch(
            "research_agent.app.run_async",
            new_callable=AsyncMock,
            side_effect=lambda func, *args: func(*args),
        ),
        patch("research_agent.app.record_user_feedback") as record,
    ):
        await app.record_feedback("tid-1", useful=True, comment="ok")
    record.assert_called_once_with(
        "tid-1",
        useful=True,
        comment="ok",
        project_name="proj",
    )


@pytest.mark.asyncio
async def test_record_feedback_propagates_observability_failure() -> None:
    app = PaperSearchApp(
        run_workflow=MagicMock(),
        project_name="proj",
    )
    with (
        patch(
            "research_agent.app.run_async",
            new_callable=AsyncMock,
            side_effect=lambda func, *args: func(*args),
        ),
        patch(
            "research_agent.app.record_user_feedback",
            side_effect=RuntimeError("opik unreachable"),
        ),
        pytest.raises(RuntimeError, match="opik unreachable"),
    ):
        await app.record_feedback("tid-1", useful=False)


def _write_lm_config(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "search-search": {
                    "model": "openai/test-search",
                    "api_key": "k",
                    "base_url": "http://search.example/v1",
                },
                "search-rerank": {
                    "model": "infinity/test-rerank",
                    "api_key": "k",
                    "base_url": "http://rerank.example/v1",
                },
                "search-suggest": {
                    "model": "openai/test-suggest",
                    "api_key": "k",
                    "base_url": "http://suggest.example/v1",
                },
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_build_paper_search_app_uses_fresh_agents_per_call(
    tmp_path: Path,
) -> None:
    lm_path = tmp_path / "lm.yaml"
    _write_lm_config(lm_path)
    instructions_path = tmp_path / "missing-instructions.yaml"

    sessions: list[object] = []
    agent_instances: list[object] = []
    suggestion_instances: list[object] = []

    class FakeSearchAgent:
        def __init__(
            self,
            _lm_config: object,
            session: object,
            _literature_search: object,
            *,
            instructions_path: Path | None = None,
        ) -> None:
            _ = instructions_path
            sessions.append(session)
            agent_instances.append(self)

        async def __call__(self, _data: ResearchQuery) -> list[PaperInfo]:
            return []

    class FakeReranker:
        def __init__(self, _cfg: object) -> None:
            pass

        async def __call__(
            self,
            data: tuple[ResearchQuery, list[PaperInfo]],
        ) -> list[PaperInfo]:
            return list(data[1])

    class FakeSuggestion:
        def __init__(
            self,
            _cfg: object,
            *,
            instructions_path: Path | None = None,
        ) -> None:
            _ = instructions_path
            suggestion_instances.append(self)

        async def __call__(
            self,
            _data: tuple[ResearchQuery, list[PaperInfo]],
        ) -> str:
            return ""

    with (
        patch("research_agent.app.SearchAgent", FakeSearchAgent),
        patch("research_agent.app.Reranker", FakeReranker),
        patch("research_agent.app.SuggestionGenerator", FakeSuggestion),
        patch("research_agent.app.LiteratureSearch"),
        patch(
            "research_agent.app.start_as_current_trace",
            side_effect=lambda **_k: _trace_cm(trace_id="t1"),
        ),
        patch("research_agent.app.run_async", new_callable=AsyncMock),
        patch("research_agent.app.configure_dspy_opik_callback"),
    ):
        app = build_paper_search_app(
            lm_config_path=lm_path,
            instructions_config_path=instructions_path,
            project_name="unit",
            configure_observability=True,
        )
        await app.search(_query())
        await app.search(_query())

    assert len(agent_instances) == 2
    assert len(suggestion_instances) == 2
    assert len(sessions) == 2
    assert sessions[0] is not sessions[1]
    assert suggestion_instances[0] is not suggestion_instances[1]


def test_build_paper_search_app_configures_observability_when_requested(
    tmp_path: Path,
) -> None:
    lm_path = tmp_path / "lm.yaml"
    _write_lm_config(lm_path)

    with (
        patch("research_agent.app.SearchAgent"),
        patch("research_agent.app.Reranker"),
        patch("research_agent.app.SuggestionGenerator"),
        patch("research_agent.app.LiteratureSearch"),
        patch("research_agent.app.configure_dspy_opik_callback") as configure,
    ):
        build_paper_search_app(
            lm_config_path=lm_path,
            instructions_config_path=tmp_path / "missing.yaml",
            project_name="unit",
            configure_observability=True,
        )

    configure.assert_called_once_with(project_name="unit")


def test_build_paper_search_app_skips_observability_when_disabled(
    tmp_path: Path,
) -> None:
    lm_path = tmp_path / "lm.yaml"
    _write_lm_config(lm_path)

    with (
        patch("research_agent.app.SearchAgent"),
        patch("research_agent.app.Reranker"),
        patch("research_agent.app.SuggestionGenerator"),
        patch("research_agent.app.LiteratureSearch"),
        patch("research_agent.app.configure_dspy_opik_callback") as configure,
    ):
        build_paper_search_app(
            lm_config_path=lm_path,
            instructions_config_path=tmp_path / "missing.yaml",
            configure_observability=False,
        )

    configure.assert_not_called()


def test_build_paper_search_app_raises_for_missing_lm_role(tmp_path: Path) -> None:
    lm_path = tmp_path / "lm.yaml"
    lm_path.write_text(
        yaml.safe_dump(
            {
                "search-search": {
                    "model": "openai/test-search",
                    "api_key": "k",
                    "base_url": "http://search.example/v1",
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(UnknownLMConfigRoleError, match="unknown LM config role"):
        build_paper_search_app(
            lm_config_path=lm_path,
            instructions_config_path=tmp_path / "missing.yaml",
            configure_observability=False,
        )


def test_build_paper_search_app_forwards_literature_api_keys(
    tmp_path: Path,
) -> None:
    lm_path = tmp_path / "lm.yaml"
    _write_lm_config(lm_path)

    with (
        patch("research_agent.app.SearchAgent"),
        patch("research_agent.app.Reranker"),
        patch("research_agent.app.SuggestionGenerator"),
        patch("research_agent.app.LiteratureSearch") as literature_cls,
        patch("research_agent.app.configure_dspy_opik_callback"),
    ):
        build_paper_search_app(
            lm_config_path=lm_path,
            instructions_config_path=tmp_path / "missing.yaml",
            configure_observability=False,
            pubmed_api_key="pubmed-key",
            openalex_api_key="openalex-key",
        )

    literature_cls.assert_called_once_with(
        pubmed_api_key="pubmed-key",
        openalex_api_key="openalex-key",
    )


def test_build_paper_search_app_includes_instruction_metadata(tmp_path: Path) -> None:
    lm_path = tmp_path / "lm.yaml"
    _write_lm_config(lm_path)
    program = tmp_path / "search-search.json"
    program.write_text("{}", encoding="utf-8")
    instructions = tmp_path / "instructions.yaml"
    instructions.write_text(
        yaml.safe_dump({"instructions": {"search-search": str(program)}}),
        encoding="utf-8",
    )

    with (
        patch("research_agent.app.SearchAgent"),
        patch("research_agent.app.Reranker"),
        patch("research_agent.app.SuggestionGenerator"),
        patch("research_agent.app.LiteratureSearch"),
        patch("research_agent.app.configure_dspy_opik_callback"),
    ):
        app = build_paper_search_app(
            lm_config_path=lm_path,
            instructions_config_path=instructions,
            configure_observability=False,
        )

    assert app._metadata is not None
    metadata = dict(app._metadata())
    assert metadata["search.instructions.path"] == str(program)
    assert "search.instructions.sha256" in metadata


@pytest.mark.asyncio
async def test_search_refreshes_instruction_sha_metadata(tmp_path: Path) -> None:
    lm_path = tmp_path / "lm.yaml"
    _write_lm_config(lm_path)
    program = tmp_path / "search-search.json"
    program.write_text("version-one", encoding="utf-8")
    instructions = tmp_path / "instructions.yaml"
    instructions.write_text(
        yaml.safe_dump({"instructions": {"search-search": str(program)}}),
        encoding="utf-8",
    )

    class FakeSearchAgent:
        def __init__(
            self,
            _lm_config: object,
            _session: object,
            _literature_search: object,
            *,
            instructions_path: Path | None = None,
        ) -> None:
            _ = instructions_path

        async def __call__(self, _data: ResearchQuery) -> list[PaperInfo]:
            return []

    class FakeReranker:
        def __init__(self, _cfg: object) -> None:
            pass

        async def __call__(
            self,
            data: tuple[ResearchQuery, list[PaperInfo]],
        ) -> list[PaperInfo]:
            return list(data[1])

    class FakeSuggestion:
        def __init__(
            self,
            _cfg: object,
            *,
            instructions_path: Path | None = None,
        ) -> None:
            _ = instructions_path

        async def __call__(
            self,
            _data: tuple[ResearchQuery, list[PaperInfo]],
        ) -> str:
            return ""

    with (
        patch("research_agent.app.SearchAgent", FakeSearchAgent),
        patch("research_agent.app.Reranker", FakeReranker),
        patch("research_agent.app.SuggestionGenerator", FakeSuggestion),
        patch("research_agent.app.LiteratureSearch"),
        patch(
            "research_agent.app.start_as_current_trace",
            side_effect=lambda **_k: _trace_cm(trace_id="t1"),
        ) as start_trace,
        patch("research_agent.app.run_async", new_callable=AsyncMock),
        patch("research_agent.app.configure_dspy_opik_callback"),
    ):
        app = build_paper_search_app(
            lm_config_path=lm_path,
            instructions_config_path=instructions,
            configure_observability=False,
        )
        await app.search(_query())
        program.write_text("version-two", encoding="utf-8")
        await app.search(_query())

    first_sha = start_trace.call_args_list[0].kwargs["metadata"][
        "search.instructions.sha256"
    ]
    second_sha = start_trace.call_args_list[1].kwargs["metadata"][
        "search.instructions.sha256"
    ]
    assert first_sha != second_sha
