"""Unit tests for ``datagen.main``.

The CLI is exercised end-to-end via ``monkeypatch``-ed ``sys.argv`` and
``sys.stderr``. Real LLM calls are stubbed out at the ``litellm.acompletion``
boundary so the tests stay deterministic.
"""

from __future__ import annotations

import inspect
import io
import json
import sys
from contextlib import redirect_stderr
from typing import TYPE_CHECKING, Any, ClassVar

import litellm
import pytest
from litellm.types.utils import Choices, Message, ModelResponse

from datagen import main as datagen_main
from datagen.config import GenerationConfig
from datagen.llm_client import LLMClient

if TYPE_CHECKING:
    from pathlib import Path


class _FakeCompletion:
    """Stand-in for ``litellm.acompletion`` that records its call kwargs."""

    calls: ClassVar[list[dict[str, object]]] = []

    def __init__(self) -> None:
        pass

    async def __call__(self, **kwargs: object) -> ModelResponse:
        type(self).calls.append(kwargs)
        return ModelResponse(
            choices=[Choices(message=Message(content="[]"))],
        )


@pytest.fixture
def fake_lm(monkeypatch: pytest.MonkeyPatch) -> type[_FakeCompletion]:
    _FakeCompletion.calls.clear()
    monkeypatch.setattr(litellm, "acompletion", _FakeCompletion())
    return _FakeCompletion


def _run_cli(
    monkeypatch: pytest.MonkeyPatch,
    *argv: str,
    out_dir: Path,
) -> None:
    monkeypatch.setattr(sys, "argv", ["datagen.main", *argv])
    datagen_main.main()
    assert out_dir.exists(), f"expected output dir at {out_dir}"


def test_extra_body_json_is_parsed_and_forwarded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_lm: type[_FakeCompletion],
) -> None:
    """A valid ``--extra-body`` JSON object reaches ``acompletion`` as a dict."""
    body: dict[str, Any] = {
        "provider": {"order": ["openai", "together"], "allow_fallbacks": False}
    }
    _run_cli(
        monkeypatch,
        "--model",
        "openrouter/openai/gpt-4o-mini",
        "--api-key",
        "k",
        "--out-dir",
        str(tmp_path),
        "--extra-body",
        json.dumps(body),
        out_dir=tmp_path,
    )

    assert fake_lm is _FakeCompletion
    assert len(_FakeCompletion.calls) >= 1
    assert _FakeCompletion.calls[0]["extra_body"] == body
    assert _FakeCompletion.calls[0]["reasoning_effort"] is None


def test_reasoning_effort_is_forwarded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_lm: type[_FakeCompletion],
) -> None:
    """``--reasoning-effort`` is forwarded to ``litellm.acompletion`` verbatim."""
    _run_cli(
        monkeypatch,
        "--model",
        "openai/o3-mini",
        "--api-key",
        "k",
        "--out-dir",
        str(tmp_path),
        "--reasoning-effort",
        "high",
        out_dir=tmp_path,
    )

    assert fake_lm is _FakeCompletion
    assert _FakeCompletion.calls[0]["reasoning_effort"] == "high"
    assert _FakeCompletion.calls[0]["extra_body"] is None


def test_extra_body_defaults_to_none_when_omitted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_lm: type[_FakeCompletion],
) -> None:
    """Omitting ``--extra-body`` leaves the ``acompletion`` kwarg as ``None``."""
    _run_cli(
        monkeypatch,
        "--model",
        "openai/gpt-4o-mini",
        "--api-key",
        "k",
        "--out-dir",
        str(tmp_path),
        out_dir=tmp_path,
    )

    assert fake_lm is _FakeCompletion
    assert _FakeCompletion.calls[0]["extra_body"] is None
    assert _FakeCompletion.calls[0]["reasoning_effort"] is None


def test_extra_body_invalid_json_exits_with_code_2(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Malformed JSON for ``--extra-body`` exits with code 2 and a clear message."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "datagen.main",
            "--model",
            "openai/gpt-4o-mini",
            "--api-key",
            "k",
            "--out-dir",
            str(tmp_path),
            "--extra-body",
            "not json",
        ],
    )
    stderr = io.StringIO()
    with redirect_stderr(stderr), pytest.raises(SystemExit) as excinfo:
        datagen_main.main()
    assert excinfo.value.code == 2
    assert "--extra-body must be valid JSON" in stderr.getvalue()


def test_extra_body_non_object_exits_with_code_2(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A JSON value that is not an object exits with code 2."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "datagen.main",
            "--model",
            "openai/gpt-4o-mini",
            "--api-key",
            "k",
            "--out-dir",
            str(tmp_path),
            "--extra-body",
            "[]",
        ],
    )
    stderr = io.StringIO()
    with redirect_stderr(stderr), pytest.raises(SystemExit) as excinfo:
        datagen_main.main()
    assert excinfo.value.code == 2
    assert "must decode to a JSON object" in stderr.getvalue()


def test_generation_config_default_fields_present() -> None:
    """``GenerationConfig`` exposes the new optional fields with ``None`` defaults."""
    config = GenerationConfig(llm_model="openai/gpt-4o-mini", api_key="k")
    assert config.reasoning_effort is None
    assert config.extra_body is None


def test_llm_client_accepts_keyword_only_kwargs() -> None:
    """``LLMClient`` requires ``reasoning_effort``/``extra_body`` to be keyword-only."""
    sig = inspect.signature(LLMClient.__init__)
    assert sig.parameters["reasoning_effort"].kind is inspect.Parameter.KEYWORD_ONLY
    assert sig.parameters["extra_body"].kind is inspect.Parameter.KEYWORD_ONLY
