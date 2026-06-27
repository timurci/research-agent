"""Unit tests for ``datagen.llm_client.LLMClient``.

The real ``litellm.acompletion`` function talks to a provider; tests stub
it out with a recorder so we can assert that ``reasoning_effort`` and
``extra_body`` flow through unchanged, and that ``.complete()`` still
returns the first response.
"""

from __future__ import annotations

from typing import Any, ClassVar

import litellm
import pytest
from litellm.types.utils import Choices, Message, ModelResponse

from datagen.llm_client import LLMClient


class _FakeCompletion:
    """Stand-in for ``litellm.acompletion`` that records its call kwargs."""

    call_kwargs: ClassVar[list[dict[str, object]]] = []

    def __init__(self) -> None:
        pass

    async def __call__(self, **kwargs: object) -> ModelResponse:
        type(self).call_kwargs.append(kwargs)
        return ModelResponse(
            choices=[Choices(message=Message(content="first response"))],
        )


def _install_fake_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> type[_FakeCompletion]:
    _FakeCompletion.call_kwargs.clear()
    monkeypatch.setattr(litellm, "acompletion", _FakeCompletion())
    return _FakeCompletion


@pytest.mark.asyncio
async def test_constructor_forwards_reasoning_effort_and_extra_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``reasoning_effort`` and ``extra_body`` reach ``acompletion`` verbatim."""
    _install_fake_completion(monkeypatch)
    body: dict[str, Any] = {
        "provider": {"order": ["openai", "together"], "allow_fallbacks": False}
    }

    client = LLMClient(
        "openrouter/openai/gpt-4o-mini",
        "key",
        reasoning_effort="medium",
        extra_body=body,
    )
    await client.complete("hello")

    assert len(_FakeCompletion.call_kwargs) == 1
    recorded = _FakeCompletion.call_kwargs[0]
    assert recorded["api_key"] == "key"
    assert recorded["reasoning_effort"] == "medium"
    assert recorded["extra_body"] == body


@pytest.mark.asyncio
async def test_constructor_defaults_are_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No reasoning or extra-body kwargs are injected when omitted."""
    _install_fake_completion(monkeypatch)

    client = LLMClient("openai/gpt-4o-mini", "key")
    await client.complete("hello")

    assert len(_FakeCompletion.call_kwargs) == 1
    recorded = _FakeCompletion.call_kwargs[0]
    assert recorded["api_key"] == "key"
    assert recorded["reasoning_effort"] is None
    assert recorded["extra_body"] is None


@pytest.mark.asyncio
async def test_complete_returns_first_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``.complete()`` forwards the prompt and returns the first response."""
    _install_fake_completion(monkeypatch)
    client = LLMClient("openai/gpt-4o-mini", "key")

    result = await client.complete("hello")

    assert result == "first response"
    assert _FakeCompletion.call_kwargs[0]["messages"] == [
        {"role": "user", "content": "hello"}
    ]
