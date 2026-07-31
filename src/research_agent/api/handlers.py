# Copyright (c) 2026 Timur Çakmakoğlu

"""FastAPI exception handlers.

Layer: Presentation.

Maps upstream rate-limit failures (HTTP 429) to a JSON 429 response with
the upstream ``Retry-After`` header; every other error is re-raised and
keeps Starlette's default 500 behavior.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from fastapi.responses import JSONResponse

if TYPE_CHECKING:
    from fastapi import FastAPI, Request

_RATE_LIMIT_STATUS_CODE: int = 429
_RATE_LIMIT_DETAIL: str = "rate limited"


def _retry_after_header(exc: Exception) -> str | None:
    """Return the upstream ``Retry-After`` header value, if any.

    Reads the header from either the exception's own ``headers`` mapping
    or a carried ``response``/``raw_response`` object, matching the error
    shapes raised by litellm and the OpenRouter SDK.
    """
    direct_headers = getattr(exc, "headers", None)
    if isinstance(direct_headers, Mapping):
        value = direct_headers.get("retry-after")
        if value is not None:
            return str(value)
    for attr in ("response", "raw_response"):
        response = getattr(exc, attr, None)
        response_headers = getattr(response, "headers", None)
        if response_headers is not None:
            value = response_headers.get("retry-after")
            if value is not None:
                return str(value)
    return None


async def handle_unhandled_exception(
    _request: Request,
    exc: Exception,
) -> JSONResponse:
    """Map upstream rate-limit errors to 429; re-raise everything else."""
    if getattr(exc, "status_code", None) == _RATE_LIMIT_STATUS_CODE:
        headers: dict[str, str] = {}
        retry_after = _retry_after_header(exc)
        if retry_after is not None:
            headers["Retry-After"] = retry_after
        return JSONResponse(
            status_code=_RATE_LIMIT_STATUS_CODE,
            content={"detail": _RATE_LIMIT_DETAIL},
            headers=headers,
        )
    raise exc


def register_exception_handlers(application: FastAPI) -> None:
    """Register the catch-all handler on *application*.

    The handler is the ``Exception``-keyed ServerErrorMiddleware handler,
    so FastAPI always re-raises the original error after its response is
    sent, keeping server-side error logging intact.
    """
    application.exception_handler(Exception)(handle_unhandled_exception)
