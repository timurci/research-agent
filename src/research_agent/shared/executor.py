"""Shared executor pool for offloading sync operations to worker threads.

Layer: Infrastructure.

Slices that wrap synchronous SDKs use the module-level executor to run
blocking calls without stalling the async event loop.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

_EXECUTOR = ThreadPoolExecutor()


async def run_async[T](
    func: Callable[..., T],
    *args: object,
) -> T:
    """Offload a blocking call to a worker thread and await the result.

    Args:
        func: A synchronous, blocking callable (typically an SDK method).
        *args: Positional arguments forwarded to *func*.

    Returns:
        The return value of *func* once it has been computed in a worker
        thread.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_EXECUTOR, func, *args)
