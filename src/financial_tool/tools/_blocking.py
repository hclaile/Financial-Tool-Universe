"""Run blocking tool implementations without corrupting the MCP stdio stream."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import sys
import threading
from collections.abc import Callable
from typing import Any


_STDOUT_LOCK = threading.Lock()


async def run_blocking(operation: Callable[[], Any], *, name: str, output_path: str | None = None) -> dict:
    """Execute a local operation; the registered tool owns its arguments and behavior."""

    def execute() -> dict:
        # Some optional libraries print during import, even before their run() call.
        with _STDOUT_LOCK, contextlib.redirect_stdout(sys.stderr):
            try:
                result = operation()
                if inspect.isawaitable(result):
                    result = asyncio.run(result)
            except ImportError as exc:
                raise RuntimeError(f"{name} needs an optional dependency: {exc}") from exc
        if result is None:
            return {"status": "completed", "tool": name, "output_path": output_path}
        normalized = json.loads(json.dumps(result, ensure_ascii=False, default=str))
        return normalized if isinstance(normalized, dict) else {"result": normalized}

    return await asyncio.to_thread(execute)
