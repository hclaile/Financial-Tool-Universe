"""Helpers for proxying provider-specific tools to configured external MCP servers."""

from __future__ import annotations

from typing import Any

from ..connectors.base import expand_temporal_limit_kwargs, get_source_connector, guard_temporal_result


async def call_external_tool(server: str, tool_name: str, **kwargs: Any) -> Any:
    conn = get_source_connector("external_mcp")
    request_kwargs = expand_temporal_limit_kwargs(kwargs)
    result = await conn.fetch(tool_name, server=server, **request_kwargs)
    return guard_temporal_result(
        result,
        source_name=f"external_mcp:{server}",
        action=tool_name,
        **kwargs,
    )
