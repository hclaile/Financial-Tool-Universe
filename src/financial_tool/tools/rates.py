"""Rates tools — swap pricing via external MCP."""

from __future__ import annotations

from typing import Any

from ._external import call_external_tool


def register(mcp) -> None:

    @mcp.tool()
    async def rates_price_swap(
        currency: str,
        tenor: str,
        index: str | None = None,
        notional: float | None = None,
    ) -> Any:
        return await call_external_tool(
            "lseg",
            "ir_swap",
            currency=currency,
            tenor=tenor,
            index=index,
            notional=notional,
        )
