"""Fixed income tools — LSEG/YieldBook analytics via external MCP."""

from __future__ import annotations

from typing import Any

from ._external import call_external_tool


def register(mcp) -> None:

    @mcp.tool()
    async def fixed_income_price_bond(
        identifier: str,
        identifier_type: str = "auto",
        price: float | None = None,
        yield_rate: float | None = None,
    ) -> Any:
        return await call_external_tool(
            "lseg",
            "bond_price",
            identifier=identifier,
            identifier_type=identifier_type,
            price=price,
            yield_rate=yield_rate,
        )

    @mcp.tool()
    async def fixed_income_price_bond_future(future_ric: str) -> Any:
        return await call_external_tool("lseg", "bond_future_price", future_ric=future_ric)

    @mcp.tool()
    async def fixed_income_get_bond_reference(identifier: str) -> Any:
        return await call_external_tool("lseg", "yieldbook_bond_reference", identifier=identifier)

    @mcp.tool()
    async def fixed_income_get_cashflow(identifier: str) -> Any:
        return await call_external_tool("lseg", "yieldbook_cashflow", identifier=identifier)

    @mcp.tool()
    async def fixed_income_run_scenario(
        identifier: str,
        rate_shifts_bp: list[int] | None = None,
    ) -> Any:
        shifts = rate_shifts_bp or [-100, -50, 0, 50, 100]
        return await call_external_tool(
            "lseg",
            "yieldbook_scenario",
            identifier=identifier,
            rate_shifts_bp=shifts,
        )

    @mcp.tool()
    async def fixed_income_get_risk_analytics(identifier: str) -> Any:
        return await call_external_tool("lseg", "fixed_income_risk_analytics", identifier=identifier)
