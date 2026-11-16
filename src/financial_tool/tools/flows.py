"""Capital flow tools — money flow and market cap analytics."""

from __future__ import annotations

from ..connectors.base import fetch_with_failover


def register(mcp) -> None:

    @mcp.tool()
    async def flows_get_money_flow(
        symbol: str,
        period: str = "1d",
        region: str = "CN",
    ) -> dict:
        return await fetch_with_failover(
            "money_flow", "money_flow", region=region, symbol=symbol, period=period,
        )

    @mcp.tool()
    async def flows_get_market_cap(
        symbol: str,
        limit: int = 10,
        region: str = "US",
    ) -> dict | list[dict]:
        if limit > 1:
            return await fetch_with_failover(
                "market_cap",
                "market_cap_historical",
                region=region,
                symbol=symbol,
                limit=limit,
            )
        return await fetch_with_failover(
            "market_cap",
            "market_cap",
            region=region,
            symbol=symbol,
        )
