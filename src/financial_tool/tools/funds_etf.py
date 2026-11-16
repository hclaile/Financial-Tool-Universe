"""Funds & ETF tools — info, holdings, sector/country weightings."""

from __future__ import annotations

from ..connectors.base import fetch_with_failover


def register(mcp) -> None:

    @mcp.tool()
    async def funds_etf_get_info(
        symbol: str,
        include_holdings: bool = False,
        region: str = "CN",
    ) -> dict:
        return await fetch_with_failover(
            "fund_info",
            "info",
            region=region,
            symbol=symbol,
            include_holdings=include_holdings,
        )

    @mcp.tool()
    async def funds_etf_get_sector_weightings(
        symbol: str,
        region: str = "US",
    ) -> list[dict]:
        return await fetch_with_failover(
            "etf_weightings",
            "etf_sector_weightings",
            region=region,
            symbol=symbol,
        )

    @mcp.tool()
    async def funds_etf_get_country_weightings(
        symbol: str,
        region: str = "US",
    ) -> list[dict]:
        return await fetch_with_failover(
            "etf_weightings",
            "etf_country_weightings",
            region=region,
            symbol=symbol,
        )
