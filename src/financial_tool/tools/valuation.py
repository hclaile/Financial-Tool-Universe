"""Valuation tools — DCF, price target consensus, WACC."""

from __future__ import annotations

from ..connectors.base import fetch_with_failover


def register(mcp) -> None:

    @mcp.tool()
    async def valuation_get_dcf(
        symbol: str,
        period: str = "annual",
        limit: int = 10,
        region: str = "US",
        start_date: str | None = None,
        end_date: str | None = None,
        as_of_date: str | None = None,
    ) -> dict | list[dict]:
        if period == "ttm" or limit == 1:
            return await fetch_with_failover(
                "dcf_valuation",
                "dcf",
                region=region,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                as_of_date=as_of_date,
            )
        data = await fetch_with_failover(
            "dcf_valuation",
            "dcf_historical",
            region=region,
            symbol=symbol,
            period=period,
            limit=limit,
            start_date=start_date,
            end_date=end_date,
            as_of_date=as_of_date,
        )
        if not data:
            return await fetch_with_failover(
                "dcf_valuation",
                "dcf",
                region=region,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                as_of_date=as_of_date,
            )
        return data

    @mcp.tool()
    async def valuation_get_price_target(
        symbol: str,
        region: str = "US",
        start_date: str | None = None,
        end_date: str | None = None,
        as_of_date: str | None = None,
    ) -> list[dict]:
        return await fetch_with_failover(
            "price_target",
            "price_target",
            region=region,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            as_of_date=as_of_date,
        )

    @mcp.tool()
    async def valuation_get_wacc(
        symbol: str,
        period: str = "annual",
        limit: int = 10,
        region: str = "US",
        start_date: str | None = None,
        end_date: str | None = None,
        as_of_date: str | None = None,
    ) -> list[dict]:
        return await fetch_with_failover(
            "wacc",
            "wacc",
            region=region,
            symbol=symbol,
            period=period,
            limit=limit,
            start_date=start_date,
            end_date=end_date,
            as_of_date=as_of_date,
        )
