"""Ownership tools — institutional holders, mutual fund holders, insider trading, 13F."""

from __future__ import annotations

from ..connectors.base import fetch_with_failover


def register(mcp) -> None:

    @mcp.tool()
    async def ownership_get_institutional(
        symbol: str,
        region: str = "US",
        start_date: str | None = None,
        end_date: str | None = None,
        as_of_date: str | None = None,
    ) -> list[dict]:
        return await fetch_with_failover(
            "institutional_holders",
            "institutional_holders",
            region=region,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            as_of_date=as_of_date,
        )

    @mcp.tool()
    async def ownership_get_mutual_fund(
        symbol: str,
        region: str = "US",
        start_date: str | None = None,
        end_date: str | None = None,
        as_of_date: str | None = None,
    ) -> list[dict]:
        return await fetch_with_failover(
            "mutual_fund_holders",
            "mutual_fund_holders",
            region=region,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            as_of_date=as_of_date,
        )

    @mcp.tool()
    async def ownership_get_insider(
        symbol: str,
        limit: int = 10,
        region: str = "US",
        start_date: str | None = None,
        end_date: str | None = None,
        as_of_date: str | None = None,
    ) -> list[dict]:
        return await fetch_with_failover(
            "insider_trading",
            "insider_trading",
            region=region,
            symbol=symbol,
            limit=limit,
            start_date=start_date,
            end_date=end_date,
            as_of_date=as_of_date,
        )

    @mcp.tool()
    async def ownership_get_13f(
        cik_id: str,
        date: str | None = None,
        region: str = "US",
    ) -> list[dict]:
        return await fetch_with_failover(
            "form_13f",
            "form_13f",
            region=region,
            cik_id=cik_id,
            date=date,
        )
