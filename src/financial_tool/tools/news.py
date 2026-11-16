"""News tools — headlines, articles, and company news from multiple sources."""

from __future__ import annotations

from ..connectors.base import (
    expand_temporal_limit_kwargs,
    fetch_with_failover,
    get_source_connector,
    guard_temporal_result,
)
from ..errors import UnsupportedRegionError


def register(mcp) -> None:

    @mcp.tool()
    async def news_get_headlines(
        query: str | None = None,
        symbols: list[str] | None = None,
        limit: int = 20,
        region: str = "US",
        start: str | None = None,
        end: str | None = None,
        as_of_date: str | None = None,
    ) -> list[dict]:
        """Get financial news headlines from the highest-priority source.

        Sources are tried in order: Alpha Vantage → Finnhub → NewsAPI → Alpaca.
        """
        return await fetch_with_failover(
            "news_list",
            "headlines",
            region=region,
            query=query,
            symbols=symbols,
            limit=limit,
            start=start,
            end=end,
            as_of_date=as_of_date,
        )

    @mcp.tool()
    async def news_get_alpaca_articles(
        symbols: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
        sort: str = "desc",
        limit: int = 10,
        include_content: bool = False,
        exclude_contentless: bool = False,
        page_token: str | None = None,
        region: str = "US",
    ) -> dict:
        if region.upper() != "US":
            raise UnsupportedRegionError("news_list", region.upper())
        conn = get_source_connector("alpaca_news")
        request = expand_temporal_limit_kwargs({"limit": limit, "start": start, "end": end})
        result = await conn.fetch(
            "articles",
            symbols=symbols,
            start=start,
            end=end,
            sort=sort,
            limit=request["limit"],
            include_content=include_content,
            exclude_contentless=exclude_contentless,
            page_token=page_token,
        )
        return guard_temporal_result(
            result,
            source_name="alpaca_news",
            action="articles",
            symbols=symbols,
            start=start,
            end=end,
        )

    @mcp.tool()
    async def news_get_finnhub_news(
        symbols: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
        limit: int = 20,
        region: str = "US",
    ) -> list[dict]:
        """Get company news from Finnhub by symbol and date range.

        Finnhub requires at least one symbol and a date range.
        Defaults to the last 7 days if dates are not provided.
        """
        if region.upper() != "US":
            raise UnsupportedRegionError("news_list", region.upper())
        conn = get_source_connector("finnhub")
        request = expand_temporal_limit_kwargs({"limit": limit, "start": start, "end": end})
        result = await conn.fetch(
            "company_news",
            symbols=symbols,
            start=start,
            end=end,
            limit=request["limit"],
        )
        return guard_temporal_result(
            result,
            source_name="finnhub",
            action="company_news",
            symbols=symbols,
            start=start,
            end=end,
        )

    @mcp.tool()
    async def news_get_alphavantage_news(
        query: str | None = None,
        symbols: list[str] | None = None,
        limit: int = 20,
        region: str = "US",
        start: str | None = None,
        end: str | None = None,
        as_of_date: str | None = None,
    ) -> list[dict]:
        """Get financial news with sentiment scores from Alpha Vantage.

        Alpha Vantage's NEWS_SENTIMENT endpoint returns news articles
        with per-article and per-ticker sentiment scores.
        """
        if region.upper() != "US":
            raise UnsupportedRegionError("news_list", region.upper())
        conn = get_source_connector("alphavantage")
        request = expand_temporal_limit_kwargs(
            {"limit": limit, "start": start, "end": end, "as_of_date": as_of_date}
        )
        result = await conn.fetch(
            "headlines",
            query=query,
            symbols=symbols,
            limit=request["limit"],
            start=start,
            end=end,
            as_of_date=as_of_date,
        )
        return guard_temporal_result(
            result,
            source_name="alphavantage",
            action="headlines",
            query=query,
            symbols=symbols,
            limit=limit,
            start=start,
            end=end,
            as_of_date=as_of_date,
        )
