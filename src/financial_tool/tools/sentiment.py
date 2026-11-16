"""Sentiment tools — news sentiment analysis for companies."""

from __future__ import annotations

from ..connectors.base import fetch_with_failover


def register(mcp) -> None:

    @mcp.tool()
    async def sentiment_get_news_sentiment(
        symbol: str,
        region: str = "US",
    ) -> dict:
        """Get news sentiment analysis for a company.

        Finnhub returns buzz volume, bullish/bearish percentages, company
        score, and sector averages. Alpha Vantage is a lower-fidelity
        article-sentiment fallback and returns only fields it actually has.
        This tool is currently available for US-listed company coverage.
        """
        return await fetch_with_failover(
            "news_sentiment",
            "sentiment",
            region=region,
            symbol=symbol,
        )
