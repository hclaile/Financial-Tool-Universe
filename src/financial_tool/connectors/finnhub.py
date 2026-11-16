"""Finnhub connector — Company News and News Sentiment."""

from __future__ import annotations

import os
from typing import Any

import httpx

from .base import BaseConnector, register_connector
from ._numeric import safe_float, safe_int

_BASE_URL = "https://finnhub.io/api/v1"


class FinnhubConnector(BaseConnector):
    name = "finnhub"
    actions = frozenset({"headlines", "company_news", "sentiment", "search"})

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        # Environment values let operators rotate a credential without editing YAML.
        token_env = self._config.get("token_env", "FINNHUB_API_KEY")
        api_key = os.getenv(token_env, "") or self._config.get("api_key", "")
        self._api_key = api_key

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        params = params or {}
        params["token"] = self._api_key
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(f"{_BASE_URL}{path}", params=params)
            resp.raise_for_status()
            return resp.json()

    async def fetch(self, action: str, **kwargs: Any) -> Any:
        dispatch = {
            "headlines": self._headlines,
            "company_news": self._company_news,
            "sentiment": self._sentiment,
            "search": self._search,
        }
        handler = dispatch.get(action)
        if handler is None:
            raise ValueError(f"Finnhub: unsupported action '{action}'")
        return await handler(**kwargs)

    # ── Company News ──────────────────────────────────────

    async def _company_news(
        self,
        symbols: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
        limit: int = 20,
        **_: Any,
    ) -> list[dict]:
        """Company News from Finnhub.

        Endpoint: GET /company-news
        Required: symbol, from (YYYY-MM-DD), to (YYYY-MM-DD)
        Returns list of news articles for the given symbol and date range.
        """
        if not self._api_key:
            return []

        from datetime import datetime, timedelta

        # Finnhub requires from/to dates; default to last 7 days if not provided
        if not end:
            end = datetime.utcnow().strftime("%Y-%m-%d")
        if not start:
            start = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")

        # Finnhub company-news accepts a single symbol
        symbol = symbols[0] if symbols else ""

        params: dict[str, Any] = {
            "symbol": symbol,
            "from": start,
            "to": end,
        }
        data = await self._get("/company-news", params)

        if not isinstance(data, list):
            return []

        return [
            {
                "title": a.get("headline", ""),
                "source": a.get("source", ""),
                "published_at": datetime.utcfromtimestamp(a.get("datetime", 0)).strftime("%Y-%m-%dT%H:%M:%SZ")
                if a.get("datetime")
                else "",
                "url": a.get("url", ""),
                "summary": a.get("summary", ""),
                "symbols": [symbol] if symbol else [],
                "image": a.get("image", ""),
                "category": a.get("category", ""),
                "id": a.get("id", ""),
            }
            for a in data[:limit]
        ]

    async def _headlines(
        self,
        query: str | None = None,
        symbols: list[str] | None = None,
        limit: int = 20,
        **_: Any,
    ) -> list[dict]:
        """Headlines via Company News — thin wrapper for compatibility with news tool interface."""
        return await self._company_news(
            symbols=symbols,
            limit=limit,
        )

    # ── News Sentiment ────────────────────────────────────

    async def _sentiment(
        self,
        symbol: str,
        **_: Any,
    ) -> dict:
        """News Sentiment from Finnhub.

        Endpoint: GET /news-sentiment
        Required: symbol
        Returns: buzz (weekly/daily article counts), sentiment (bearish/bullish %),
                 company news sentiment score, and sector average sentiment.
        """
        if not self._api_key:
            return {}

        data = await self._get("/news-sentiment", {"symbol": symbol})

        if not data or "buzz" not in data:
            return {}

        buzz = data.get("buzz", {})
        sentiment = data.get("sentiment", {})

        return {
            "symbol": symbol,
            "source": "finnhub",
            "data_type": "sentiment",
            "buzz": {
                "articles_in_last_week": safe_int(buzz.get("articlesInLastWeek")),
                "articles_in_last_day": safe_int(buzz.get("articlesInLastDay")),
                "articles_last_week_vs_week_before": safe_float(buzz.get("articlesInLastWeekVsWeekBefore")),
                "weekly_average": safe_float(buzz.get("weeklyAverage")),
            },
            "sentiment": {
                "bearish_percent": safe_float(sentiment.get("bearishPercent")),
                "bullish_percent": safe_float(sentiment.get("bullishPercent")),
            },
            "company_news_score": safe_float(data.get("companyNewsScore")),
            "sector_average_news_score": safe_float(data.get("sectorAverageNewsScore")),
            "sector_average_bullish_percent": safe_float(data.get("sectorAverageBullishPercent")),
            "sector_average_bearish_percent": safe_float(data.get("sectorAverageBearishPercent")),
        }

    async def _search(self, query: str, limit: int = 10, **_: Any) -> list[dict]:
        headlines = await self._headlines(query=query, limit=limit)
        return [
            {
                "query": query,
                "source": "finnhub",
                "data_type": "news",
                "relevance": 1.0,
                "payload": h,
            }
            for h in headlines
        ]

    async def health_check(self) -> bool:
        if not self._api_key:
            return False
        try:
            # Use a lightweight endpoint to check connectivity
            data = await self._get("/news-sentiment", {"symbol": "AAPL"})
            return isinstance(data, dict)
        except Exception:
            return False


register_connector("finnhub", FinnhubConnector)
