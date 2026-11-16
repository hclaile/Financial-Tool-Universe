"""News connector — financial news aggregation."""

from __future__ import annotations

import os
from typing import Any

import httpx

from .base import BaseConnector, register_connector

# Supports multiple news backends; default uses NewsAPI.org
_NEWSAPI_URL = "https://newsapi.org/v2"


class NewsConnector(BaseConnector):
    name = "news_api"
    actions = frozenset({"headlines", "search"})

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        # Resolve API key: explicit api_key > token_env env var > NEWS_API_KEY fallback
        api_key = self._config.get("api_key", "")
        if not api_key:
            token_env = self._config.get("token_env", "NEWS_API_KEY")
            api_key = os.getenv(token_env, "")
        self._api_key = api_key

    async def fetch(self, action: str, **kwargs: Any) -> Any:
        dispatch = {
            "headlines": self._headlines,
            "search": self._search,
        }
        handler = dispatch.get(action)
        if handler is None:
            raise ValueError(f"News: unsupported action '{action}'")
        return await handler(**kwargs)

    async def _headlines(
        self,
        query: str | None = None,
        symbols: list[str] | None = None,
        limit: int = 20,
        **_: Any,
    ) -> list[dict]:
        if not self._api_key:
            return self._fallback_headlines(query, symbols, limit)

        params: dict[str, Any] = {
            "apiKey": self._api_key,
            "pageSize": str(min(limit, 100)),
        }
        if query:
            params["q"] = query
        if symbols:
            params["q"] = " ".join(symbols)

        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(f"{_NEWSAPI_URL}/everything", params=params)
            resp.raise_for_status()
            data = resp.json()

        articles = data.get("articles", [])
        return [
            {
                "title": a.get("title", ""),
                "source": a.get("source", {}).get("name", ""),
                "published_at": a.get("publishedAt", ""),
                "url": a.get("url", ""),
                "symbols": symbols or [],
            }
            for a in articles[:limit]
        ]

    def _fallback_headlines(
        self,
        query: str | None = None,
        symbols: list[str] | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """Return empty list when no API key is configured."""
        return []

    async def _search(self, query: str, limit: int = 10, **_: Any) -> list[dict]:
        headlines = await self._headlines(query=query, limit=limit)
        return [
            {
                "query": query,
                "source": "news_api",
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
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    f"{_NEWSAPI_URL}/top-headlines",
                    params={"apiKey": self._api_key, "country": "us", "pageSize": "1"},
                )
                return resp.status_code == 200
        except Exception:
            return False


register_connector("news_api", NewsConnector)
