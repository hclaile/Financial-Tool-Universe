"""Alpaca News connector — historical stock and crypto news articles."""

from __future__ import annotations

import os
from typing import Any

import httpx

from .base import BaseConnector, register_connector

_ALPACA_NEWS_URL = "https://data.alpaca.markets/v1beta1/news"


class AlpacaNewsConnector(BaseConnector):
    name = "alpaca_news"
    actions = frozenset({"articles", "headlines"})

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._api_key_id = (
            self._config.get("api_key_id", "")
            or self._config.get("key_id", "")
            or os.getenv("ALPACA_API_KEY_ID", "")
            or os.getenv("APCA_API_KEY_ID", "")
        )
        self._api_secret_key = (
            self._config.get("api_secret_key", "")
            or self._config.get("secret_key", "")
            or os.getenv("ALPACA_API_SECRET_KEY", "")
            or os.getenv("APCA_API_SECRET_KEY", "")
        )

    async def fetch(self, action: str, **kwargs: Any) -> Any:
        dispatch = {
            "articles": self._articles,
            "headlines": self._headlines,
        }
        handler = dispatch.get(action)
        if handler is None:
            raise ValueError(f"AlpacaNews: unsupported action '{action}'")
        return await handler(**kwargs)

    async def _articles(
        self,
        symbols: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
        sort: str = "desc",
        limit: int = 10,
        include_content: bool = False,
        exclude_contentless: bool = False,
        page_token: str | None = None,
        **_: Any,
    ) -> dict:
        if not self._api_key_id or not self._api_secret_key:
            return {"articles": [], "next_page_token": None}

        params: dict[str, str] = {
            "sort": sort,
            "limit": str(max(1, min(limit, 50))),
        }
        if symbols:
            params["symbols"] = ",".join(symbols)
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        if include_content:
            params["include_content"] = "true"
        if exclude_contentless:
            params["exclude_contentless"] = "true"
        if page_token:
            params["page_token"] = page_token

        headers = {
            "APCA-API-KEY-ID": self._api_key_id,
            "APCA-API-SECRET-KEY": self._api_secret_key,
        }
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(_ALPACA_NEWS_URL, params=params, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        return {
            "articles": [self._normalize_article(item) for item in data.get("news", [])],
            "next_page_token": data.get("next_page_token"),
        }

    async def _headlines(self, **kwargs: Any) -> list[dict]:
        result = await self._articles(**kwargs)
        return result["articles"]

    def _normalize_article(self, article: dict[str, Any]) -> dict:
        return {
            "id": article.get("id"),
            "title": article.get("headline", ""),
            "summary": article.get("summary", ""),
            "author": article.get("author", ""),
            "published_at": article.get("created_at", ""),
            "updated_at": article.get("updated_at", ""),
            "content": article.get("content", ""),
            "url": article.get("url", ""),
            "symbols": article.get("symbols", []),
            "source": article.get("source", ""),
        }

    async def health_check(self) -> bool:
        if not self._api_key_id or not self._api_secret_key:
            return False
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    _ALPACA_NEWS_URL,
                    params={"limit": "1"},
                    headers={
                        "APCA-API-KEY-ID": self._api_key_id,
                        "APCA-API-SECRET-KEY": self._api_secret_key,
                    },
                )
                return resp.status_code == 200
        except Exception:
            return False


register_connector("alpaca_news", AlpacaNewsConnector)
