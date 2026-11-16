"""Tavily connector — AI-optimized web search with synthesized answers."""

from __future__ import annotations

import hashlib
import json
import os
import time
from threading import RLock
from typing import Any

import httpx

from .base import BaseConnector, register_connector
from ._numeric import safe_float

_TAVILY_URL = "https://api.tavily.com"
_CACHE_TTL_SECONDS = 300.0


class TavilyConnector(BaseConnector):
    name = "tavily"
    actions = frozenset({"web_search"})
    _cache: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}
    _cache_lock = RLock()

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._api_key = self._config.get("api_key", "") or os.getenv("TAVILY_API_KEY", "")

    async def _post(self, path: str, payload: dict) -> Any:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{_TAVILY_URL}{path}",
                json=dict(payload),
                headers=headers,
            )
            resp.raise_for_status()
            return resp.json()

    async def fetch(self, action: str, **kwargs: Any) -> Any:
        dispatch = {
            "web_search": self._web_search,
        }
        handler = dispatch.get(action)
        if handler is None:
            raise ValueError(f"Tavily: unsupported action '{action}'")
        return await handler(**kwargs)

    async def _web_search(
        self,
        query: str,
        search_depth: str = "advanced",
        max_results: int = 5,
        include_answer: bool = True,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
        **_: Any,
    ) -> dict:
        payload: dict[str, Any] = {
            "query": query,
            "search_depth": search_depth,
            "max_results": max_results,
            "include_answer": include_answer,
        }
        if include_domains:
            payload["include_domains"] = include_domains
        if exclude_domains:
            payload["exclude_domains"] = exclude_domains

        cache_key = self._cache_key(payload)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        data = await self._post("/search", payload)

        results = []
        for r in data.get("results", []):
            results.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", ""),
                "score": safe_float(r.get("score")),
            })

        result = {
            "query": query,
            "answer": data.get("answer", ""),
            "results": results,
            "source": "tavily",
        }
        self._store_cached(cache_key, result)
        return result

    @classmethod
    def clear_cache(cls) -> None:
        """Clear process-local search responses; primarily useful for tests."""

        with cls._cache_lock:
            cls._cache.clear()

    def _cache_key(self, payload: dict[str, Any]) -> tuple[str, str]:
        credential_fingerprint = hashlib.sha256(
            self._api_key.encode("utf-8")
        ).hexdigest()[:16]
        normalized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return credential_fingerprint, normalized

    @classmethod
    def _get_cached(cls, key: tuple[str, str]) -> dict[str, Any] | None:
        now = time.monotonic()
        with cls._cache_lock:
            cached = cls._cache.get(key)
            if cached is None:
                return None
            expires_at, result = cached
            if expires_at <= now:
                cls._cache.pop(key, None)
                return None
            return dict(result)

    @classmethod
    def _store_cached(cls, key: tuple[str, str], result: dict[str, Any]) -> None:
        with cls._cache_lock:
            cls._cache[key] = (time.monotonic() + _CACHE_TTL_SECONDS, dict(result))

    async def health_check(self) -> bool:
        if not self._api_key:
            return False
        try:
            data = await self._post("/search", {
                "query": "test",
                "max_results": 1,
                "search_depth": "basic",
            })
            return isinstance(data, dict)
        except Exception:
            return False


register_connector("tavily", TavilyConnector)
