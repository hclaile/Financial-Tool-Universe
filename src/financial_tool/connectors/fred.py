"""FRED connector — Federal Reserve Economic Data."""

from __future__ import annotations

import os
import logging
from typing import Any

import httpx

from .base import BaseConnector, register_connector
from ._numeric import safe_float

_BASE_URL = "https://api.stlouisfed.org/fred"
logger = logging.getLogger("financial_tool.connector.fred")


class FREDConnector(BaseConnector):
    name = "fred"
    actions = frozenset({"series", "search", "search_note", "search_code"})

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._api_key = self._config.get("api_key", "") or os.getenv("FRED_API_KEY", "")

    async def _get(self, path: str, params: dict | None = None) -> Any:
        params = params or {}
        params["api_key"] = self._api_key
        params["file_type"] = "json"
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(f"{_BASE_URL}{path}", params=params)
            if resp.status_code == 200:
                return resp.json()
            # 400 = invalid series_id, future dates beyond available data,
            # or malformed request parameters.  All are deterministic —
            # retrying won't fix them.  Return a structured non-retryable
            # result so failover can try the next source without retry delays.
            # 422 = similarly deterministic "unprocessable" responses.
            if resp.status_code in (400, 422):
                logger.debug(
                    "FRED %s returned %s: %s",
                    path, resp.status_code, resp.text[:120],
                )
                return {
                    "status": "invalid_request",
                    "source": "fred",
                    "http_status": resp.status_code,
                    "reason": "fred_invalid_series_or_parameters",
                }
            resp.raise_for_status()
            return resp.json()

    async def fetch(self, action: str, **kwargs: Any) -> Any:
        dispatch = {
            "series": self._series,
            "search": self._search,
            "search_note": self._search_note,
            "search_code": self._search,
        }
        handler = dispatch.get(action)
        if handler is None:
            raise ValueError(f"FRED: unsupported action '{action}'")
        return await handler(**kwargs)

    # ── Series ───────────────────────────────────────────

    async def _series(
        self,
        series_id: str,
        start_date: str | None = None,
        end_date: str | None = None,
        **_: Any,
    ) -> Any:
        params: dict[str, Any] = {"series_id": series_id, "sort_order": "asc"}
        if start_date:
            params["observation_start"] = start_date
        if end_date:
            params["observation_end"] = end_date
        data = await self._get("/series/observations", params)
        if isinstance(data, dict) and data.get("status") == "invalid_request":
            return data
        observations = data.get("observations", [])
        return [
            {
                "series_id": series_id,
                "date": obs.get("date", ""),
                "value": safe_float(obs.get("value")),
            }
            for obs in observations
            if obs.get("value", ".") != "."
        ]

    # ── Search ───────────────────────────────────────────

    async def _search(self, query: str, limit: int = 10, **_: Any) -> list[dict]:
        data = await self._get("/series/search", {"search_text": query, "limit": str(limit)})
        results = data.get("seriess", [])
        return [
            {
                "query": query,
                "source": "fred",
                "data_type": "macro",
                "relevance": 1.0,
                "payload": {
                    "series_id": r.get("id", ""),
                    "title": r.get("title", ""),
                    "frequency": r.get("frequency", ""),
                },
            }
            for r in results[:limit]
        ]

    async def _search_note(self, query: str, limit: int = 10, **_: Any) -> dict:
        """Return the matching FRED series metadata for the legacy note mode."""
        results = await self._search(query=query, limit=limit)
        for result in results:
            payload = result.get("payload", {})
            if str(payload.get("series_id", "")).casefold() == query.casefold():
                return payload
        return {"error": "Series not found", "query": query}

    async def health_check(self) -> bool:
        if not self._api_key:
            return False
        try:
            await self._get("/series", {"series_id": "GDP"})
            return True
        except Exception:
            return False


register_connector("fred", FREDConnector)
