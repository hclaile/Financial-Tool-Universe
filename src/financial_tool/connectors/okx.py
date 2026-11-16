"""OKX connector — cryptocurrency exchange data via REST API."""

from __future__ import annotations

import hmac
import hashlib
import base64
import time
from typing import Any

import httpx

from .base import BaseConnector, register_connector
from ._numeric import safe_float

_BASE_URL = "https://www.okx.com/api/v5"


class OKXConnector(BaseConnector):
    name = "okx"
    actions = frozenset({"price", "quote", "ohlcv"})

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._api_key = self._config.get("api_key", "")
        self._secret = self._config.get("secret", "")
        self._passphrase = self._config.get("passphrase", "")

    def _sign(self, timestamp: str, method: str, path: str, body: str = "") -> str:
        msg = timestamp + method + path + body
        mac = hmac.new(self._secret.encode(), msg.encode(), hashlib.sha256)
        return base64.b64encode(mac.digest()).decode()

    def _headers(self, method: str, path: str, body: str = "") -> dict[str, str]:
        ts = str(int(time.time()))
        return {
            "OK-ACCESS-KEY": self._api_key,
            "OK-ACCESS-SIGN": self._sign(ts, method, path, body),
            "OK-ACCESS-TIMESTAMP": ts,
            "OK-ACCESS-PASSPHRASE": self._passphrase,
            "Content-Type": "application/json",
        }

    async def _get(self, path: str, params: dict | None = None) -> Any:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(f"{_BASE_URL}{path}", params=params)
            resp.raise_for_status()
            data = resp.json()
        if data.get("code") != "0":
            raise RuntimeError(f"OKX error: {data.get('msg', 'unknown')}")
        return data.get("data", [])

    # ── Public endpoints (no auth needed) ────────────────

    async def fetch(self, action: str, **kwargs: Any) -> Any:
        dispatch = {
            "price": self._price,
            "quote": self._ticker,
            "ohlcv": self._candles,
        }
        handler = dispatch.get(action)
        if handler is None:
            raise ValueError(f"OKX: unsupported action '{action}'")
        return await handler(**kwargs)

    async def _price(self, symbol: str, vs_currency: str = "USDT", **_: Any) -> dict:
        inst_id = f"{symbol}-{vs_currency}"
        data = await self._get("/market/ticker", {"instId": inst_id})
        if not data:
            return {"symbol": symbol, "price": 0, "volume_24h": 0, "change_24h": 0, "timestamp": ""}
        t = data[0]
        return {
            "symbol": symbol,
            "price": safe_float(t.get("last")),
            "volume_24h": safe_float(t.get("vol24h")),
            "change_24h": safe_float(t.get("last")) - safe_float(t.get("open24h")) if t.get("open24h") else 0,
            "timestamp": t.get("ts", ""),
        }

    async def _ticker(self, symbol: str, **_: Any) -> dict:
        return await self._price(symbol=symbol)

    async def _candles(
        self,
        symbol: str,
        vs_currency: str = "USDT",
        start_date: str | None = None,
        end_date: str | None = None,
        interval: str = "1d",
        **_: Any,
    ) -> list[dict]:
        inst_id = f"{symbol}-{vs_currency}"
        bar_map = {"1d": "1D", "1w": "1W", "1M": "1M"}
        params = {"instId": inst_id, "bar": bar_map.get(interval, "1D")}
        if start_date:
            params["after"] = start_date
        if end_date:
            params["before"] = end_date
        data = await self._get("/market/candles", params)
        # OKX returns [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
        return [
            {
                "symbol": symbol,
                "date": str(candle[0]),
                "open": safe_float(candle[1]),
                "high": safe_float(candle[2]),
                "low": safe_float(candle[3]),
                "close": safe_float(candle[4]),
                "volume": safe_float(candle[5]),
            }
            for candle in data[:100]
        ]

    async def health_check(self) -> bool:
        try:
            data = await self._get("/market/ticker", {"instId": "BTC-USDT"})
            return len(data) > 0
        except Exception:
            return False


register_connector("okx", OKXConnector)
