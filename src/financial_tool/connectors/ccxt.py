"""CCXT connector — unified crypto exchange library."""

from __future__ import annotations

from typing import Any

import ccxt

from .base import BaseConnector, register_connector
from ._numeric import safe_float


class CCXTConnector(BaseConnector):
    name = "ccxt"
    actions = frozenset({"price", "quote", "ohlcv"})

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._exchange_id = self._config.get("exchange", "binance")
        self._exchange = self._init_exchange()

    def _init_exchange(self) -> ccxt.Exchange:
        exchange_class = getattr(ccxt, self._exchange_id, None)
        if exchange_class is None:
            raise ValueError(f"CCXT: unknown exchange '{self._exchange_id}'")
        opts: dict[str, Any] = {"enableRateLimit": True}
        if self._config.get("api_key"):
            opts["apiKey"] = self._config["api_key"]
        if self._config.get("secret"):
            opts["secret"] = self._config["secret"]
        return exchange_class(opts)

    async def fetch(self, action: str, **kwargs: Any) -> Any:
        dispatch = {
            "price": self._price,
            "quote": self._ticker,
            "ohlcv": self._ohlcv,
        }
        handler = dispatch.get(action)
        if handler is None:
            raise ValueError(f"CCXT: unsupported action '{action}'")
        return await handler(**kwargs)

    async def _price(self, symbol: str, vs_currency: str = "USDT", **_: Any) -> dict:
        pair = f"{symbol}/{vs_currency}"
        ticker = self._exchange.fetch_ticker(pair)
        return {
            "symbol": symbol,
            "price": safe_float(ticker.get("last")),
            "volume_24h": safe_float(ticker.get("quoteVolume")),
            "change_24h": safe_float(ticker.get("change")),
            "timestamp": str(ticker.get("timestamp", "")),
        }

    async def _ticker(self, symbol: str, vs_currency: str = "USDT", **_: Any) -> dict:
        return await self._price(symbol=symbol, vs_currency=vs_currency)

    async def _ohlcv(
        self,
        symbol: str,
        vs_currency: str = "USDT",
        start_date: str | None = None,
        end_date: str | None = None,
        interval: str = "1d",
        **_: Any,
    ) -> list[dict]:
        pair = f"{symbol}/{vs_currency}"
        timeframe_map = {"1d": "1d", "1w": "1w", "1M": "1M"}
        timeframe = timeframe_map.get(interval, "1d")
        since = None
        if start_date:
            import datetime
            since = int(datetime.datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
        ohlcv = self._exchange.fetch_ohlcv(pair, timeframe=timeframe, since=since, limit=500)
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
            for candle in ohlcv
        ]

    async def health_check(self) -> bool:
        try:
            self._exchange.fetch_ticker("BTC/USDT")
            return True
        except Exception:
            return False


register_connector("ccxt", CCXTConnector)
