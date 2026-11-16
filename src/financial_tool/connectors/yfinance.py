"""YFinance connector — global equity market data."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import requests

from .base import BaseConnector, register_connector
from ._numeric import safe_float, safe_int
from ..symbols import classify_instrument
from ..network_config import DEFAULT_USER_AGENT

logger = logging.getLogger("financial_tool.connector.yfinance")

# ── Symbol normalization ──────────────────────────────────────
# Yahoo Finance uses different ticker formats for different asset
# classes.  Slash-format forex pairs (EUR/USD), commodity codes
# (XAUUSD), and FMP-style codes (GCUSD) are NOT valid Yahoo
# tickers — they must be mapped before yf.Ticker() is called.
#
# Common mapping rules:
#   Forex pairs (EUR/USD → EURUSD=X)
#   Commodities (XAU/USD → GC=F, XAG/USD → SI=F)
#   Crypto (BTC/USD → BTC-USD, ETH/USD → ETH-USD)
#   FMP codes (GCUSD → GC=F, SIUSD → SI=F)

_YF_SYMBOL_MAP: dict[str, str] = {
    # Gold
    "XAU/USD": "GC=F",
    "XAUUSD": "GC=F",
    "GOLD": "GC=F",
    "GLD-USD": "GLD",
    # Silver
    "XAG/USD": "SI=F",
    "XAGUSD": "SI=F",
    # Forex major pairs (Yahoo uses =X suffix)
    "EUR/USD": "EURUSD=X",
    "EURUSD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USD/JPY": "USDJPY=X",
    "USDJPY": "USDJPY=X",
    "USD/CNY": "USDCNY=X",
    "USDCNY": "USDCNY=X",
    "USD/EUR": "USDEUR=X",
    "USD/GBP": "USDGBP=X",
    "AUD/USD": "AUDUSD=X",
    "AUDUSD": "AUDUSD=X",
    "NZD/USD": "NZDUSD=X",
    "NZDUSD": "NZDUSD=X",
    "USD/CAD": "USDCAD=X",
    "USDCAD": "USDCAD=X",
    "USD/CHF": "USDCHF=X",
    "USDCHF": "USDCHF=X",
    # Crypto (Yahoo uses dash format)
    "BTC/USD": "BTC-USD",
    "ETH/USD": "ETH-USD",
    "SOL/USD": "SOL-USD",
    # FMP-style commodity codes (GCUSD etc.)
    "GCUSD": "GC=F",
    "SIUSD": "SI=F",
    "BRENT": "BZ=F",
    "BRENTUSD": "BZ=F",
    "BRENTCRUDE": "BZ=F",
    "BRENTCRUDEOIL": "BZ=F",
    "WTI": "CL=F",
    "WTIUSD": "CL=F",
    "CRUDEOILWTI": "CL=F",
    # Bloomberg index aliases ($ prefix → Yahoo ^ prefix)
    # Bloomberg terminal uses $ for indices; Yahoo Finance uses ^
    "$SOX": "^SOX",    # Philadelphia Semiconductor Index
    "$SPX": "^GSPC",   # S&P 500 Index
    "$NDX": "^NDX",    # NASDAQ 100 Index
    "$DJI": "^DJI",    # Dow Jones Industrial Average
    "$RUT": "^RUT",    # Russell 2000
    "$VIX": "^VIX",    # CBOE Volatility Index
    "$DX-Y-NYB": "DX-Y.NYB",  # ICE US Dollar Index
    "DX-Y-NYB": "DX-Y.NYB",
    "VIX": "^VIX",
    "$TNX": "^TNX",    # 10-Year Treasury Yield
    "US10Y": "^TNX",
    "10YUSTREASURY": "^TNX",
    "$TYX": "^TYX",    # 30-Year Treasury Yield
    "$FVX": "^FVX",    # 5-Year Treasury Yield
    "$IRX": "^IRX",    # 13-Week Treasury Bill
    "$HSI": "^HSI",
    "HSI": "^HSI",
    "HANGSENG": "^HSI",
    "NASDAQCOM": "^IXIC",
    "CSI300": "000300.SS",
    "CSI500": "000905.SS",
    "SIE-DE": "SIE.DE",
    "SAP-DE": "SAP.DE",
    "VOW-DE": "VOW3.DE",
    "NESN-SW": "NESN.SW",
    "005930-KS": "005930.KS",
}


def _normalize_yf_symbol(symbol: str) -> str:
    """Legacy compatibility helper; ProFinR execution does not call it."""
    key = symbol.upper().replace(" ", "")
    if key in _YF_SYMBOL_MAP:
        return _YF_SYMBOL_MAP[key]
    canonical = classify_instrument(symbol).provider_symbol
    if "." in canonical and not canonical.endswith((".SZ", ".SH", ".BJ", ".HK", ".SS")):
        return canonical.replace(".", "-")
    return canonical


def _passthrough_yf_symbol(symbol: str) -> str:
    """Pass the explicitly supplied Yahoo symbol through unchanged.

    ProFinR must not turn an alias, company name, index code, or exchange
    symbol into a different asset before the provider call.  Invalid provider
    symbols are returned by yfinance as a typed source error instead.
    """

    return str(symbol or "").strip()

# ── Session selection ──────────────────────────────────────────
# yfinance 1.x ships with curl_cffi, which impersonates Chrome's TLS
# fingerprint (JA3/JA4).  Yahoo Finance uses TLS fingerprinting to
# detect automated clients; plain python-requests is identified and
# rate-limited (429 "Too Many Requests") even with a browser User-Agent
# header, because the TLS handshake differs from a real browser.
#
# Previously this connector created a plain requests.Session with a
# Chrome UA string.  That UA header alone is NOT sufficient — Yahoo
# checks the TLS fingerprint, not just the HTTP headers.  Passing a
# requests.Session to yf.Ticker() overrides yfinance's default
# curl_cffi session, stripping the browser TLS impersonation that
# prevents rate-limiting.
#
# The fix: let yfinance create its own default session (curl_cffi with
# impersonate="chrome") when curl_cffi is available.  Only fall back
# to a plain requests.Session when curl_cffi is not installed.

def _try_import_yfinance() -> Any:
    """Import yfinance with graceful fallback for Python version incompatibility.

    yfinance depends on ``multitasking``, which uses ``type[X]`` syntax
    in its TypedDict definition — valid only in Python 3.9+.  Under
    Python 3.8 the import chain raises ``TypeError: 'type' object is
    not subscriptable``, making yfinance unusable.  This helper imports
    yfinance if available and returns the module; otherwise returns None
    so that the connector can decline actions rather than crashing.
    """
    try:
        import yfinance as yf
        # Verify that Ticker can be instantiated (catches runtime import
        # failures that may not surface at module level)
        _ = yf.Ticker.__init__
        return yf
    except (TypeError, ImportError, AttributeError) as exc:
        logger.warning(
            "yfinance is not available in this Python environment: %s. "
            "All yfinance-backed tools will return empty results.",
            exc,
        )
        return None


_yf = _try_import_yfinance()

if _yf is not None:
    try:
        from curl_cffi import requests as _curl_requests
        # Keep Chrome TLS impersonation while setting the shared contact UA.
        _YF_SESSION = _curl_requests.Session(impersonate="chrome")
        _YF_SESSION.headers.update({"User-Agent": DEFAULT_USER_AGENT})
        logger.info("curl_cffi available — yfinance uses Chrome TLS impersonation (no 429)")
    except ImportError:
        # curl_cffi not available — fall back to plain requests
        # with a browser-like UA.  This WILL trigger Yahoo's rate
        # limiting, so the failover engine's 429 cooldown and
        # circuit breaker are essential.
        _YF_SESSION = requests.Session()
        _YF_SESSION.headers.update({
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;"
                "q=0.9,image/webp,*/*;q=0.8"
            ),
            "Accept-Language": "en-US,en;q=0.5",
        })
        logger.warning(
            "curl_cffi not available — using plain requests.Session. "
            "Yahoo Finance WILL rate-limit this client (429). "
            "Install curl_cffi (>=0.15) to avoid rate-limiting."
        )
else:
    # yfinance itself not available — session is irrelevant
    _YF_SESSION = None


class YFinanceConnector(BaseConnector):
    name = "yfinance"
    actions = frozenset({
        "quote", "ohlcv", "options", "forex", "forex_historical",
        "commodity", "commodity_historical",
    })

    async def fetch(self, action: str, **kwargs: Any) -> Any:
        if _yf is None:
            logger.info("YFinance: skipping action '%s' — yfinance not importable", action)
            return [] if action in {
                "ohlcv", "forex", "forex_historical", "commodity", "commodity_historical",
            } else {}
        dispatch = {
            "quote": self._quote,
            "ohlcv": self._ohlcv,
            "options": self._options,
            "forex": self._forex,
            "forex_historical": self._forex_historical,
            "commodity": self._commodity,
            "commodity_historical": self._commodity_historical,
        }
        handler = dispatch.get(action)
        if handler is None:
            raise ValueError(f"YFinance: unsupported action '{action}'")
        try:
            return await handler(**kwargs)
        except TypeError as exc:
            # Catch the infamous "unsupported operand type(s) for +:
            # 'float' and 'str'" that can arise inside yfinance's
            # internal data reconstruction or pandas arithmetic.
            # Return an empty result so the failover router can try
            # the next source instead of propagating the raw exception.
            logger.warning(
                "YFinance TypeError for action=%s symbol=%s: %s. "
                "Returning empty result to allow failover.",
                action, kwargs.get("symbol", "?"), exc,
            )
            return [] if action == "ohlcv" else {}

    # ── Quote ────────────────────────────────────────────

    async def _quote(self, symbol: str, **_: Any) -> dict:
        yf_symbol = _passthrough_yf_symbol(symbol)
        ticker = _yf.Ticker(yf_symbol, session=_YF_SESSION) if _YF_SESSION is not None else _yf.Ticker(yf_symbol)
        try:
            info = ticker.info or {}
        except TypeError as exc:
            # yfinance can raise TypeError internally during info
            # reconstruction (float + str in pandas arithmetic).
            # Return empty dict so failover can try the next source.
            logger.warning(
                "YFinance _quote TypeError for symbol=%s (yf_symbol=%s): %s. "
                "Returning empty dict to allow failover.",
                symbol, yf_symbol, exc,
            )
            return {}
        # ── Numeric coercion for info dict ─────────────────
        # ticker.info may contain str values in numeric fields
        # (e.g. "N/A", "--", "-") when Yahoo returns partial data.
        # safe_float handles these, but we also need to ensure
        # the fallback chain (currentPrice → regularMarketPrice)
        # doesn't produce float + str arithmetic.
        primary_price = info.get("currentPrice")
        fallback_price = info.get("regularMarketPrice")
        price = primary_price if primary_price is not None else fallback_price
        return {
            "symbol": symbol,
            "open": safe_float(info.get("open")),
            "high": safe_float(info.get("dayHigh")),
            "low": safe_float(info.get("dayLow")),
            "close": safe_float(price),
            "volume": safe_int(info.get("volume")),
            "change": safe_float(info.get("regularMarketChange")),
            "change_pct": safe_float(info.get("regularMarketChangePercent")),
            "timestamp": "",
        }

    # ── OHLCV ────────────────────────────────────────────

    async def _ohlcv(
        self,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
        interval: str = "1d",
        **_: Any,
    ) -> list[dict]:
        yf_symbol = _passthrough_yf_symbol(symbol)
        ticker = _yf.Ticker(yf_symbol, session=_YF_SESSION) if _YF_SESSION is not None else _yf.Ticker(yf_symbol)
        interval_map = {"1d": "1d", "1w": "1wk", "1M": "1mo"}
        yf_interval = interval_map.get(interval, "1d")
        inclusive_end = (
            None
            if end_date is None
            else (pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime(
                "%Y-%m-%d"
            )
        )
        try:
            df = ticker.history(
                start=start_date,
                end=inclusive_end,
                interval=yf_interval,
            )
        except TypeError as exc:
            # yfinance 1.x can raise TypeError internally during
            # data reconstruction (float + str in pandas arithmetic).
            # Catch here to prevent the error from propagating to
            # fetch_with_failover where it would trigger circuit-breaker
            # and retry delays (30s each).  Return empty list so the
            # failover router can immediately try the next source.
            logger.warning(
                "YFinance _ohlcv TypeError for symbol=%s: %s", symbol, exc,
            )
            return []

        if df.empty:
            return []

        # ── Numeric coercion guard ─────────────────────────
        # yfinance sometimes returns mixed-type columns (e.g. '0' as
        # str in Dividends, or NaN-like strings, or comma-formatted
        # numbers like "1,200").  Force ALL columns to numeric before
        # further processing to prevent TypeError: unsupported operand
        # type(s) for +: 'float' and 'str' in downstream pandas ops.
        # Comma-formatted strings are normalized first (pd.to_numeric
        # would otherwise turn "1,200" into NaN).
        def _coerce_column(series: pd.Series) -> pd.Series:
            if series.dtype.kind in "biufc":  # already numeric
                return series
            cleaned = series.astype(str).str.replace(",", "", regex=False)
            return pd.to_numeric(cleaned, errors="coerce")

        for col in list(df.columns):
            if col == "Date" or df.index.name == col:
                continue
            try:
                df[col] = _coerce_column(df[col])
            except (TypeError, ValueError):
                logger.debug("YFinance: skipping to_numeric for column '%s'", col)

        # ── NaN → None conversion ──────────────────────────
        # Replace NaN with None to avoid float + NaN arithmetic issues
        # in downstream code.
        df = df.where(df.notna(), None).reset_index()

        # ── Date column detection ──────────────────────────
        date_col = "Date" if "Date" in df.columns else df.columns[0]

        # ── Row iteration with safe coercion ────────────────
        # safe_float/safe_int handle None and any remaining mixed-type
        # values as a final safety net.
        records = []
        for _, row in df.iterrows():
            raw_date = row[date_col]
            try:
                date_str = str(raw_date.date()) if hasattr(raw_date, "date") else str(raw_date)
            except Exception:
                date_str = str(raw_date) if raw_date is not None else ""
            records.append({
                "symbol": symbol,
                "date": date_str,
                "open": safe_float(row.get("Open")),
                "high": safe_float(row.get("High")),
                "low": safe_float(row.get("Low")),
                "close": safe_float(row.get("Close")),
                "volume": safe_int(row.get("Volume")),
            })
        return records

    # ── Cross-asset fallback adapters ─────────────────────

    async def _spot_from_history(self, symbol: str) -> dict:
        """Build a spot-style response from Yahoo's most recent daily closes."""
        rows = await self._ohlcv(symbol=symbol, interval="1d")
        if not rows:
            return {}
        price = safe_float(rows[-1].get("close"))
        previous = safe_float(rows[-2].get("close")) if len(rows) > 1 else price
        change = price - previous
        return {
            "symbol": symbol,
            "name": symbol,
            "price": price,
            "change": change,
            "change_pct": change / previous * 100 if previous else 0.0,
        }

    async def _forex(self, symbol: str | None = None, **_: Any) -> list[dict]:
        if not symbol:
            return []
        spot = await self._spot_from_history(symbol)
        return [spot] if spot else []

    async def _forex_historical(
        self,
        symbol: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        **_: Any,
    ) -> list[dict]:
        if not symbol:
            return []
        return await self._ohlcv(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
        )

    async def _commodity(self, symbol: str | None = None, **_: Any) -> list[dict]:
        if not symbol:
            return []
        spot = await self._spot_from_history(symbol)
        return [spot] if spot else []

    async def _commodity_historical(
        self,
        symbol: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        **_: Any,
    ) -> list[dict]:
        if not symbol:
            return []
        return await self._ohlcv(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
        )

    # ── Options ──────────────────────────────────────────

    async def _options(self, symbol: str, expiry: str | None = None, **_: Any) -> dict:
        yf_symbol = _passthrough_yf_symbol(symbol)
        ticker = _yf.Ticker(yf_symbol, session=_YF_SESSION) if _YF_SESSION is not None else _yf.Ticker(yf_symbol)
        expirations = ticker.options
        if not expirations:
            return {"symbol": symbol, "expirations": [], "options": []}
        target = expiry or expirations[0]
        chain = ticker.option_chain(target)
        # ── Numeric coercion for option chain data ──────────
        # Apply to_numeric to all numeric columns in calls/puts
        # to prevent float+str TypeError during downstream processing.
        for df_side in ("calls", "puts"):
            side_df = getattr(chain, df_side, None)
            if side_df is not None and not side_df.empty:
                for col in side_df.columns:
                    try:
                        side_df[col] = pd.to_numeric(side_df[col], errors="coerce")
                    except (TypeError, ValueError):
                        pass
        calls = chain.calls.where(chain.calls.notna(), None).to_dict(orient="records") if not chain.calls.empty else []
        puts = chain.puts.where(chain.puts.notna(), None).to_dict(orient="records") if not chain.puts.empty else []
        return {
            "symbol": symbol,
            "expiry": target,
            "expirations": list(expirations),
            "calls": calls[:50],
            "puts": puts[:50],
        }

    async def health_check(self) -> bool:
        if _yf is None:
            return False
        try:
            ticker = _yf.Ticker("AAPL", session=_YF_SESSION) if _YF_SESSION is not None else _yf.Ticker("AAPL")
            info = ticker.info
            return bool(info)
        except (TypeError, Exception):
            return False


register_connector("yfinance", YFinanceConnector)
