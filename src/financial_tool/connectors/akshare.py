"""AKShare connector — open-source Chinese financial data (primary A-share source)."""

from __future__ import annotations

import asyncio
import logging
import math
from datetime import datetime, timezone
from typing import Any

import akshare as ak
import pandas as pd
import requests

from .base import BaseConnector, register_connector
from ._numeric import safe_float, safe_int
from ..errors import SourceUnavailableError


logger = logging.getLogger(__name__)


class AKShareConnector(BaseConnector):
    name = "akshare"
    actions = frozenset({
        "quote", "ohlcv", "realtime_quote", "headlines", "info",
        "series", "money_flow", "search", "income", "balance", "cashflow",
        "filing_search",
    })

    async def fetch(self, action: str, **kwargs: Any) -> Any:
        dispatch = {
            "quote": self._quote,
            "ohlcv": self._ohlcv,
            "realtime_quote": self._realtime,
            "headlines": self._news,
            "info": self._fund_info,
            "series": self._macro,
            "money_flow": self._money_flow,
            "search": self._search,
            # Financial statements
            "income": self._income,
            "balance": self._balance,
            "cashflow": self._cashflow,
            # Filings / disclosures
            "filing_search": self._filing_search,
        }
        handler = dispatch.get(action)
        if handler is None:
            raise ValueError(f"AKShare: unsupported action '{action}'")
        try:
            return await handler(**kwargs)
        except (requests.exceptions.ProxyError, requests.exceptions.ConnectionError) as exc:
            raise SourceUnavailableError("akshare", "proxy/network unavailable") from exc

    @staticmethod
    def _df_to_records(df: pd.DataFrame) -> list[dict]:
        return df.where(df.notna(), None).to_dict(orient="records")

    @staticmethod
    def _iso_date(value: Any) -> str | None:
        """Return a stable ISO date when a provider value is date-like."""
        if value is None or not str(value).strip():
            return None
        try:
            parsed = pd.Timestamp(value)
        except (TypeError, ValueError):
            return None
        if pd.isna(parsed):
            return None
        return parsed.strftime("%Y-%m-%d")

    @classmethod
    def _akshare_date(cls, value: str | None, default: str) -> str:
        """AKShare historical endpoints require compact ``YYYYMMDD`` dates."""
        normalized = cls._iso_date(value)
        return normalized.replace("-", "") if normalized else default

    @staticmethod
    def _akshare_a_share_symbol(symbol: str) -> str:
        """Return the bare six-digit code expected by ``stock_zh_a_hist``."""
        normalized = str(symbol or "").strip().upper()
        for suffix in (".SZ", ".SH", ".BJ"):
            if normalized.endswith(suffix):
                return normalized[: -len(suffix)]
        return normalized

    @staticmethod
    def _quote_number(value: Any) -> float | None:
        number = safe_float(value, float("nan"))
        return number if math.isfinite(number) else None

    @classmethod
    def _normalize_ohlcv_frame(cls, df: Any, symbol: str) -> list[dict]:
        """Normalize known AKShare OHLCV variants without leaking schema errors.

        AKShare returns an empty DataFrame without columns when EastMoney has no
        ``klines`` payload.  Some fallback endpoints also expose the date as an
        index.  Both cases are normal provider outcomes that must let failover
        try the next source rather than raising ``KeyError('date')``.
        """
        if not isinstance(df, pd.DataFrame) or df.empty:
            logger.info("AKShare OHLCV returned no rows for symbol=%s", symbol)
            return []

        normalized = df.copy()
        if "date" not in normalized.columns and (
            isinstance(normalized.index, pd.DatetimeIndex)
            or str(normalized.index.name or "").lower() in {"date", "datetime", "\u65e5\u671f"}
        ):
            normalized = normalized.reset_index()

        normalized = normalized.rename(columns={
            "\u65e5\u671f": "date", "datetime": "date",
            "\u5f00\u76d8": "open", "\u6536\u76d8": "close", "\u6700\u9ad8": "high", "\u6700\u4f4e": "low", "\u6210\u4ea4\u91cf": "volume",
        })
        required = ("date", "open", "high", "low", "close", "volume")
        missing = [column for column in required if column not in normalized.columns]
        if missing:
            logger.warning(
                "AKShare OHLCV schema mismatch for symbol=%s; missing=%s columns=%s",
                symbol,
                missing,
                list(normalized.columns),
            )
            return []

        dates = pd.to_datetime(normalized["date"], errors="coerce")
        if dates.isna().all():
            logger.warning("AKShare OHLCV has no parseable dates for symbol=%s", symbol)
            return []
        normalized = normalized.loc[dates.notna()].copy()
        normalized["date"] = dates.loc[dates.notna()].dt.strftime("%Y-%m-%d")
        normalized["symbol"] = symbol
        for column in ("open", "high", "low", "close", "volume"):
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce").fillna(0)
        return cls._df_to_records(normalized[["symbol", *required]])

    # ── Market data ──────────────────────────────────────

    async def _quote(self, symbol: str, **_: Any) -> dict:
        requested_symbol = str(symbol or "").strip()
        provider_symbol = self._akshare_a_share_symbol(requested_symbol)
        df = await asyncio.to_thread(ak.stock_zh_a_spot_em)
        if not isinstance(df, pd.DataFrame) or df.empty or "\u4ee3\u7801" not in df:
            return {}
        codes = df["\u4ee3\u7801"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(6)
        row = df.loc[codes == provider_symbol]
        if row.empty and "\u540d\u79f0" in df:
            row = df.loc[df["\u540d\u79f0"].astype(str).str.strip() == requested_symbol]
        if row.empty:
            return {}
        r = row.iloc[0]
        price = self._quote_number(r.get("\u6700\u65b0\u4ef7"))
        if price is None or price <= 0:
            return {}
        provider_code = str(r.get("\u4ee3\u7801", provider_symbol)).strip().removesuffix(".0").zfill(6)
        volume = self._quote_number(r.get("\u6210\u4ea4\u91cf"))
        raw_timestamp = r.get("\u65f6\u95f4")
        return {
            "symbol": requested_symbol,
            "provider_symbol": provider_code,
            "price": price,
            "open": self._quote_number(r.get("\u4eca\u5f00")),
            "high": self._quote_number(r.get("\u6700\u9ad8")),
            "low": self._quote_number(r.get("\u6700\u4f4e")),
            "close": price,
            "previous_close": self._quote_number(r.get("\u6628\u6536")),
            "volume": int(volume) if volume is not None else None,
            "change": self._quote_number(r.get("\u6da8\u8dcc\u989d")),
            "change_pct": self._quote_number(r.get("\u6da8\u8dcc\u5e45")),
            "timestamp": "" if raw_timestamp is None or pd.isna(raw_timestamp) else str(raw_timestamp),
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "source": "akshare",
        }

    async def _ohlcv(
        self,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
        interval: str = "1d",
        **_: Any,
    ) -> list[dict]:
        period_map = {"1d": "daily", "1w": "weekly", "1M": "monthly"}
        period = period_map.get(interval, "daily")
        adjust = "qfq"

        # ── Detect market type from symbol format ──────────
        # A-share: 6-digit numeric code starting with 0/3/6 (e.g. "000001", "300750", "600519")
        # HK stock: 5-digit numeric code (e.g. "00700", "09988") or code with .HK suffix
        # A-share suffixes are accepted at the MCP boundary and normalized
        # locally before calling AKShare; HK symbols stay on the HK path.
        if _is_hk_symbol(symbol):
            return await self._ohlcv_hk(symbol, start_date, end_date, period, adjust)

        # Default: A-share
        provider_symbol = self._akshare_a_share_symbol(symbol)
        df = await asyncio.to_thread(
            ak.stock_zh_a_hist,
            symbol=provider_symbol,
            period=period,
            start_date=self._akshare_date(start_date, "20200101"),
            end_date=self._akshare_date(end_date, "20991231"),
            adjust=adjust,
        )
        return self._normalize_ohlcv_frame(df, symbol)

    async def _ohlcv_hk(
        self,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
        period: str = "daily",
        adjust: str = "qfq",
    ) -> list[dict]:
        """Fetch Hong Kong stock OHLCV via akshare ``stock_hk_hist``."""
        try:
            # stock_hk_hist supports date filtering and period selection
            df = await asyncio.to_thread(
                ak.stock_hk_hist,
                symbol=symbol,
                period=period,
                start_date=self._akshare_date(start_date, "19700101"),
                end_date=self._akshare_date(end_date, "22220101"),
                adjust=adjust,
            )
        except requests.exceptions.ProxyError as exc:
            raise SourceUnavailableError("akshare", "proxy/network unavailable") from exc
        except Exception:
            # Fallback: stock_hk_daily returns full history (no date filter)
            try:
                df = await asyncio.to_thread(ak.stock_hk_daily, symbol=symbol, adjust=adjust)
            except requests.exceptions.ProxyError as exc:
                raise SourceUnavailableError("akshare", "proxy/network unavailable") from exc
            except Exception:
                return []

        rows = self._normalize_ohlcv_frame(df, symbol)
        if not rows:
            return []
        df = pd.DataFrame(rows)
        # Date filtering for stock_hk_daily fallback (which returns full history)
        normalized_start = self._iso_date(start_date)
        normalized_end = self._iso_date(end_date)
        if normalized_start:
            df = df[df["date"] >= normalized_start]
        if normalized_end:
            df = df[df["date"] <= normalized_end]
        return self._df_to_records(df)

    async def _realtime(self, symbols: list[str], **_: Any) -> list[dict]:
        df = ak.stock_zh_a_spot_em()
        results = []
        for sym in symbols:
            row = df[df["\u4ee3\u7801"] == sym]
            if row.empty:
                continue
            r = row.iloc[0]
            results.append({
                "symbol": sym,
                "price": safe_float(r.get("\u6700\u65b0\u4ef7")),
                "bid": 0.0,
                "ask": 0.0,
                "volume": safe_int(r.get("\u6210\u4ea4\u91cf")),
                "timestamp": str(r.get("\u65f6\u95f4", "")),
            })
        return results

    # ── News ─────────────────────────────────────────────

    async def _news(self, query: str | None = None, symbols: list[str] | None = None, limit: int = 20, **_: Any) -> list[dict]:
        df = ak.stock_news_em(symbol=symbols[0] if symbols else "")
        df = df.rename(columns={
            "\u65b0\u95fb\u6807\u9898": "title", "\u65b0\u95fb\u5185\u5bb9": "content",
            "\u53d1\u5e03\u65f6\u95f4": "published_at", "\u6587\u7ae0\u6765\u6e90": "source",
        })
        results = []
        for _, r in df.head(limit).iterrows():
            results.append({
                "title": str(r.get("title", "")),
                "source": str(r.get("source", "")),
                "published_at": str(r.get("published_at", "")),
                "url": "",
                "symbols": symbols or [],
            })
        return results

    # ── Fund info ────────────────────────────────────────

    async def _fund_info(self, symbol: str, include_holdings: bool = False, **_: Any) -> dict:
        df = ak.fund_open_fund_info_em(symbol=symbol, indicator="\u5355\u4f4d\u51c0\u503c\u8d70\u52bf")
        if df.empty:
            return {}
        latest = df.iloc[-1]
        result = {
            "symbol": symbol,
            "name": "",
            "nav": safe_float(latest.get("\u5355\u4f4d\u51c0\u503c")),
            "total_assets": 0.0,
            "holdings": [],
        }
        if include_holdings:
            try:
                h_df = ak.fund_portfolio_hold_em(symbol=symbol, date="")
                if not h_df.empty:
                    result["holdings"] = self._df_to_records(h_df.head(10))
            except Exception:
                pass
        return result

    # ── Macro ────────────────────────────────────────────

    async def _macro(
        self, series_id: str, start_date: str | None = None, end_date: str | None = None, **_: Any,
    ) -> list[dict]:
        try:
            df = ak.macro_china_gdp()
            df = df.rename(columns={"\u65e5\u671f": "date", "\u56fd\u5185\u751f\u4ea7\u603b\u503c-\u7edd\u5bf9\u503c": "value"})
            df["series_id"] = series_id
            df["date"] = df["date"].astype(str)
            df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0)
            return self._df_to_records(df[["series_id", "date", "value"]].tail(20))
        except Exception:
            return []

    # ── Money flow ───────────────────────────────────────

    async def _money_flow(self, symbol: str, period: str = "1d", **_: Any) -> dict:
        try:
            df = ak.stock_individual_fund_flow(stock=symbol, market="sh")
            if df.empty:
                return {"symbol": symbol, "date": "", "inflow": 0, "outflow": 0, "net_flow": 0}
            r = df.iloc[-1]
            net_flow = safe_float(r.get("\u4e3b\u529b\u51c0\u6d41\u5165-\u51c0\u989d"))
            return {
                "symbol": symbol,
                "date": str(r.get("\u65e5\u671f", "")),
                "inflow": net_flow if net_flow > 0 else 0,
                "outflow": -net_flow if net_flow < 0 else 0,
                "net_flow": net_flow,
            }
        except Exception:
            return {"symbol": symbol, "date": "", "inflow": 0, "outflow": 0, "net_flow": 0}

    # ── Search ───────────────────────────────────────────

    async def _search(self, query: str, limit: int = 10, **_: Any) -> list[dict]:
        df = ak.stock_zh_a_spot_em()
        mask = df["\u540d\u79f0"].str.contains(query, na=False) | df["\u4ee3\u7801"].str.contains(query, na=False)
        results = []
        for _, r in df[mask].head(limit).iterrows():
            results.append({
                "query": query,
                "source": "akshare",
                "data_type": "stock",
                "relevance": 1.0,
                "payload": {"symbol": r["\u4ee3\u7801"], "name": r["\u540d\u79f0"]},
            })
        return results

    # ── Financial statements ─────────────────────────────

    async def _income(self, symbol: str, period: str = "annual", **_: Any) -> dict:
        """Get income statement via AKShare — em-style financial report."""
        try:
            # symbol is e.g. "600519"; select the exchange prefix locally.
            df = ak.stock_financial_report_sina(stock=f"sh{symbol}" if symbol.startswith("6") else f"sz{symbol}", symbol="\u5229\u6da6\u8868")
            if df.empty:
                return {}
            # Rename key columns from Chinese to canonical English
            col_map = {
                "\u62a5\u544a\u65e5": "date",
                "\u8425\u4e1a\u6536\u5165": "revenue",
                "\u8425\u4e1a\u6210\u672c": "cogs",
                "\u8425\u4e1a\u5229\u6da6": "operating_profit",
                "\u5229\u6da6\u603b\u989d": "total_profit",
                "\u51c0\u5229\u6da6": "net_income",
            }
            df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
            r = df.iloc[0] if period == "annual" else df.iloc[0]
            revenue = safe_float(r.get("revenue"))
            cogs = safe_float(r.get("cogs"))
            gross_profit = revenue - cogs
            return {
                "symbol": symbol,
                "date": str(r.get("date", "")),
                "revenue": revenue,
                "cogs": cogs,
                "gross_profit": gross_profit,
                "net_income": safe_float(r.get("net_income")),
            }
        except Exception:
            # Fallback: try em-style profit sheet
            try:
                df = ak.stock_profit_sheet_by_report_em(symbol=symbol)
                if df.empty:
                    return {}
                r = df.iloc[0]
                revenue = safe_float(r.get("\u8425\u4e1a\u6536\u5165"))
                cogs = safe_float(r.get("\u8425\u4e1a\u6210\u672c"))
                return {
                    "symbol": symbol,
                    "date": str(r.get("\u62a5\u544a\u671f", "")),
                    "revenue": revenue,
                    "cogs": cogs,
                    "gross_profit": revenue - cogs,
                    "net_income": safe_float(r.get("\u51c0\u5229\u6da6")),
                }
            except Exception:
                return {}

    async def _balance(self, symbol: str, period: str = "annual", **_: Any) -> dict:
        """Get balance sheet via AKShare."""
        try:
            df = ak.stock_balance_sheet_by_report_em(symbol=symbol)
            if df.empty:
                return {}
            r = df.iloc[0]
            return {
                "symbol": symbol,
                "date": str(r.get("\u62a5\u544a\u671f", "")),
                "total_assets": safe_float(r.get("\u603b\u8d44\u4ea7")),
                "total_liabilities": safe_float(r.get("\u603b\u8d1f\u503a")),
                "total_equity": safe_float(r.get("\u6240\u6709\u8005\u6743\u76ca\u5408\u8ba1")),
            }
        except Exception:
            return {}

    async def _cashflow(self, symbol: str, period: str = "annual", **_: Any) -> dict:
        """Get cash flow statement via AKShare."""
        try:
            df = ak.stock_cash_flow_statement_by_report_em(symbol=symbol)
            if df.empty:
                return {}
            r = df.iloc[0]
            return {
                "symbol": symbol,
                "date": str(r.get("\u62a5\u544a\u671f", "")),
                "operating_cf": safe_float(r.get("\u7ecf\u8425\u6d3b\u52a8\u4ea7\u751f\u7684\u73b0\u91d1\u6d41\u91cf\u51c0\u989d")),
                "investing_cf": safe_float(r.get("\u6295\u8d44\u6d3b\u52a8\u4ea7\u751f\u7684\u73b0\u91d1\u6d41\u91cf\u51c0\u989d")),
                "financing_cf": safe_float(r.get("\u7b79\u8d44\u6d3b\u52a8\u4ea7\u751f\u7684\u73b0\u91d1\u6d41\u91cf\u51c0\u989d")),
            }
        except Exception:
            return {}

    # ── Filings / disclosures ────────────────────────────

    async def _filing_search(
        self,
        query: str,
        form_type: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        **_: Any,
    ) -> list[dict]:
        """Search A-share announcements / filings via AKShare."""
        try:
            # Use stock_notice_report from East Money
            df = ak.stock_notice_report(symbol=query)
            if df.empty:
                return []
            results = []
            for _, r in df.head(20).iterrows():
                filing_date = str(r.get("\u516c\u544a\u65e5\u671f", r.get("\u65e5\u671f", "")))
                # Optional date filtering
                if date_from and filing_date < date_from:
                    continue
                if date_to and filing_date > date_to:
                    continue
                results.append({
                    "filing_id": str(r.get("\u516c\u544a\u4ee3\u7801", r.get("\u4ee3\u7801", ""))),
                    "symbol": query,
                    "form_type": str(r.get("\u516c\u544a\u7c7b\u578b", r.get("\u7c7b\u578b", ""))),
                    "filed_date": filing_date,
                    "title": str(r.get("\u516c\u544a\u6807\u9898", r.get("\u6807\u9898", ""))),
                })
            return results
        except Exception:
            return []

    async def health_check(self) -> bool:
        try:
            ak.stock_zh_a_spot_em()
            return True
        except Exception:
            return False


def _is_hk_symbol(symbol: str) -> bool:
    """Return True if *symbol* looks like a Hong Kong stock code.

    HK codes are typically 5-digit numeric strings starting with ``0``
    (e.g. ``00700`` = Tencent, ``09988`` = Alibaba HK).  They differ
    from A-share codes which are 6-digit and start with 0/3/6.
    """
    s = symbol.strip()
    # 5-digit numeric starting with 0 → HK stock
    if len(s) == 5 and s.isdigit() and s.startswith("0"):
        return True
    # Explicit .HK suffix (should already be stripped by _clean_cn_symbol
    # but handle it here as a safety net)
    if s.endswith(".HK"):
        return True
    return False


register_connector("akshare", AKShareConnector)
