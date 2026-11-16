"""FMP (Financial Modeling Prep) connector — US market data and fundamentals.

Uses the FMP **stable** API (`https://financialmodelingprep.com/stable`).

The legacy v3 API (`/api/v3/...`) was deprecated on 2025-08-31 — new
subscriptions return HTTP 403 with "Legacy Endpoint" error for every v3
call.  The stable API uses **query-parameter style** routing
(`/endpoint?symbol=AAPL`) rather than v3's path-style (`/endpoint/AAPL`),
and most endpoint names changed (e.g. `/historical-price-full/{sym}` →
`/historical-price-eod/full?symbol={sym}`).

Each handler below builds the request via `_get(endpoint, params)` where
``endpoint`` is the *bare* stable path (no symbol in the path) and
``params`` carries ``symbol``/``from``/``to``/etc. as query parameters.
Endpoints that are not available on stable (rating, technical-indicator,
sec-filings, institutional-holder, etf-weightings, sectors-performance,
market-status, form-13f, altman-z, piotroski, wacc, esg-score) simply
don't exist on stable — for those the handler returns an empty result
and lets the failover router try the next source (yfinance / sec_edgar).
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from .base import BaseConnector, register_connector
from ._numeric import safe_float as _safe_float
from ._numeric import safe_int as _safe_int

logger = logging.getLogger("financial_tool.connector.fmp")

_STABLE_URL = "https://financialmodelingprep.com/stable"
_HISTORICAL_MIN_LIMIT = 100


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in ("data", "historical", "historicalStockList", "results"):
            nested = value.get(key)
            if isinstance(nested, list):
                return nested
        return [
            {"name": key, **item} if isinstance(item, dict) else {"name": key, "value": item}
            for key, item in value.items()
        ]
    return []


def _row_date(row: dict[str, Any], date_fields: tuple[str, ...] = ("date",)) -> str:
    for field in date_fields:
        value = row.get(field)
        if value:
            return str(value)[:10]
    return ""


def _filter_rows_by_window(
    rows: list[dict[str, Any]],
    start_date: str | None = None,
    end_date: str | None = None,
    as_of_date: str | None = None,
    date_fields: tuple[str, ...] = ("date",),
) -> list[dict[str, Any]]:
    if not (start_date or end_date or as_of_date):
        return rows
    filtered: list[dict[str, Any]] = []
    for row in rows:
        date = _row_date(row, date_fields)
        if not date:
            continue
        if start_date and date < start_date:
            continue
        if end_date and date > end_date:
            continue
        if as_of_date and date > as_of_date:
            continue
        filtered.append(row)
    return filtered


def _out_of_window_marker(
    symbol: str | None,
    rows: list[dict[str, Any]],
    start_date: str | None,
    end_date: str | None,
    as_of_date: str | None,
    date_fields: tuple[str, ...] = ("date",),
) -> dict[str, Any]:
    dates = sorted(date for date in (_row_date(row, date_fields) for row in rows) if date)
    marker: dict[str, Any] = {
        "status": "out_of_window",
        "source": "fmp",
        "requested_window": {
            "start_date": start_date,
            "end_date": end_date,
            "as_of_date": as_of_date,
        },
        "actual_min_date": dates[0] if dates else "",
        "actual_max_date": dates[-1] if dates else "",
        "temporal_safe": False,
        "reason": "FMP endpoint returned rows outside requested window",
    }
    if symbol:
        marker["symbol"] = symbol
    return marker


def _snapshot_blocked(
    symbol: str | None,
    as_of_date: str | None,
    start_date: str | None = None,
    end_date: str | None = None,
    reason: str = "Current FMP snapshot is not safe for historical as-of requests",
) -> dict[str, Any]:
    marker: dict[str, Any] = {
        "status": "snapshot_blocked",
        "source": "fmp",
        "requested_window": {
            "start_date": start_date,
            "end_date": end_date,
            "as_of_date": as_of_date,
        },
        "temporal_safe": False,
        "reason": reason,
    }
    if symbol:
        marker["symbol"] = symbol
    return marker


def _temporal_rows_or_marker(
    symbol: str | None,
    rows: list[dict[str, Any]],
    start_date: str | None,
    end_date: str | None,
    as_of_date: str | None,
    date_fields: tuple[str, ...] = ("date",),
) -> list[dict[str, Any]]:
    filtered = _filter_rows_by_window(rows, start_date, end_date, as_of_date, date_fields)
    if filtered or not (start_date or end_date or as_of_date):
        return filtered
    if not rows:
        return []
    return [_out_of_window_marker(symbol, rows, start_date, end_date, as_of_date, date_fields)]


def _historical_limit(
    limit: int,
    start_date: str | None = None,
    end_date: str | None = None,
    as_of_date: str | None = None,
) -> int:
    if start_date or end_date or as_of_date:
        return max(int(limit), _HISTORICAL_MIN_LIMIT)
    return int(limit)


_FMP_STATEMENT_ENDPOINTS = {
    "income": "/income-statement",
    "balance": "/balance-sheet-statement",
    "cashflow": "/cash-flow-statement",
}

_FMP_FIELD_ALIASES = {
    "research and development": "research_and_development_expense",
    "r&d": "research_and_development_expense",
    "r&d expense": "research_and_development_expense",
    "retained earnings": "retained_earnings",
    "long term debt": "long_term_debt",
    "long-term debt": "long_term_debt",
    "cash and cash equivalents": "cash_and_cash_equivalents",
    "short term investments": "short_term_investments",
    "interest expense": "interest_expense",
    "provision for income taxes": "income_tax_expense",
    "income tax expense": "income_tax_expense",
    "earnings per share": "earnings_per_share",
    "eps": "earnings_per_share",
}

_FMP_FIELD_KEYS = {
    "revenue": ("revenue",),
    "cogs": ("costOfRevenue",),
    "gross_profit": ("grossProfit",),
    "operating_income": ("operatingIncome",),
    "net_income": ("netIncome",),
    "research_and_development_expense": ("researchAndDevelopmentExpenses", "researchAndDevelopmentExpense"),
    "interest_expense": ("interestExpense",),
    "income_tax_expense": ("incomeTaxExpense", "incomeTaxExpenseBenefit"),
    "earnings_per_share": ("eps", "epsdiluted", "weightedAverageShsOut"),
    "total_assets": ("totalAssets",),
    "current_assets": ("totalCurrentAssets",),
    "total_liabilities": ("totalLiabilities",),
    "current_liabilities": ("totalCurrentLiabilities",),
    "stockholders_equity": ("totalStockholdersEquity", "totalEquity"),
    "retained_earnings": ("retainedEarnings",),
    "long_term_debt": ("longTermDebt",),
    "cash_and_cash_equivalents": ("cashAndCashEquivalents",),
    "short_term_investments": ("shortTermInvestments",),
    "operating_cash_flow": ("operatingCashFlow",),
    "investing_cash_flow": ("netCashUsedForInvestingActivites", "netCashUsedForInvestingActivities"),
    "financing_cash_flow": ("netCashUsedProvidedByFinancingActivities",),
    "capital_expenditures": ("capitalExpenditure", "capitalExpenditures"),
    "free_cash_flow": ("freeCashFlow",),
}


def _canonical_fmp_field(field: Any) -> str:
    text = " ".join(str(field or "").strip().lower().replace("-", " ").replace("_", " ").split())
    return _FMP_FIELD_ALIASES.get(text, text.replace(" ", "_"))


def _fmp_row_fiscal_year(row: dict[str, Any]) -> int | None:
    for key in ("calendarYear", "fiscalYear"):
        try:
            value = row.get(key)
            if value not in (None, ""):
                return int(value)
        except (TypeError, ValueError):
            pass
    date_value = str(row.get("date") or "")[:4]
    return int(date_value) if date_value.isdigit() else None


def _select_fmp_statement_row(
    rows: list[dict[str, Any]],
    *,
    fiscal_year: int | None = None,
    fiscal_period: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    requested_fp = str(fiscal_period or "").strip().upper()
    candidates: list[dict[str, Any]] = []
    for row in rows:
        date_value = str(row.get("date") or "")[:10]
        if start_date and date_value and date_value < start_date:
            continue
        if end_date and date_value and date_value > end_date:
            continue
        row_fy = _fmp_row_fiscal_year(row)
        if fiscal_year is not None and row_fy != int(fiscal_year):
            continue
        row_fp = str(row.get("period") or row.get("periodLabel") or "").strip().upper()
        if requested_fp and row_fp != requested_fp:
            continue
        candidates.append(row)
    if not candidates:
        return {}
    return max(candidates, key=lambda item: str(item.get("date") or ""))


def _extract_fmp_statement_fields(
    row: dict[str, Any],
    requested_fields: list[str],
) -> dict[str, float]:
    if not row:
        return {}
    values: dict[str, float] = {}
    for field in requested_fields:
        for key in _FMP_FIELD_KEYS.get(field, (field,)):
            if key not in row or row.get(key) in (None, ""):
                continue
            values[field] = _safe_float(row.get(key))
            break
    return values


def _fmp_period_metadata(row: dict[str, Any]) -> dict[str, Any]:
    if not row:
        return {}
    return {
        "period_end": str(row.get("date") or "")[:10],
        "fiscal_year": _fmp_row_fiscal_year(row),
        "fiscal_period": str(row.get("period") or row.get("periodLabel") or ""),
    }


class FMPConnector(BaseConnector):
    name = "fmp"
    actions = frozenset({
        "quote", "ohlcv", "income", "balance", "cashflow", "search", "profile",
        "ratios", "ratios_ttm", "enterprise_value", "growth", "dcf", "dcf_historical",
        "rating", "rating_historical", "key_metrics", "key_metrics_ttm",
        "institutional_holders", "mutual_fund_holders", "insider_trading", "form_13f",
        "earning_calendar", "earning_calendar_historical", "earnings_surprises",
        "dividend_calendar", "dividend_historical", "ipo_calendar", "stock_split_calendar",
        "economic_calendar", "market_status", "market_cap", "market_cap_historical",
        "etf_sector_weightings", "etf_country_weightings", "sec_filings", "treasury",
        "analyst_estimates", "price_target", "earnings_transcript", "esg", "altman_z",
        "piotroski", "wacc", "commodity", "commodity_historical", "forex", "forex_historical",
        "statement_fields",
    })

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._api_key = self._config.get("api_key", "") or os.getenv("FMP_API_KEY", "")
        self._base_url = str(self._config.get("base_url") or _STABLE_URL).rstrip("/")

    async def _get(self, endpoint: str, params: dict | None = None) -> Any:
        """Call a **stable** API endpoint (query-param style).

        ``endpoint`` is the bare path (e.g. ``"/quote"``); the symbol and
        other filters are passed as query params (e.g. ``{"symbol": "AAPL"}``).
        Returns the parsed JSON body (usually a list of records).  Returns
        ``[]`` for any non-200 response so the caller can fall back to
        another source via the failover engine.
        """
        params = dict(params or {})
        params["apikey"] = self._api_key
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(f"{self._base_url}{endpoint}", params=params)
            if resp.status_code == 200:
                return resp.json()
            # 402 = plan-limited, 404 = endpoint not on stable, 403 = legacy block.
            # All of these mean "FMP cannot serve this request" — return []
            # so fetch_with_failover treats it as an empty result and tries
            # the next source (yfinance/sec_edgar) without raising.
            if resp.status_code in (400, 402, 403, 404):
                logger.debug(
                    "FMP stable %s returned %s: %s",
                    endpoint, resp.status_code, resp.text[:120],
                )
                return []
            # For unexpected errors (5xx, network), raise so the failover
            # engine's retry logic can kick in.
            resp.raise_for_status()
            return resp.json()

    async def _get_stable(self, path: str, params: dict | None = None) -> Any:
        params = params or {}
        params["apikey"] = self._api_key
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(f"{self._base_url}{path}", params=params)
            resp.raise_for_status()
            return resp.json()

    async def fetch(self, action: str, **kwargs: Any) -> Any:
        dispatch = {
            "quote": self._quote,
            "ohlcv": self._ohlcv,
            "income": self._income,
            "balance": self._balance,
            "cashflow": self._cashflow,
            "statement_fields": self._statement_fields,
            "search": self._search,
            # Profile & valuation
            "profile": self._profile,
            "ratios": self._ratios,
            "ratios_ttm": self._ratios_ttm,
            "enterprise_value": self._enterprise_value,
            "growth": self._growth,
            "dcf": self._dcf,
            "dcf_historical": self._dcf_historical,
            "rating": self._rating,
            "rating_historical": self._rating_historical,
            "key_metrics": self._key_metrics,
            "key_metrics_ttm": self._key_metrics_ttm,
            # Ownership
            "institutional_holders": self._institutional_holders,
            "mutual_fund_holders": self._mutual_fund_holders,
            "insider_trading": self._insider_trading,
            "form_13f": self._form_13f,
            # Calendar
            "earning_calendar": self._earning_calendar,
            "earning_calendar_historical": self._earning_calendar_historical,
            "earnings_surprises": self._earnings_surprises,
            "dividend_calendar": self._dividend_calendar,
            "dividend_historical": self._dividend_historical,
            "ipo_calendar": self._ipo_calendar,
            "stock_split_calendar": self._stock_split_calendar,
            "economic_calendar": self._economic_calendar,
            # Market data extensions — technical_indicator, market_status,
            # and sectors_performance endpoints mostly do NOT exist on the
            # FMP stable API.  market_status is available through
            # /is-the-market-open-all; technical_indicator remains excluded
            # because indicators are computed locally from OHLCV.
            "market_status": self._market_status,
            "market_cap": self._market_cap,
            "market_cap_historical": self._market_cap_historical,
            # ETF extensions
            "etf_sector_weightings": self._etf_sector_weightings,
            "etf_country_weightings": self._etf_country_weightings,
            # Filings
            "sec_filings": self._sec_filings,
            # New: Treasury, Analyst, ESG, Risk Models, Commodity, Forex
            "treasury": self._treasury,
            "analyst_estimates": self._analyst_estimates,
            "price_target": self._price_target,
            "earnings_transcript": self._earnings_transcript,
            "esg": self._esg,
            "altman_z": self._altman_z,
            "piotroski": self._piotroski,
            "wacc": self._wacc,
            "commodity": self._commodity,
            "commodity_historical": self._commodity_historical,
            "forex": self._forex,
            "forex_historical": self._forex_historical,
        }
        handler = dispatch.get(action)
        if handler is None:
            raise ValueError(f"FMP: unsupported action '{action}'")
        return await handler(**kwargs)

    # ── Quote ────────────────────────────────────────────

    async def _quote(
        self,
        symbol: str,
        as_of_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        **_: Any,
    ) -> dict:
        if as_of_date or start_date or end_date:
            return _snapshot_blocked(symbol, as_of_date, start_date, end_date)
        data = await self._get("/quote", {"symbol": symbol})
        if not data:
            return {}
        q = data[0]
        # Stable API uses 'price' for close, 'changePercentage' for pct,
        # 'dayHigh'/'dayLow' for high/low (same as v3 but confirmed).
        return {
            "symbol": q.get("symbol", symbol),
            "open": _safe_float(q.get("open")),
            "high": _safe_float(q.get("dayHigh")),
            "low": _safe_float(q.get("dayLow")),
            "close": _safe_float(q.get("price")),
            "volume": _safe_int(q.get("volume")),
            "change": _safe_float(q.get("change")),
            "change_pct": _safe_float(q.get("changePercentage")),
            "timestamp": str(q.get("timestamp", "")),
        }

    async def _market_status(self, **_: Any) -> dict:
        data = await self._get("/is-the-market-open-all")
        stock_markets = data.get("stockMarkets", {}) if isinstance(data, dict) else {}
        if isinstance(stock_markets, dict):
            stock_market_rows = [
                {
                    "name": str(name),
                    "open": bool(value.get("isOpen"))
                    if isinstance(value, dict)
                    else bool(value),
                    "current_time": str(value.get("currentTime", ""))
                    if isinstance(value, dict)
                    else "",
                }
                for name, value in stock_markets.items()
            ]
        elif isinstance(stock_markets, list):
            stock_market_rows = [
                {
                    "name": str(item.get("name", item.get("exchange", ""))),
                    "open": bool(item.get("isOpen", item.get("open", False))),
                    "current_time": str(item.get("currentTime", item.get("current_time", ""))),
                }
                for item in stock_markets
                if isinstance(item, dict)
            ]
        else:
            stock_market_rows = []
        forex = data.get("forex", {}) if isinstance(data, dict) else {}
        crypto = data.get("crypto", {}) if isinstance(data, dict) else {}
        return {
            "source": "fmp",
            "stock_markets": stock_market_rows,
            "forex_open": bool(forex.get("isOpen")) if isinstance(forex, dict) else bool(forex),
            "crypto_open": bool(crypto.get("isOpen")) if isinstance(crypto, dict) else bool(crypto),
        }

    async def _technical_indicator(
        self,
        symbol: str,
        period: int = 14,
        statistics_type: str = "SMA",
        time_delta: str = "daily",
        **_: Any,
    ) -> list[dict]:
        # FMP stable does not expose the legacy technical indicator endpoint.
        # Keep this method for compatibility with tests/direct callers, but do
        # not register it in fetch(); local tools compute indicators from
        # OHLCV locally.
        return []

    # ── OHLCV ────────────────────────────────────────────

    async def _ohlcv(
        self,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
        interval: str = "1d",
        **_: Any,
    ) -> list[dict]:
        # Stable API: /historical-price-eod/full?symbol=AAPL&from=...&to=...
        # Returns a flat list of {symbol, date, open, high, low, close,
        # volume, change, changePercent, vwap} dicts (no nested wrapper).
        if not self._api_key:
            return []
        # The shared crypto router uses exchange-style BTC-USD / ETH-USD;
        # FMP's stable historical endpoint expects BTCUSD / ETHUSD.
        fmp_symbol = symbol.replace("-", "") if symbol.upper() in {"BTC-USD", "ETH-USD"} else symbol
        params: dict[str, Any] = {"symbol": fmp_symbol}
        if start_date:
            params["from"] = start_date
        if end_date:
            params["to"] = end_date
        data = await self._get("/historical-price-eod/full", params)
        items = _filter_rows_by_window(
            _as_list(data),
            start_date,
            end_date,
            end_date,
        )
        return [
            {
                "symbol": symbol,
                "date": item.get("date", ""),
                "open": _safe_float(item.get("open")),
                "high": _safe_float(item.get("high")),
                "low": _safe_float(item.get("low")),
                "close": _safe_float(item.get("close")),
                "volume": _safe_int(item.get("volume")),
            }
            for item in items[:5000 if fmp_symbol != symbol else 500]
        ]

    # ── Fundamentals ─────────────────────────────────────

    @staticmethod
    def _statement_in_window(
        data: list[dict[str, Any]],
        start_date: str | None,
        end_date: str | None,
    ) -> dict[str, Any]:
        candidates = [
            item
            for item in data
            if item.get("date")
            and (not start_date or str(item["date"]) >= start_date)
            and (not end_date or str(item["date"]) <= end_date)
        ]
        if not candidates:
            return {}
        return max(candidates, key=lambda item: str(item["date"]))

    @staticmethod
    def _statement_temporal_metadata(
        statement: dict[str, Any],
        *,
        start_date: str | None,
        end_date: str | None,
        as_of_date: str | None,
    ) -> dict[str, Any]:
        period_end = str(statement.get("date") or "")[:10]
        period_label = str(statement.get("period") or statement.get("periodLabel") or "")
        return {
            "requested_start": start_date,
            "requested_end": end_date,
            "requested_as_of": as_of_date,
            "actual_min_date": period_end or None,
            "actual_max_date": period_end or None,
            "date_dimension": "fiscal_period",
            "fiscal_period_labels": [period_label] if period_label else [],
            "period_ends": [period_end] if period_end else [],
            "published_at": [],
            "filed_at": [],
            "temporal_safe": bool(period_end),
            "date_verification": "verified" if period_end else "unverified",
            "coverage_completeness": "known",
        }

    async def _statement_fields(
        self,
        symbol: str,
        fields: list[str] | None = None,
        normalized_fields: list[str] | None = None,
        statement: str | None = None,
        period: str = "annual",
        fiscal_year: int | None = None,
        fiscal_period: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        requested = [
            _canonical_fmp_field(item)
            for item in (normalized_fields or fields or [])
            if str(item or "").strip()
        ]
        requested = list(dict.fromkeys(requested))
        statement_name = str(statement or "").strip().lower()
        if not statement_name:
            balance_fields = {
                "total_assets",
                "current_assets",
                "total_liabilities",
                "current_liabilities",
                "stockholders_equity",
                "retained_earnings",
                "long_term_debt",
                "cash_and_cash_equivalents",
                "short_term_investments",
            }
            cashflow_fields = {
                "operating_cash_flow",
                "investing_cash_flow",
                "financing_cash_flow",
                "capital_expenditures",
                "free_cash_flow",
            }
            field_set = set(requested)
            if field_set and field_set <= balance_fields:
                statement_name = "balance"
            elif field_set and field_set <= cashflow_fields:
                statement_name = "cashflow"
            else:
                statement_name = "income"
        endpoint = _FMP_STATEMENT_ENDPOINTS.get(statement_name)
        if endpoint is None:
            return {
                "status": "invalid_request",
                "symbol": symbol,
                "source": "fmp",
                "values": {},
                "missing_fields": requested,
                "reason": "unsupported_statement",
            }
        rows = [
            dict(item)
            for item in _as_list(await self._get(endpoint, {"symbol": symbol, "period": period}))
            if isinstance(item, dict)
        ]
        selected = _select_fmp_statement_row(
            rows,
            fiscal_year=fiscal_year,
            fiscal_period=fiscal_period,
            start_date=start_date,
            end_date=end_date,
        )
        values = _extract_fmp_statement_fields(selected, requested)
        missing = [field for field in requested if field not in values]
        return {
            "status": "success" if not missing else ("partial" if values else "field_missing"),
            "symbol": symbol,
            "source": "fmp",
            "statement": statement_name,
            "values": values,
            "missing_fields": missing,
            "periods": [_fmp_period_metadata(selected)] if selected else [],
        }

    async def _income(
        self,
        symbol: str,
        period: str = "annual",
        start_date: str | None = None,
        end_date: str | None = None,
        as_of_date: str | None = None,
        **_: Any,
    ) -> dict:
        data = await self._get("/income-statement", {"symbol": symbol, "period": period})
        s = self._statement_in_window(
            _as_list(data), start_date, min(end_date, as_of_date) if end_date and as_of_date else end_date
        )
        if not s:
            return {}
        return {
            "symbol": s.get("symbol", symbol),
            "date": s.get("date", ""),
            "revenue": _safe_float(s.get("revenue")),
            "cogs": _safe_float(s.get("costOfRevenue")),
            "gross_profit": _safe_float(s.get("grossProfit")),
            "net_income": _safe_float(s.get("netIncome")),
            "temporal_metadata": self._statement_temporal_metadata(
                s, start_date=start_date, end_date=end_date, as_of_date=as_of_date
            ),
        }

    async def _balance(
        self,
        symbol: str,
        period: str = "annual",
        start_date: str | None = None,
        end_date: str | None = None,
        as_of_date: str | None = None,
        **_: Any,
    ) -> dict:
        data = await self._get("/balance-sheet-statement", {"symbol": symbol, "period": period})
        s = self._statement_in_window(
            _as_list(data), start_date, min(end_date, as_of_date) if end_date and as_of_date else end_date
        )
        if not s:
            return {}
        return {
            "symbol": s.get("symbol", symbol),
            "date": s.get("date", ""),
            "total_assets": _safe_float(s.get("totalAssets")),
            "total_liabilities": _safe_float(s.get("totalLiabilities")),
            "total_equity": _safe_float(s.get("totalStockholdersEquity")),
            "temporal_metadata": self._statement_temporal_metadata(
                s, start_date=start_date, end_date=end_date, as_of_date=as_of_date
            ),
        }

    async def _cashflow(
        self,
        symbol: str,
        period: str = "annual",
        start_date: str | None = None,
        end_date: str | None = None,
        as_of_date: str | None = None,
        **_: Any,
    ) -> dict:
        data = await self._get("/cash-flow-statement", {"symbol": symbol, "period": period})
        s = self._statement_in_window(
            _as_list(data), start_date, min(end_date, as_of_date) if end_date and as_of_date else end_date
        )
        if not s:
            return {}
        return {
            "symbol": s.get("symbol", symbol),
            "date": s.get("date", ""),
            "operating_cf": _safe_float(s.get("operatingCashFlow")),
            "investing_cf": _safe_float(s.get("netCashUsedForInvestingActivites")),
            "financing_cf": _safe_float(s.get("netCashUsedProvidedByFinancingActivities")),
            "temporal_metadata": self._statement_temporal_metadata(
                s, start_date=start_date, end_date=end_date, as_of_date=as_of_date
            ),
        }

    # ── Search ───────────────────────────────────────────

    async def _search(self, query: str, limit: int = 10, **_: Any) -> list[dict]:
        # Stable API: /search-name (the /search endpoint is not on stable)
        data = await self._get("/search-name", {"query": query, "limit": str(limit)})
        return [
            {
                "query": query,
                "source": "fmp",
                "data_type": r.get("type", ""),
                "relevance": 1.0,
                "payload": {"symbol": r.get("symbol", ""), "name": r.get("name", "")},
            }
            for r in _as_list(data)[:limit]
        ]

    # ── Profile ───────────────────────────────────────────

    async def _profile(
        self,
        symbol: str,
        as_of_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        **_: Any,
    ) -> dict:
        data = await self._get("/profile", {"symbol": symbol})
        if not data:
            return {}
        p = data[0]
        profile = {
            "symbol": p.get("symbol", symbol),
            "company_name": p.get("companyName", ""),
            "exchange": p.get("exchangeShortName", ""),
            "currency": p.get("currency", ""),
            "cik": p.get("cik", ""),
            "isin": p.get("isin", ""),
            "cusip": p.get("cusip", ""),
            "sector": p.get("sector", ""),
            "industry": p.get("industry", ""),
            "website": p.get("website", ""),
            "description": p.get("description", ""),
            "ceo": p.get("ceo", ""),
            "country": p.get("country", ""),
            "full_time_employees": p.get("fullTimeEmployees"),
            "ipo_date": p.get("ipoDate", ""),
            "image": p.get("image", ""),
            "default_image": bool(p.get("defaultImage", False)),
        }
        if as_of_date or start_date or end_date:
            profile["snapshot_fields_blocked"] = ["market_cap", "price", "last_dividend", "dcf_diff", "dcf"]
            profile["temporal_safe"] = True
        else:
            profile.update(
                {
                    "market_cap": _safe_float(p.get("mktCap")),
                    "price": _safe_float(p.get("price")),
                    "last_dividend": _safe_float(p.get("lastDiv")),
                    "dcf_diff": _safe_float(p.get("dcfDiff")),
                    "dcf": _safe_float(p.get("dcf")),
                }
            )
        return profile

    # ── Ratios ───────────────────────────────────────────

    async def _ratios(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 10,
        start_date: str | None = None,
        end_date: str | None = None,
        as_of_date: str | None = None,
        **_: Any,
    ) -> list[dict]:
        request_limit = _historical_limit(limit, start_date, end_date, as_of_date)
        data = await self._get("/ratios", {"symbol": symbol, "period": period, "limit": str(request_limit)})
        if not data:
            return []
        rows = _temporal_rows_or_marker(symbol, _as_list(data), start_date, end_date, as_of_date)
        if rows and rows[0].get("status") == "out_of_window":
            return rows
        return [
            {
                "symbol": r.get("symbol", symbol),
                "date": r.get("date", ""),
                "period": r.get("period", ""),
                "current_ratio": _safe_float(r.get("currentRatio")),
                "quick_ratio": _safe_float(r.get("quickRatio")),
                "debt_to_equity": _safe_float(r.get("debtToEquity")),
                "gross_profit_margin": _safe_float(r.get("grossProfitMargin")),
                "operating_profit_margin": _safe_float(r.get("operatingProfitMargin")),
                "net_profit_margin": _safe_float(r.get("netProfitMargin")),
                "return_on_assets": _safe_float(r.get("returnOnAssets")),
                "return_on_equity": _safe_float(r.get("returnOnEquity")),
                "asset_turnover": _safe_float(r.get("assetTurnover")),
                "receivables_turnover": _safe_float(r.get("receivablesTurnover")),
                "payables_turnover": _safe_float(r.get("payablesTurnover")),
                "inventory_turnover": _safe_float(r.get("inventoryTurnover")),
                "price_to_earnings": _safe_float(r.get("priceEarningsRatio")),
                "price_to_book": _safe_float(r.get("priceToBookRatio")),
                "price_to_sales": _safe_float(r.get("priceToSalesRatio")),
                "ev_to_ebitda": _safe_float(r.get("enterpriseValueToEBITDA")),
                "ev_to_revenue": _safe_float(r.get("evToRevenue")),
                "dividend_yield": _safe_float(r.get("dividendYield")),
                "payout_ratio": _safe_float(r.get("payoutRatio")),
            }
            for r in rows[:limit]
        ]

    async def _ratios_ttm(
        self,
        symbol: str,
        as_of_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        **_: Any,
    ) -> dict:
        # ratios-ttm endpoint not on stable; return {} to allow failover.
        if as_of_date or start_date or end_date:
            return _snapshot_blocked(symbol, as_of_date, start_date, end_date)
        data = await self._get("/ratios-ttm", {"symbol": symbol})
        rows = _as_list(data)
        if not rows:
            return {}
        r = rows[0]
        return {
            "symbol": r.get("symbol", symbol),
            "current_ratio_ttm": _safe_float(r.get("currentRatio")),
            "debt_to_equity_ttm": _safe_float(r.get("debtToEquity")),
            "gross_profit_margin_ttm": _safe_float(r.get("grossProfitMargin")),
            "net_profit_margin_ttm": _safe_float(r.get("netProfitMargin")),
            "return_on_assets_ttm": _safe_float(r.get("returnOnAssets")),
            "return_on_equity_ttm": _safe_float(r.get("returnOnEquity")),
            "price_to_earnings_ttm": _safe_float(r.get("priceEarningsRatio")),
            "price_to_book_ttm": _safe_float(r.get("priceToBookRatio")),
            "price_to_sales_ttm": _safe_float(r.get("priceToSalesRatio")),
            "ev_to_ebitda_ttm": _safe_float(r.get("enterpriseValueToEBITDA")),
            "dividend_yield_ttm": _safe_float(r.get("dividendYield")),
        }

    # ── Enterprise Value ─────────────────────────────────

    async def _enterprise_value(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 10,
        start_date: str | None = None,
        end_date: str | None = None,
        as_of_date: str | None = None,
        **_: Any,
    ) -> list[dict]:
        request_limit = _historical_limit(limit, start_date, end_date, as_of_date)
        data = await self._get("/enterprise-values", {"symbol": symbol, "period": period, "limit": str(request_limit)})
        if not data:
            return []
        rows = _temporal_rows_or_marker(symbol, _as_list(data), start_date, end_date, as_of_date)
        if rows and rows[0].get("status") == "out_of_window":
            return rows
        return [
            {
                "symbol": r.get("symbol", symbol),
                "date": r.get("date", ""),
                "market_cap": _safe_float(r.get("marketCapitalization")),
                "enterprise_value": _safe_float(r.get("enterpriseValue")),
                "cash_and_short_term": _safe_float(r.get("cashAndShortTermInvestments")),
                "total_debt": _safe_float(r.get("totalDebt")),
                "minority_interest": _safe_float(r.get("minorityInterest")),
                "preferred_stock": _safe_float(r.get("preferredStock")),
                "total_cash": _safe_float(r.get("cashAndCashEquivalents")),
            }
            for r in rows[:limit]
        ]

    # ── Financial Growth ─────────────────────────────────

    async def _growth(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 10,
        start_date: str | None = None,
        end_date: str | None = None,
        as_of_date: str | None = None,
        **_: Any,
    ) -> list[dict]:
        request_limit = _historical_limit(limit, start_date, end_date, as_of_date)
        data = await self._get("/financial-growth", {"symbol": symbol, "period": period, "limit": str(request_limit)})
        if not data:
            return []
        rows = _temporal_rows_or_marker(symbol, _as_list(data), start_date, end_date, as_of_date)
        if rows and rows[0].get("status") == "out_of_window":
            return rows
        return [
            {
                "symbol": r.get("symbol", symbol),
                "date": r.get("date", ""),
                "period": r.get("period", ""),
                "revenue_growth": _safe_float(r.get("revenueGrowth")),
                "gross_profit_growth": _safe_float(r.get("grossProfitGrowth")),
                "ebitgrowth": _safe_float(r.get("ebitgrowth")),
                "net_income_growth": _safe_float(r.get("netIncomeGrowth")),
                "eps_growth": _safe_float(r.get("epsgrowth")),
                "dividend_growth": _safe_float(r.get("dividendsGrowth")),
                "assets_growth": _safe_float(r.get("growthAssets")),
                "debt_growth": _safe_float(r.get("growthDebt")),
                "operating_income_growth": _safe_float(r.get("operatingIncomeGrowth")),
                "receivables_growth": _safe_float(r.get("growthReceivables")),
                "inventory_growth": _safe_float(r.get("growthInventory")),
            }
            for r in rows[:limit]
        ]

    # ── DCF Valuation ────────────────────────────────────

    async def _dcf(
        self,
        symbol: str,
        as_of_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        **_: Any,
    ) -> dict:
        if as_of_date or start_date or end_date:
            return _snapshot_blocked(symbol, as_of_date, start_date, end_date)
        data = await self._get("/discounted-cash-flow", {"symbol": symbol})
        rows = _as_list(data)
        if not rows:
            return {}
        r = rows[0]
        return {
            "symbol": r.get("symbol", symbol),
            "date": r.get("date", ""),
            "dcf": _safe_float(r.get("dcf")),
            "stock_price": _safe_float(r.get("Stock Price")),
            "dcf_diff_pct": _safe_float(r.get("dcfDiffPct")),
        }

    async def _dcf_historical(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 10,
        start_date: str | None = None,
        end_date: str | None = None,
        as_of_date: str | None = None,
        **_: Any,
    ) -> list[dict]:
        # historical-discounted-cash-flow endpoint not on stable — returns [].
        request_limit = _historical_limit(limit, start_date, end_date, as_of_date)
        data = await self._get("/historical-discounted-cash-flow", {"symbol": symbol, "period": period, "limit": str(request_limit)})
        if not data:
            return []
        rows = _temporal_rows_or_marker(symbol, _as_list(data), start_date, end_date, as_of_date)
        if rows and rows[0].get("status") == "out_of_window":
            return rows
        return [
            {
                "symbol": r.get("symbol", symbol),
                "date": r.get("date", ""),
                "dcf": _safe_float(r.get("dcf")),
                "stock_price": _safe_float(r.get("stockPrice")),
            }
            for r in rows[:limit]
        ]

    # ── Rating ───────────────────────────────────────────

    async def _rating(
        self,
        symbol: str,
        as_of_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        **_: Any,
    ) -> dict:
        if as_of_date or start_date or end_date:
            return _snapshot_blocked(symbol, as_of_date, start_date, end_date)
        data = await self._get("/ratings-snapshot", {"symbol": symbol})
        rows = _as_list(data)
        if not rows:
            return {}
        r = rows[0]
        return {
            "symbol": r.get("symbol", symbol),
            "date": r.get("date", ""),
            "rating_recommendation": r.get("ratingRecommendation", ""),
            "rating_score_dcf": _safe_float(r.get("ratingDetailsDCFScore")),
            "rating_score_roe": _safe_float(r.get("ratingDetailsROEScore")),
            "rating_score_roa": _safe_float(r.get("ratingDetailsROAScore")),
            "rating_score_de": _safe_float(r.get("ratingDetailsDEScore")),
            "rating_score_pe": _safe_float(r.get("ratingDetailsPEScore")),
            "rating_score_pb": _safe_float(r.get("ratingDetailsPBScore")),
        }

    async def _rating_historical(
        self,
        symbol: str,
        limit: int = 10,
        start_date: str | None = None,
        end_date: str | None = None,
        as_of_date: str | None = None,
        **_: Any,
    ) -> list[dict]:
        request_limit = _historical_limit(limit, start_date, end_date, as_of_date)
        data = await self._get(
            "/ratings-historical",
            {"symbol": symbol, "limit": str(request_limit)},
        )
        if not data:
            return []
        rows = _temporal_rows_or_marker(symbol, _as_list(data), start_date, end_date, as_of_date)
        if rows and rows[0].get("status") == "out_of_window":
            return rows
        return [
            {
                "symbol": r.get("symbol", symbol),
                "date": r.get("date", ""),
                "rating": "" if r.get("rating") is None else str(r.get("rating", "")),
                "rating_recommendation": (
                    "" if r.get("ratingRecommendation") is None else str(r.get("ratingRecommendation", ""))
                ),
            }
            for r in rows[:limit]
        ]

    # ── Key Metrics ──────────────────────────────────────

    async def _key_metrics(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 10,
        start_date: str | None = None,
        end_date: str | None = None,
        as_of_date: str | None = None,
        **_: Any,
    ) -> list[dict]:
        request_limit = _historical_limit(limit, start_date, end_date, as_of_date)
        data = await self._get("/key-metrics", {"symbol": symbol, "period": period, "limit": str(request_limit)})
        if not data:
            return []
        rows = _temporal_rows_or_marker(symbol, _as_list(data), start_date, end_date, as_of_date)
        if rows and rows[0].get("status") == "out_of_window":
            return rows
        return [
            {
                "symbol": r.get("symbol", symbol),
                "date": r.get("date", ""),
                "period": r.get("period", ""),
                "market_cap": _safe_float(r.get("marketCap")),
                "pe_ratio": _safe_float(r.get("peRatio")),
                "ps_ratio": _safe_float(r.get("psRatio")),
                "pb_ratio": _safe_float(r.get("pbRatio")),
                "ev_to_sales": _safe_float(r.get("evToSales")),
                "ev_to_ebitda": _safe_float(r.get("enterpriseValueOverEBITDA")),
                "ev_to_operating_cash_flow": _safe_float(r.get("evToOperatingCashFlow")),
                "ev_to_free_cash_flow": _safe_float(r.get("evToFreeCashFlow")),
                "earnings_yield": _safe_float(r.get("earningsYield")),
                "free_cash_flow_yield": _safe_float(r.get("freeCashFlowYield")),
                "debt_to_equity": _safe_float(r.get("debtToEquity")),
                "debt_to_assets": _safe_float(r.get("debtToAssets")),
                "net_debt_to_ebitda": _safe_float(r.get("netDebtToEBITDA")),
                "current_ratio": _safe_float(r.get("currentRatio")),
                "roe": _safe_float(r.get("roe")),
                "roa": _safe_float(r.get("roa")),
                "dividend_yield": _safe_float(r.get("dividendYield")),
                "payout_ratio": _safe_float(r.get("payoutRatio")),
                "revenue_per_share": _safe_float(r.get("revenuePerShare")),
                "net_income_per_share": _safe_float(r.get("netIncomePerShare")),
            }
            for r in rows[:limit]
        ]

    async def _key_metrics_ttm(
        self,
        symbol: str,
        limit: int = 10,
        as_of_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        **_: Any,
    ) -> dict:
        if as_of_date or start_date or end_date:
            return _snapshot_blocked(symbol, as_of_date, start_date, end_date)
        data = await self._get("/key-metrics-ttm", {"symbol": symbol, "limit": str(limit)})
        rows = _as_list(data)
        if not rows:
            return {}
        r = rows[0]
        return {
            "symbol": r.get("symbol", symbol),
            "pe_ratio_ttm": _safe_float(r.get("peRatio")),
            "ps_ratio_ttm": _safe_float(r.get("psRatio")),
            "pb_ratio_ttm": _safe_float(r.get("pbRatio")),
            "ev_to_sales_ttm": _safe_float(r.get("evToSales")),
            "ev_to_ebitda_ttm": _safe_float(r.get("enterpriseValueOverEBITDA")),
            "roe_ttm": _safe_float(r.get("roe")),
            "roa_ttm": _safe_float(r.get("roa")),
            "debt_to_equity_ttm": _safe_float(r.get("debtToEquity")),
            "dividend_yield_ttm": _safe_float(r.get("dividendYield")),
            "free_cash_flow_yield_ttm": _safe_float(r.get("freeCashFlowYield")),
        }

    # ── Ownership: Institutional Holders ─────────────────

    async def _institutional_holders(
        self,
        symbol: str,
        as_of_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        **_: Any,
    ) -> list[dict]:
        # institutional-holder endpoint not on stable — returns [].
        data = await self._get("/institutional-holder", {"symbol": symbol})
        if not data:
            return []
        rows = _temporal_rows_or_marker(
            symbol,
            _as_list(data),
            start_date,
            end_date,
            as_of_date,
            ("dateReported", "date"),
        )
        if rows and rows[0].get("status") == "out_of_window":
            return rows
        return [
            {
                "holder": r.get("holder", ""),
                "shares": _safe_int(r.get("shares")),
                "value_reported": _safe_float(r.get("value")),
                "date_reported": r.get("dateReported", ""),
                "change": _safe_int(r.get("change")),
            }
            for r in rows[:20]
        ]

    # ── Ownership: Mutual Fund Holders ───────────────────

    async def _mutual_fund_holders(
        self,
        symbol: str,
        as_of_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        **_: Any,
    ) -> list[dict]:
        # mutual-fund-holder endpoint not on stable — returns [].
        data = await self._get("/mutual-fund-holder", {"symbol": symbol})
        if not data:
            return []
        rows = _temporal_rows_or_marker(
            symbol,
            _as_list(data),
            start_date,
            end_date,
            as_of_date,
            ("dateReported", "date"),
        )
        if rows and rows[0].get("status") == "out_of_window":
            return rows
        return [
            {
                "holder": r.get("holder", ""),
                "shares": _safe_int(r.get("shares")),
                "value_reported": _safe_float(r.get("value")),
                "date_reported": r.get("dateReported", ""),
            }
            for r in rows[:20]
        ]

    # ── Ownership: Insider Trading ────────────────────────

    async def _insider_trading(
        self,
        symbol: str,
        limit: int = 10,
        start_date: str | None = None,
        end_date: str | None = None,
        as_of_date: str | None = None,
        **_: Any,
    ) -> list[dict]:
        request_limit = _historical_limit(limit, start_date, end_date, as_of_date)
        data = await self._get("/insider-trading", {"symbol": symbol, "limit": str(request_limit)})
        if not data:
            return []
        rows = _temporal_rows_or_marker(
            symbol,
            _as_list(data),
            start_date,
            end_date,
            as_of_date,
            ("transactionDate", "date"),
        )
        if rows and rows[0].get("status") == "out_of_window":
            return rows
        return [
            {
                "symbol": r.get("symbol", symbol),
                "insider_name": r.get("name", ""),
                "title": r.get("title", ""),
                "transaction_type": r.get("transactionType", ""),
                "transaction_date": r.get("transactionDate", ""),
                "securities_owned": _safe_int(r.get("securitiesOwned")),
                "securities_transacted": _safe_int(r.get("securitiesTransacted")),
                "price": _safe_float(r.get("price")),
                "type": r.get("type", ""),
            }
            for r in rows[:limit]
        ]

    # ── Ownership: Form 13F ──────────────────────────────

    async def _form_13f(self, cik_id: str, date: str | None = None, **_: Any) -> list[dict]:
        # form-thirteen endpoint not on stable — returns [].
        params: dict[str, Any] = {"cik": cik_id}
        if date:
            params["date"] = date
        data = await self._get("/form-thirteen", params)
        if not data:
            return []
        return [
            {
                "ticker": r.get("ticker", ""),
                "company_name": r.get("companyName", ""),
                "shares": _safe_int(r.get("shares")),
                "value": _safe_float(r.get("value")),
                "date_reported": r.get("dateReported", ""),
            }
            for r in _as_list(data)[:30]
        ]

    # ── Calendar: Earnings ───────────────────────────────

    async def _earning_calendar(
        self,
        from_date: str | None = None,
        to_date: str | None = None,
        as_of_date: str | None = None,
        **_: Any,
    ) -> list[dict]:
        # earnings-calendar endpoint is plan-limited (402) on stable.
        params: dict[str, Any] = {}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        data = await self._get("/earnings-calendar", params)
        if not data:
            return []
        rows = _temporal_rows_or_marker(None, _as_list(data), from_date, to_date, as_of_date)
        if rows and rows[0].get("status") == "out_of_window":
            return rows
        return [
            {
                "symbol": r.get("symbol", ""),
                "date": r.get("date", ""),
                "eps_estimated": _safe_float(r.get("epsEstimated")),
                "eps_reported": _safe_float(r.get("epsActual")) if r.get("epsActual") else None,
                "revenue_estimated": _safe_float(r.get("revenueEstimated")),
                "revenue_reported": _safe_float(r.get("revenueActual")) if r.get("revenueActual") else None,
                "time": r.get("time", ""),
                "exchange": r.get("exchangeShortName", ""),
            }
            for r in rows[:50]
        ]

    async def _earning_calendar_historical(
        self,
        symbol: str,
        limit: int = 10,
        from_date: str | None = None,
        to_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        as_of_date: str | None = None,
        **_: Any,
    ) -> list[dict]:
        # Use /earnings (per-symbol earnings history) on stable.
        window_start = start_date or from_date
        window_end = end_date or to_date
        request_limit = _historical_limit(limit, window_start, window_end, as_of_date)
        data = await self._get("/earnings", {"symbol": symbol, "limit": str(request_limit)})
        if not data:
            return []
        rows = _temporal_rows_or_marker(symbol, _as_list(data), window_start, window_end, as_of_date)
        if rows and rows[0].get("status") == "out_of_window":
            return rows
        return [
            {
                "symbol": r.get("symbol", symbol),
                "date": r.get("date", ""),
                "eps_estimated": _safe_float(r.get("epsEstimated")),
                "eps_reported": _safe_float(r.get("epsActual")) if r.get("epsActual") else None,
            }
            for r in rows[:limit]
        ]

    # ── Calendar: Earnings Surprises ─────────────────────

    async def _earnings_surprises(
        self,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
        as_of_date: str | None = None,
        **_: Any,
    ) -> list[dict]:
        # /earnings on stable returns epsActual + epsEstimated (no surprise pct).
        data = await self._get("/earnings", {"symbol": symbol})
        if not data:
            return []
        rows = _temporal_rows_or_marker(symbol, _as_list(data), start_date, end_date, as_of_date)
        if rows and rows[0].get("status") == "out_of_window":
            return rows
        return [
            {
                "symbol": r.get("symbol", symbol),
                "date": r.get("date", ""),
                "eps_estimated": _safe_float(r.get("epsEstimated")),
                "eps_reported": _safe_float(r.get("epsActual")),
                "eps_surprise": _safe_float(r.get("epsSurprise")),
                "eps_surprise_pct": _safe_float(r.get("epsSurprisePercent")),
                "revenue_estimated": _safe_float(r.get("revenueEstimated")),
                "revenue_reported": _safe_float(r.get("revenueActual")),
            }
            for r in rows[:20]
        ]

    # ── Calendar: Dividends ──────────────────────────────

    async def _dividend_calendar(
        self,
        from_date: str | None = None,
        to_date: str | None = None,
        as_of_date: str | None = None,
        **_: Any,
    ) -> list[dict]:
        # stock_dividend_calendar not on stable; use /dividends?symbol=...
        # Caller must pass a symbol via kwargs; otherwise returns [].
        params: dict[str, Any] = {}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        data = await self._get("/dividends", params)
        if not data:
            return []
        rows = _temporal_rows_or_marker(None, _as_list(data), from_date, to_date, as_of_date)
        if rows and rows[0].get("status") == "out_of_window":
            return rows
        return [
            {
                "symbol": r.get("symbol", ""),
                "date": r.get("date", ""),
                "label": r.get("label", ""),
                "adj_dividend": _safe_float(r.get("adjDividend")),
                "dividend": _safe_float(r.get("dividend")),
                "record_date": r.get("recordDate", ""),
                "payment_date": r.get("paymentDate", ""),
                "declaration_date": r.get("declarationDate", ""),
            }
            for r in rows[:50]
        ]

    async def _dividend_historical(
        self,
        symbol: str,
        from_date: str | None = None,
        to_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        as_of_date: str | None = None,
        **_: Any,
    ) -> list[dict]:
        # Stable API: /dividends?symbol=AAPL returns a flat list of dividend
        # records (no nested historicalStockList wrapper).
        data = await self._get("/dividends", {"symbol": symbol})
        if not data:
            return []
        window_start = start_date or from_date
        window_end = end_date or to_date
        items = _temporal_rows_or_marker(
            symbol,
            _as_list(data),
            window_start,
            window_end,
            as_of_date,
        )
        if items and items[0].get("status") == "out_of_window":
            return items
        return [
            {
                "symbol": symbol,
                "date": r.get("date", ""),
                "label": r.get("label", ""),
                "adj_dividend": _safe_float(r.get("adjDividend")),
                "dividend": _safe_float(r.get("dividend")),
                "record_date": r.get("recordDate", ""),
                "payment_date": r.get("paymentDate", ""),
                "declaration_date": r.get("declarationDate", ""),
            }
            for r in items[:50]
        ]

    # ── Calendar: IPO ────────────────────────────────────

    async def _ipo_calendar(
        self,
        from_date: str | None = None,
        to_date: str | None = None,
        as_of_date: str | None = None,
        **_: Any,
    ) -> list[dict]:
        # ipo_calendar not on stable — returns [].
        params: dict[str, Any] = {}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        data = await self._get("/ipo_calendar", params)
        if not data:
            return []
        rows = _temporal_rows_or_marker(None, _as_list(data), from_date, to_date, as_of_date)
        if rows and rows[0].get("status") == "out_of_window":
            return rows
        return [
            {
                "symbol": r.get("symbol", ""),
                "company_name": r.get("companyName", ""),
                "date": r.get("date", ""),
                "exchange": r.get("exchange", ""),
                "price": _safe_float(r.get("price")),
                "shares": _safe_int(r.get("shares")),
            }
            for r in rows[:50]
        ]

    # ── Calendar: Stock Splits ───────────────────────────

    async def _stock_split_calendar(
        self,
        symbol: str | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
        as_of_date: str | None = None,
        **_: Any,
    ) -> list[dict]:
        # stock_split_calendar not on stable; use /splits (needs symbol).
        params: dict[str, Any] = {}
        if symbol:
            params["symbol"] = symbol
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        data = await self._get("/splits", params)
        if not data:
            return []
        rows = _temporal_rows_or_marker(symbol, _as_list(data), from_date, to_date, as_of_date)
        if rows and rows[0].get("status") == "out_of_window":
            return rows
        return [
            {
                "symbol": r.get("symbol", symbol or ""),
                "date": r.get("date", ""),
                "label": r.get("label", ""),
                "numerator": _safe_float(r.get("numerator")),
                "denominator": _safe_float(r.get("denominator")),
            }
            for r in rows[:50]
        ]

    # ── Calendar: Economic Calendar ──────────────────────

    async def _economic_calendar(
        self,
        from_date: str | None = None,
        to_date: str | None = None,
        as_of_date: str | None = None,
        **_: Any,
    ) -> list[dict]:
        # economic-calendar is plan-limited (402) on stable.
        params: dict[str, Any] = {}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        data = await self._get("/economic-calendar", params)
        if not data:
            return []
        rows = _temporal_rows_or_marker(None, _as_list(data), from_date, to_date, as_of_date)
        if rows and rows[0].get("status") == "out_of_window":
            return rows
        return [
            {
                "event": r.get("event", ""),
                "date": r.get("date", ""),
                "country": r.get("country", ""),
                "actual": r.get("actual", ""),
                "previous": r.get("previous", ""),
                "estimate": r.get("estimate", ""),
                "change": r.get("change", ""),
                "change_pct": r.get("changePct", ""),
                "importance": r.get("importance", ""),
            }
            for r in rows[:50]
        ]

    # ── Market Cap ───────────────────────────────────────

    async def _market_cap(
        self,
        symbol: str,
        as_of_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        **_: Any,
    ) -> dict:
        # market-capitalization endpoint not on stable — returns {}.
        # Use /quote (which includes marketCap) as a fallback.
        if as_of_date or start_date or end_date:
            return _snapshot_blocked(symbol, as_of_date, start_date, end_date)
        data = await self._get("/market-capitalization", {"symbol": symbol})
        rows = _as_list(data)
        if not rows:
            return {}
        r = rows[0]
        return {
            "symbol": r.get("symbol", symbol),
            "market_cap": _safe_float(r.get("marketCap")),
            "date": r.get("date", ""),
        }

    async def _market_cap_historical(
        self,
        symbol: str,
        limit: int = 10,
        start_date: str | None = None,
        end_date: str | None = None,
        as_of_date: str | None = None,
        **_: Any,
    ) -> list[dict]:
        request_limit = _historical_limit(limit, start_date, end_date, as_of_date)
        data = await self._get("/historical-market-capitalization", {"symbol": symbol, "limit": str(request_limit)})
        if not data:
            return []
        rows = _temporal_rows_or_marker(symbol, _as_list(data), start_date, end_date, as_of_date)
        if rows and rows[0].get("status") == "out_of_window":
            return rows
        return [
            {
                "symbol": r.get("symbol", symbol),
                "date": r.get("date", ""),
                "market_cap": _safe_float(r.get("marketCap")),
            }
            for r in rows[:limit]
        ]

    # ── ETF Weightings ───────────────────────────────────

    async def _etf_sector_weightings(self, symbol: str, **_: Any) -> list[dict]:
        # etf-sector-weightings endpoint not on stable — returns [].
        data = await self._get("/etf-sector-weightings", {"symbol": symbol})
        if not data:
            return []
        return [
            {
                "sector": r.get("sector", ""),
                "weight_pct": _safe_float(r.get("weightPercentage")),
            }
            for r in _as_list(data)[:20]
        ]

    async def _etf_country_weightings(self, symbol: str, **_: Any) -> list[dict]:
        # etf-country-weightings endpoint not on stable — returns [].
        data = await self._get("/etf-country-weightings", {"symbol": symbol})
        if not data:
            return []
        return [
            {
                "country": r.get("country", ""),
                "weight_pct": _safe_float(r.get("weightPercentage")),
            }
            for r in _as_list(data)[:20]
        ]

    # ── SEC Filings ──────────────────────────────────────

    async def _sec_filings(
        self,
        symbol: str,
        filing_type: str = "",
        limit: int = 10,
        start_date: str | None = None,
        end_date: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        as_of_date: str | None = None,
        **_: Any,
    ) -> list[dict]:
        # sec_filings endpoint not on stable — returns [].
        # Caller (fetch_with_failover) falls back to sec_edgar connector.
        window_start = start_date or date_from
        window_end = end_date or date_to
        request_limit = _historical_limit(limit, window_start, window_end, as_of_date)
        params: dict[str, Any] = {"symbol": symbol, "limit": str(request_limit)}
        if filing_type:
            params["type"] = filing_type
        data = await self._get("/sec_filings", params)
        if not data:
            return []
        rows = _temporal_rows_or_marker(
            symbol,
            _as_list(data),
            window_start,
            window_end,
            as_of_date,
            ("fillingDate", "acceptedDate", "date"),
        )
        if rows and rows[0].get("status") == "out_of_window":
            return rows
        return [
            {
                "symbol": r.get("symbol", symbol),
                "filing_type": r.get("type", ""),
                "filed_date": r.get("fillingDate", ""),
                "accepted_date": r.get("acceptedDate", ""),
                "period": r.get("periodOfReport", ""),
                "final_link": r.get("finalLink", ""),
                "description": r.get("description", ""),
            }
            for r in rows[:limit]
        ]

    # ── Treasury / Yield Curve ──────────────────────────

    async def _treasury(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        **_: Any,
    ) -> list[dict]:
        # Stable API: /treasury-rates?from=...&to=...
        params: dict[str, Any] = {}
        if start_date:
            params["from"] = start_date
        if end_date:
            params["to"] = end_date
        data = await self._get("/treasury-rates", params)
        if not data:
            return []
        return [
            {
                "date": r.get("date", ""),
                "month_1": _safe_float(r.get("month1")),
                "month_2": _safe_float(r.get("month2")),
                "month_3": _safe_float(r.get("month3")),
                "month_6": _safe_float(r.get("month6")),
                "year_1": _safe_float(r.get("year1")),
                "year_2": _safe_float(r.get("year2")),
                "year_3": _safe_float(r.get("year3")),
                "year_5": _safe_float(r.get("year5")),
                "year_7": _safe_float(r.get("year7")),
                "year_10": _safe_float(r.get("year10")),
                "year_20": _safe_float(r.get("year20")),
                "year_30": _safe_float(r.get("year30")),
            }
            for r in _as_list(data)[:100]
        ]

    # ── Analyst Estimates ───────────────────────────────

    async def _analyst_estimates(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 10,
        start_date: str | None = None,
        end_date: str | None = None,
        as_of_date: str | None = None,
        **_: Any,
    ) -> list[dict]:
        request_limit = _historical_limit(limit, start_date, end_date, as_of_date)
        params: dict[str, Any] = {"symbol": symbol, "period": period, "limit": str(request_limit)}
        data = await self._get("/analyst-estimates", params)
        if not data:
            return []
        rows = _temporal_rows_or_marker(symbol, _as_list(data), start_date, end_date, as_of_date)
        if rows and rows[0].get("status") == "out_of_window":
            return rows
        return [
            {
                "symbol": r.get("symbol", symbol),
                "date": r.get("date", ""),
                "estimated_eps_avg": _safe_float(r.get("epsAvg")),
                "estimated_eps_low": _safe_float(r.get("epsLow")),
                "estimated_eps_high": _safe_float(r.get("epsHigh")),
                "estimated_revenue_avg": _safe_float(r.get("revenueAvg")),
                "estimated_revenue_low": _safe_float(r.get("revenueLow")),
                "estimated_revenue_high": _safe_float(r.get("revenueHigh")),
                "number_of_analysts": _safe_int(r.get("numAnalystsRevenue") or r.get("numAnalystsEps")),
            }
            for r in rows[:limit]
        ]

    # ── Price Target Consensus ──────────────────────────

    async def _price_target(
        self,
        symbol: str,
        as_of_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        **_: Any,
    ) -> list[dict] | dict:
        if as_of_date or start_date or end_date:
            return _snapshot_blocked(symbol, as_of_date, start_date, end_date)
        data = await self._get("/price-target-summary", {"symbol": symbol})
        if not data:
            return []
        return [
            {
                "symbol": r.get("symbol", symbol),
                "target_consensus": _safe_float(r.get("targetConsensus")),
                "target_median": _safe_float(r.get("targetMedian")),
                "target_high": _safe_float(r.get("targetHigh")),
                "target_low": _safe_float(r.get("targetLow")),
                "analyst_count": _safe_int(r.get("analystCount")),
            }
            for r in (data if isinstance(data, list) else [])[:20]
        ]

    # ── Earnings Call Transcript ────────────────────────

    async def _earnings_transcript(
        self,
        symbol: str,
        quarter: int | None = None,
        year: int | None = None,
        **_: Any,
    ) -> list[dict]:
        # earning-call-transcript is plan-limited (402) on stable.
        params: dict[str, Any] = {"symbol": symbol}
        if quarter:
            params["quarter"] = str(quarter)
        if year:
            params["year"] = str(year)
        data = await self._get("/earning-call-transcript", params)
        if not data:
            return []
        return [
            {
                "symbol": r.get("symbol", symbol),
                "quarter": _safe_int(r.get("quarter")),
                "year": _safe_int(r.get("year")),
                "date": r.get("date", ""),
                "content": r.get("content", ""),
            }
            for r in _as_list(data)[:5]
        ]

    # ── ESG Scores ─────────────────────────────────────

    async def _esg(
        self,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
        as_of_date: str | None = None,
        **_: Any,
    ) -> list[dict]:
        # esg-score not on stable; esg-ratings is plan-limited (402).
        data = await self._get("/esg-ratings", {"symbol": symbol})
        if not data:
            return []
        rows = _temporal_rows_or_marker(symbol, _as_list(data), start_date, end_date, as_of_date)
        if rows and rows[0].get("status") == "out_of_window":
            return rows
        return [
            {
                "symbol": r.get("symbol", symbol),
                "date": r.get("date", ""),
                "environmental_score": _safe_float(r.get("environmentalScore")),
                "social_score": _safe_float(r.get("socialScore")),
                "governance_score": _safe_float(r.get("governanceScore")),
                "esg_score": _safe_float(r.get("ESGScore") or r.get("esgScore")),
            }
            for r in rows[:20]
        ]

    # ── Altman Z-Score ─────────────────────────────────

    async def _altman_z(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 10,
        start_date: str | None = None,
        end_date: str | None = None,
        as_of_date: str | None = None,
        **_: Any,
    ) -> list[dict]:
        # altman-z-score endpoint not on stable — returns [].
        request_limit = _historical_limit(limit, start_date, end_date, as_of_date)
        data = await self._get("/altman-z-score", {"symbol": symbol, "period": period, "limit": str(request_limit)})
        if not data:
            return []
        rows = _temporal_rows_or_marker(symbol, _as_list(data), start_date, end_date, as_of_date)
        if rows and rows[0].get("status") == "out_of_window":
            return rows
        results = []
        for r in rows[:limit]:
            if "altmanZScore" not in r:
                continue
            results.append({
                "symbol": r.get("symbol", symbol),
                "date": r.get("date", ""),
                "altman_z_score": _safe_float(r.get("altmanZScore")),
            })
        return results

    # ── Piotroski F-Score ──────────────────────────────

    async def _piotroski(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 10,
        start_date: str | None = None,
        end_date: str | None = None,
        as_of_date: str | None = None,
        **_: Any,
    ) -> list[dict]:
        # piotroski-score endpoint not on stable — returns [].
        request_limit = _historical_limit(limit, start_date, end_date, as_of_date)
        data = await self._get("/piotroski-score", {"symbol": symbol, "period": period, "limit": str(request_limit)})
        if not data:
            return []
        rows = _temporal_rows_or_marker(symbol, _as_list(data), start_date, end_date, as_of_date)
        if rows and rows[0].get("status") == "out_of_window":
            return rows
        return [
            {
                "symbol": r.get("symbol", symbol),
                "date": r.get("date", ""),
                "piotroski_score": _safe_float(r.get("piotroskiScore")),
            }
            for r in rows[:limit]
        ]

    # ── WACC ────────────────────────────────────────────

    async def _wacc(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 10,
        start_date: str | None = None,
        end_date: str | None = None,
        as_of_date: str | None = None,
        **_: Any,
    ) -> list[dict]:
        # wacc endpoint not on stable — returns [].
        request_limit = _historical_limit(limit, start_date, end_date, as_of_date)
        data = await self._get("/wacc", {"symbol": symbol, "period": period, "limit": str(request_limit)})
        if not data:
            return []
        rows = _temporal_rows_or_marker(symbol, _as_list(data), start_date, end_date, as_of_date)
        if rows and rows[0].get("status") == "out_of_window":
            return rows
        results = []
        for r in rows[:limit]:
            results.append({
                "symbol": r.get("symbol", symbol),
                "date": r.get("date", ""),
                "wacc": _safe_float(r.get("wacc")),
                "cost_of_equity": _safe_float(r.get("costOfEquity")),
                "cost_of_debt": _safe_float(r.get("costOfDebt")),
            })
        return results

    # ── Commodity Data ──────────────────────────────────

    async def _commodity(self, symbol: str | None = None, **_: Any) -> list[dict]:
        # commodity endpoints not on stable — returns [].
        if symbol:
            data = await self._get("/commodity", {"symbol": symbol.upper()})
        else:
            data = await self._get("/commodity")
        if not data:
            return []
        return [
            {
                "symbol": r.get("symbol", ""),
                "name": r.get("name", r.get("symbol", "")),
                "price": _safe_float(r.get("price")),
                "change": _safe_float(r.get("change")),
                "change_pct": _safe_float(r.get("changesPercentage")),
            }
            for r in _as_list(data)[:50]
        ]

    async def _commodity_historical(
        self,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
        **_: Any,
    ) -> list[dict]:
        # Use /historical-price-eod/full (stable) for commodity EOD history.
        params: dict[str, Any] = {"symbol": symbol.upper()}
        if start_date:
            params["from"] = start_date
        if end_date:
            params["to"] = end_date
        data = await self._get("/historical-price-eod/full", params)
        if not data:
            return []
        return [
            {
                "symbol": symbol,
                "date": r.get("date", ""),
                "open": _safe_float(r.get("open")),
                "high": _safe_float(r.get("high")),
                "low": _safe_float(r.get("low")),
                "close": _safe_float(r.get("close")),
                "volume": _safe_int(r.get("volume")),
            }
            for r in _as_list(data)[:200]
        ]

    # ── Forex Data ─────────────────────────────────────

    async def _forex(self, symbol: str | None = None, **_: Any) -> list[dict]:
        # forex endpoint not on stable — returns [].
        params: dict[str, Any] = {}
        if symbol:
            params["symbol"] = symbol.upper()
        data = await self._get("/forex", params)
        if not data:
            return []
        return [
            {
                "symbol": r.get("symbol", ""),
                "name": r.get("name", r.get("symbol", "")),
                "price": _safe_float(r.get("price")),
                "change": _safe_float(r.get("change")),
                "change_pct": _safe_float(r.get("changesPercentage")),
            }
            for r in _as_list(data)[:50]
        ]

    async def _forex_historical(
        self,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
        **_: Any,
    ) -> list[dict]:
        # Use /historical-price-eod/full (stable) for forex EOD history.
        params: dict[str, Any] = {"symbol": symbol.upper()}
        if start_date:
            params["from"] = start_date
        if end_date:
            params["to"] = end_date
        data = await self._get("/historical-price-eod/full", params)
        if not data:
            return []
        return [
            {
                "symbol": symbol,
                "date": r.get("date", ""),
                "open": _safe_float(r.get("open")),
                "high": _safe_float(r.get("high")),
                "low": _safe_float(r.get("low")),
                "close": _safe_float(r.get("close")),
            }
            for r in _as_list(data)[:200]
        ]

    async def health_check(self) -> bool:
        if not self._api_key:
            return False
        try:
            data = await self._get("/quote", {"symbol": "AAPL"})
            return bool(data)
        except Exception:
            return False


register_connector("fmp", FMPConnector)
