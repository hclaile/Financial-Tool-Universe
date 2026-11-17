"""Alpha Vantage connector — fundamentals (Company Overview, Income Statement,
Balance Sheet, Cash Flow) and News & Sentiments."""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from .base import BaseConnector, register_connector
from ._numeric import safe_float, safe_int

logger = logging.getLogger("financial_tool.connector.alphavantage")

_BASE_URL = "https://www.alphavantage.co/query"


def _select_report_in_window(
    reports: list[dict[str, Any]],
    start_date: str | None = None,
    end_date: str | None = None,
    as_of_date: str | None = None,
) -> dict[str, Any]:
    for report in reports:
        date = str(report.get("fiscalDateEnding", ""))[:10]
        if not date:
            continue
        if start_date and date < start_date:
            continue
        if end_date and date > end_date:
            continue
        if as_of_date and date > as_of_date:
            continue
        return report
    return {}


class AlphaVantageConnector(BaseConnector):
    name = "alphavantage"
    actions = frozenset({"income", "balance", "cashflow", "profile", "headlines", "sentiment", "search"})

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        token_env = self._config.get("token_env", "ALPHAVANTAGE_API_KEY")
        api_key = os.getenv(token_env, "") or self._config.get("api_key", "")
        self._api_key = api_key

    async def _get(self, params: dict[str, Any]) -> Any:
        params["apikey"] = self._api_key
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(_BASE_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
        # Alpha Vantage returns error info in the body
        if "Error Message" in data:
            raise ValueError(f"AlphaVantage error: {data['Error Message']}")
        if "Note" in data:
            raise ValueError(f"AlphaVantage rate limit: {data['Note']}")
        if "Information" in data:
            raise ValueError(f"AlphaVantage info: {data['Information']}")
        return data

    async def fetch(self, action: str, **kwargs: Any) -> Any:
        dispatch = {
            "income": self._income,
            "balance": self._balance,
            "cashflow": self._cashflow,
            "profile": self._profile,
            "headlines": self._headlines,
            "sentiment": self._sentiment,
            "search": self._search,
        }
        handler = dispatch.get(action)
        if handler is None:
            raise ValueError(f"AlphaVantage: unsupported action '{action}'")
        return await handler(**kwargs)

    # ── Fundamentals ──────────────────────────────────────

    async def _profile(self, symbol: str, **_: Any) -> dict:
        """Company Overview — full company profile from Alpha Vantage."""
        if not self._api_key:
            return {}
        data = await self._get({"function": "OVERVIEW", "symbol": symbol})
        if not data or "Symbol" not in data:
            return {}
        return {
            "symbol": data.get("Symbol", symbol),
            "company_name": data.get("Name", ""),
            "exchange": data.get("Exchange", ""),
            "currency": data.get("Currency", ""),
            "sector": data.get("Sector", ""),
            "industry": data.get("Industry", ""),
            "description": data.get("Description", ""),
            "ceo": data.get("CEO", ""),
            "country": data.get("Country", ""),
            "market_cap": safe_float(data.get("MarketCapitalization")),
            "pe_ratio": safe_float(data.get("PERatio")),
            "peg_ratio": safe_float(data.get("PEGRatio")),
            "book_value": safe_float(data.get("BookValue")),
            "dividend_per_share": safe_float(data.get("DividendPerShare")),
            "dividend_yield": safe_float(data.get("DividendYield")),
            "eps": safe_float(data.get("EPS")),
            "revenue_ttm": safe_float(data.get("RevenueTTM")),
            "gross_profit_ttm": safe_float(data.get("GrossProfitTTM")),
            "quarterly_earnings_growth_yoy": safe_float(data.get("QuarterlyEarningsgrowthYOY")),
            "quarterly_revenue_growth_yoy": safe_float(data.get("QuarterlyRevenuegrowthYOY")),
            "analyst_target_price": safe_float(data.get("AnalystTargetPrice")),
            "trailing_pe": safe_float(data.get("TrailingPE")),
            "forward_pe": safe_float(data.get("ForwardPE")),
            "price_to_sales_ttm": safe_float(data.get("PriceToSalesRatioTTM")),
            "price_to_book": safe_float(data.get("PriceToBookRatio")),
            "ev_to_revenue": safe_float(data.get("EVToRevenue")),
            "ev_to_ebitda": safe_float(data.get("EVToEBITDA")),
            "beta": safe_float(data.get("Beta")),
            "52week_high": safe_float(data.get("52WeekHigh")),
            "52week_low": safe_float(data.get("52WeekLow")),
            "50day_ma": safe_float(data.get("50DayMovingAverage")),
            "200day_ma": safe_float(data.get("200DayMovingAverage")),
            "shares_outstanding": safe_int(data.get("SharesOutstanding")),
            "payout_ratio": safe_float(data.get("PayoutRatio")),
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
        """Income Statement from Alpha Vantage."""
        if not self._api_key:
            return {}
        func = "INCOME_STATEMENT"
        data = await self._get({"function": func, "symbol": symbol})
        reports = data.get("annualReports" if period == "annual" else "quarterlyReports", [])
        if not reports:
            return {}
        r = _select_report_in_window(reports, start_date, end_date, as_of_date) or reports[0]
        return {
            "symbol": symbol,
            "date": r.get("fiscalDateEnding", ""),
            "period": period,
            "revenue": safe_float(r.get("totalRevenue")),
            "cogs": safe_float(r.get("costofGoodsAndServicesSold")),
            "gross_profit": safe_float(r.get("grossProfit")),
            "operating_income": safe_float(r.get("operatingIncome")),
            "ebitda": safe_float(r.get("ebitda")),
            "net_income": safe_float(r.get("netIncome")),
            "eps": safe_float(r.get("eps")),
            "interest_expense": safe_float(r.get("interestExpense")),
            "depreciation": safe_float(r.get("depreciationAndAmortization")),
            "rd_expense": safe_float(r.get("researchAndDevelopment")),
            "sga_expense": safe_float(r.get("sellingGeneralAndAdministrative")),
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
        """Balance Sheet from Alpha Vantage."""
        if not self._api_key:
            return {}
        data = await self._get({"function": "BALANCE_SHEET", "symbol": symbol})
        reports = data.get("annualReports" if period == "annual" else "quarterlyReports", [])
        if not reports:
            return {}
        r = _select_report_in_window(reports, start_date, end_date, as_of_date) or reports[0]
        return {
            "symbol": symbol,
            "date": r.get("fiscalDateEnding", ""),
            "period": period,
            "total_assets": safe_float(r.get("totalAssets")),
            "total_current_assets": safe_float(r.get("totalCurrentAssets")),
            "cash": safe_float(r.get("cashAndCashEquivalentsAtCarryingValue")),
            "short_term_investments": safe_float(r.get("shortTermInvestments")),
            "net_receivables": safe_float(r.get("currentNetReceivables")),
            "inventory": safe_float(r.get("inventory")),
            "total_liabilities": safe_float(r.get("totalLiabilities")),
            "total_current_liabilities": safe_float(r.get("totalCurrentLiabilities")),
            "long_term_debt": safe_float(r.get("longTermDebt")),
            "total_equity": safe_float(r.get("totalShareholderEquity")),
            "retained_earnings": safe_float(r.get("retainedEarnings")),
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
        """Cash Flow Statement from Alpha Vantage."""
        if not self._api_key:
            return {}
        data = await self._get({"function": "CASH_FLOW", "symbol": symbol})
        reports = data.get("annualReports" if period == "annual" else "quarterlyReports", [])
        if not reports:
            return {}
        r = _select_report_in_window(reports, start_date, end_date, as_of_date) or reports[0]
        return {
            "symbol": symbol,
            "date": r.get("fiscalDateEnding", ""),
            "period": period,
            "operating_cf": safe_float(r.get("operatingCashflow")),
            "depreciation": safe_float(r.get("depreciationDepletionAndAmortization")),
            "change_in_receivables": safe_float(r.get("changeInReceivables")),
            "change_in_inventory": safe_float(r.get("changeInInventory")),
            "investing_cf": safe_float(r.get("cashflowFromInvestment")),
            "capital_expenditures": safe_float(r.get("capitalExpenditures")),
            "financing_cf": safe_float(r.get("cashflowFromFinancing")),
            "dividend_payout": safe_float(r.get("dividendPayout")),
            "net_income": safe_float(r.get("netIncome")),
        }

    # ── News & Sentiments ─────────────────────────────────

    async def _sentiment(self, symbol: str, **_: Any) -> dict:
        """Return the compatible subset available from NEWS_SENTIMENT."""
        if not self._api_key:
            return {}
        data = await self._get({"function": "NEWS_SENTIMENT", "tickers": symbol})
        feed = data.get("feed", []) if isinstance(data, dict) else []
        rows = [
            row
            for article in feed
            for row in article.get("ticker_sentiment", [])
            if str(row.get("ticker", "")).upper() == symbol.upper()
        ]
        if not feed:
            return {}
        scores = [safe_float(row.get("ticker_sentiment_score")) for row in rows]
        return {
            "symbol": symbol,
            "source": "alphavantage",
            "data_type": "sentiment",
            "coverage": "article_sentiment_fallback",
            "articles_retrieved": len(feed),
            "company_news_score": sum(scores) / len(scores) if scores else 0.0,
        }

    async def _headlines(
        self,
        query: str | None = None,
        symbols: list[str] | None = None,
        limit: int = 20,
        **_: Any,
    ) -> list[dict]:
        """News & Sentiments from Alpha Vantage.

        Alpha Vantage NEWS_SENTIMENT endpoint accepts optional 'tickers' and
        'topics' parameters, plus time_from/time_to for date filtering.

        Note: AlphaVantage only supports STOCK tickers in the 'tickers'
        parameter.  Forex/commodity/crypto symbols (XAUUSD, BTC-USD, GC=F)
        will cause malformed responses or TypeError.  These are filtered
        out before the API call.
        """
        if not self._api_key:
            return []

        # ── Filter out non-stock tickers ──────────────────────
        # AlphaVantage NEWS_SENTIMENT only accepts stock tickers.
        # Forex (EUR/USD, XAUUSD), commodity (GC=F), and crypto
        # (BTC-USD) symbols must be excluded to prevent TypeError
        # from malformed API responses.
        filtered_symbols: list[str] | None = None
        if symbols:
            filtered_symbols = [
                s for s in symbols
                if not any(c in s for c in ("/", "-", "="))
                and not s.endswith("=X")
                and not s.endswith("=F")
                and not s.upper().startswith(("XAU", "XAG", "BTC", "ETH", "SOL"))
            ]
            if not filtered_symbols:
                filtered_symbols = None  # Drop tickers param; use query only

        params: dict[str, Any] = {"function": "NEWS_SENTIMENT"}
        if filtered_symbols:
            params["tickers"] = ",".join(filtered_symbols)
        if query:
            # Alpha Vantage uses 'topics' for keyword filtering
            params["topics"] = query
        params["limit"] = str(min(limit, 50))

        try:
            data = await self._get(params)
        except TypeError as exc:
            # AlphaVantage can return malformed data with mixed str/float
            # fields when non-stock tickers slip through or when the API
            # response format changes.  Return [] to allow failover.
            logger.warning(
                "AlphaVantage _headlines TypeError: %s. Returning [].", exc,
            )
            return []
        feed = data.get("feed", [])
        return [
            {
                "title": a.get("title", ""),
                "source": a.get("source", ""),
                "published_at": a.get("time_published", ""),
                "url": a.get("url", ""),
                "summary": a.get("summary", ""),
                "sentiment_score": safe_float(a.get("overall_sentiment_score")),
                "sentiment_label": a.get("overall_sentiment_rating", ""),
                "ticker_sentiments": a.get("ticker_sentiment", []),
                "symbols": [ts.get("ticker", "") for ts in a.get("ticker_sentiment", [])],
            }
            for a in feed[:limit]
        ]

    async def _search(self, query: str, limit: int = 10, **_: Any) -> list[dict]:
        headlines = await self._headlines(query=query, limit=limit)
        return [
            {
                "query": query,
                "source": "alphavantage",
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
            await self._get({"function": "OVERVIEW", "symbol": "IBM"})
            return True
        except Exception:
            return False


register_connector("alphavantage", AlphaVantageConnector)
