"""Tushare connector — A-share market data, fundamentals, macro."""

from __future__ import annotations

import os
from typing import Any

import httpx

from .base import BaseConnector, register_connector
from ._numeric import safe_float, safe_int

_API_URL = "https://api.tushare.pro"


class TushareConnector(BaseConnector):
    name = "tushare"
    actions = frozenset({"quote", "ohlcv", "realtime_quote", "income", "balance", "cashflow", "series", "headlines", "info", "money_flow", "search"})

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._token = self._config.get("token", "") or os.getenv("TUSHARE_TOKEN", "")

    async def _request(self, api_name: str, fields: str = "", **params: Any) -> Any:
        payload = {
            "api_name": api_name,
            "token": self._token,
            "params": {k: v for k, v in params.items() if v is not None},
            "fields": fields,
        }
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(_API_URL, json=payload)
            resp.raise_for_status()
            data = resp.json()
        if data.get("code") != 0:
            raise RuntimeError(f"Tushare error: {data.get('msg', 'unknown')}")
        return data.get("data", {})

    async def fetch(self, action: str, **kwargs: Any) -> Any:
        dispatch = {
            "quote": self._quote,
            "ohlcv": self._ohlcv,
            "realtime_quote": self._realtime,
            "income": self._income,
            "balance": self._balance,
            "cashflow": self._cashflow,
            "series": self._macro,
            "headlines": self._news,
            "info": self._fund_info,
            "money_flow": self._money_flow,
            "search": self._search,
        }
        handler = dispatch.get(action)
        if handler is None:
            raise ValueError(f"Tushare: unsupported action '{action}'")
        return await handler(**kwargs)

    # ── Market data ───────────────────────────────────────

    async def _quote(self, symbol: str, exchange: str | None = None, **_: Any) -> dict:
        fields = "ts_code,open,high,low,close,vol,change,pct_chg,trade_date"
        data = await self._request("daily", fields=fields, ts_code=symbol)
        rows = data.get("rows", [])
        if not rows:
            return {}
        item = dict(zip(data.get("fields", []), rows[-1]))
        return {
            "symbol": item.get("ts_code", symbol),
            "open": safe_float(item.get("open")),
            "high": safe_float(item.get("high")),
            "low": safe_float(item.get("low")),
            "close": safe_float(item.get("close")),
            "volume": safe_int(item.get("vol")),
            "change": safe_float(item.get("change")),
            "change_pct": safe_float(item.get("pct_chg")),
            "timestamp": item.get("trade_date", ""),
        }

    async def _ohlcv(
        self,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
        interval: str = "1d",
        **_: Any,
    ) -> list[dict]:
        fields = "ts_code,trade_date,open,high,low,close,vol"
        data = await self._request(
            "daily", fields=fields, ts_code=symbol,
            start_date=start_date, end_date=end_date,
        )
        rows = data.get("rows", [])
        fields_list = data.get("fields", [])
        return [
            {
                "symbol": dict(zip(fields_list, r)).get("ts_code", symbol),
                "date": dict(zip(fields_list, r)).get("trade_date", ""),
                "open": safe_float(dict(zip(fields_list, r)).get("open")),
                "high": safe_float(dict(zip(fields_list, r)).get("high")),
                "low": safe_float(dict(zip(fields_list, r)).get("low")),
                "close": safe_float(dict(zip(fields_list, r)).get("close")),
                "volume": safe_int(dict(zip(fields_list, r)).get("vol")),
            }
            for r in rows
        ]

    async def _realtime(self, symbols: list[str], **_: Any) -> list[dict]:
        results = []
        for sym in symbols:
            q = await self._quote(symbol=sym)
            if q:
                results.append(q)
        return results

    # ── Fundamentals ──────────────────────────────────────

    async def _income(self, symbol: str, period: str = "annual", **_: Any) -> dict:
        fields = "ts_code,ann_date,f_ann_date,end_date,report_type,total_revenue,revenue,oper_cost,total_cogs,oper_profit,total_profit,income,net_profit"
        api = "income" if period == "annual" else "income"
        data = await self._request(api, fields=fields, ts_code=symbol)
        rows = data.get("rows", [])
        if not rows:
            return {}
        item = dict(zip(data.get("fields", []), rows[0]))
        return {
            "symbol": item.get("ts_code", symbol),
            "date": item.get("end_date", ""),
            "revenue": safe_float(item.get("total_revenue")),
            "cogs": safe_float(item.get("total_cogs")),
            "gross_profit": safe_float(item.get("total_revenue")) - safe_float(item.get("total_cogs")),
            "net_income": safe_float(item.get("net_profit")),
        }

    async def _balance(self, symbol: str, period: str = "annual", **_: Any) -> dict:
        fields = "ts_code,end_date,total_assets,total_liab,total_hldr_eqy_exc_min_int"
        data = await self._request("balancesheet", fields=fields, ts_code=symbol)
        rows = data.get("rows", [])
        if not rows:
            return {}
        item = dict(zip(data.get("fields", []), rows[0]))
        return {
            "symbol": item.get("ts_code", symbol),
            "date": item.get("end_date", ""),
            "total_assets": safe_float(item.get("total_assets")),
            "total_liabilities": safe_float(item.get("total_liab")),
            "total_equity": safe_float(item.get("total_hldr_eqy_exc_min_int")),
        }

    async def _cashflow(self, symbol: str, period: str = "annual", **_: Any) -> dict:
        fields = "ts_code,end_date,c_pay_for_goods,n_cashflow_act,n_cashflow_inv_act,n_cashflow_fnc_act"
        data = await self._request("cashflow", fields=fields, ts_code=symbol)
        rows = data.get("rows", [])
        if not rows:
            return {}
        item = dict(zip(data.get("fields", []), rows[0]))
        return {
            "symbol": item.get("ts_code", symbol),
            "date": item.get("end_date", ""),
            "operating_cf": safe_float(item.get("n_cashflow_act")),
            "investing_cf": safe_float(item.get("n_cashflow_inv_act")),
            "financing_cf": safe_float(item.get("n_cashflow_fnc_act")),
        }

    # ── Macro ─────────────────────────────────────────────

    async def _macro(
        self, series_id: str, start_date: str | None = None, end_date: str | None = None, **_: Any,
    ) -> list[dict]:
        fields = "trade_date,close"
        data = await self._request(
            "index_daily", fields=fields, ts_code=series_id,
            start_date=start_date, end_date=end_date,
        )
        rows = data.get("rows", [])
        fields_list = data.get("fields", [])
        return [
            {"series_id": series_id, "date": dict(zip(fields_list, r)).get("trade_date", ""),
             "value": safe_float(dict(zip(fields_list, r)).get("close"))}
            for r in rows
        ]

    # ── News ─────────────────────────────────────────────

    async def _news(self, query: str | None = None, **_: Any) -> list[dict]:
        fields = "title,channels,pub_time,url"
        params = {}
        if query:
            params["src"] = query
        data = await self._request("news", fields=fields, **params)
        rows = data.get("rows", [])
        fields_list = data.get("fields", [])
        return [
            {
                "title": dict(zip(fields_list, r)).get("title", ""),
                "source": dict(zip(fields_list, r)).get("channels", ""),
                "published_at": dict(zip(fields_list, r)).get("pub_time", ""),
                "url": dict(zip(fields_list, r)).get("url", ""),
                "symbols": [],
            }
            for r in rows[:20]
        ]

    # ── Fund info ─────────────────────────────────────────

    async def _fund_info(self, symbol: str, include_holdings: bool = False, **_: Any) -> dict:
        fields = "ts_code,name,nv,total_mv"
        data = await self._request("fund_nav", fields=fields, ts_code=symbol)
        rows = data.get("rows", [])
        if not rows:
            return {}
        item = dict(zip(data.get("fields", []), rows[0]))
        result = {
            "symbol": item.get("ts_code", symbol),
            "name": item.get("name", ""),
            "nav": safe_float(item.get("nv")),
            "total_assets": safe_float(item.get("total_mv")),
            "holdings": [],
        }
        if include_holdings:
            h_fields = "ts_code,stock_code,stock_name,vol,amount"
            h_data = await self._request("fund_portfolio", fields=h_fields, ts_code=symbol)
            h_rows = h_data.get("rows", [])
            h_flds = h_data.get("fields", [])
            result["holdings"] = [
                dict(zip(h_flds, r)) for r in h_rows[:10]
            ]
        return result

    # ── Money flow ────────────────────────────────────────

    async def _money_flow(self, symbol: str, period: str = "1d", **_: Any) -> dict:
        fields = "ts_code,trade_date,buy_sm_amount,sell_sm_amount,buy_lg_amount,sell_lg_amount,net_mf_amount"
        data = await self._request("moneyflow", fields=fields, ts_code=symbol)
        rows = data.get("rows", [])
        if not rows:
            return {}
        item = dict(zip(data.get("fields", []), rows[-1]))
        return {
            "symbol": item.get("ts_code", symbol),
            "date": item.get("trade_date", ""),
            "inflow": safe_float(item.get("buy_lg_amount")),
            "outflow": safe_float(item.get("sell_lg_amount")),
            "net_flow": safe_float(item.get("net_mf_amount")),
        }

    # ── Search ────────────────────────────────────────────

    async def _search(self, query: str, **_: Any) -> list[dict]:
        fields = "ts_code,name,area,industry,market,list_date"
        data = await self._request("stock_basic", fields=fields)
        rows = data.get("rows", [])
        fields_list = data.get("fields", [])
        results = []
        for r in rows:
            item = dict(zip(fields_list, r))
            if query.lower() in str(item).lower():
                results.append({
                    "query": query,
                    "source": "tushare",
                    "data_type": "stock",
                    "relevance": 1.0,
                    "payload": item,
                })
        return results[:10]

    async def health_check(self) -> bool:
        if not self._token:
            return False
        try:
            await self._request("trade_cal", fields="exchange", exchange="SSE", limit="1")
            return True
        except Exception:
            return False


register_connector("tushare", TushareConnector)
