"""Calendar tools — earnings, dividends, IPOs, stock splits, economic events, transcripts, analyst estimates."""

from __future__ import annotations

from typing import Any

from ..connectors.base import fetch_with_failover
from ._external import call_external_tool
from ._region import require_region


def _rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("rows", "results", "events", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
    return []


def _row_date(row: dict[str, Any], *fields: str) -> str:
    for field in fields or ("date",):
        value = row.get(field)
        if value:
            return str(value)[:10]
    return ""


def _split_ratio(row: dict[str, Any]) -> str:
    if row.get("ratio"):
        return str(row["ratio"])
    numerator = row.get("numerator") or row.get("numeratorRatio") or row.get("to_factor")
    denominator = row.get("denominator") or row.get("denominatorRatio") or row.get("from_factor")
    if numerator not in (None, "") and denominator not in (None, ""):
        return f"{numerator}:{denominator}"
    return ""


def register(mcp) -> None:

    @mcp.tool()
    async def calendar_get_earnings(
        from_date: str | None = None,
        to_date: str | None = None,
        symbol: str | None = None,
        limit: int = 10,
        region: str = "US",
        as_of_date: str | None = None,
    ) -> list[dict]:
        if symbol:
            return await fetch_with_failover(
                "earning_calendar",
                "earning_calendar_historical",
                region=region,
                symbol=symbol,
                limit=limit,
                from_date=from_date,
                to_date=to_date,
                as_of_date=as_of_date,
            )
        return await fetch_with_failover(
            "earning_calendar",
            "earning_calendar",
            region=region,
            from_date=from_date,
            to_date=to_date,
            as_of_date=as_of_date,
        )

    @mcp.tool()
    async def calendar_get_earnings_surprises(
        symbol: str,
        region: str = "US",
        start_date: str | None = None,
        end_date: str | None = None,
        as_of_date: str | None = None,
    ) -> list[dict]:
        return await fetch_with_failover(
            "earnings_surprises",
            "earnings_surprises",
            region=region,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            as_of_date=as_of_date,
        )

    @mcp.tool()
    async def calendar_get_analyst_estimates(
        symbol: str,
        period: str = "annual",
        limit: int = 10,
        region: str = "US",
        start_date: str | None = None,
        end_date: str | None = None,
        as_of_date: str | None = None,
    ) -> list[dict]:
        return await fetch_with_failover(
            "analyst_estimates",
            "analyst_estimates",
            region=region,
            symbol=symbol,
            period=period,
            limit=limit,
            start_date=start_date,
            end_date=end_date,
            as_of_date=as_of_date,
        )

    @mcp.tool()
    async def calendar_get_earnings_transcript(
        symbol: str,
        quarter: int | None = None,
        year: int | None = None,
        region: str = "US",
    ) -> list[dict]:
        return await fetch_with_failover(
            "earnings_transcript",
            "earnings_transcript",
            region=region,
            symbol=symbol,
            quarter=quarter,
            year=year,
        )

    @mcp.tool()
    async def calendar_get_dividends(
        symbol: str | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
        region: str = "US",
        as_of_date: str | None = None,
    ) -> list[dict]:
        if symbol:
            return await fetch_with_failover(
                "dividend_calendar",
                "dividend_historical",
                region=region,
                symbol=symbol,
                from_date=from_date,
                to_date=to_date,
                as_of_date=as_of_date,
            )
        return await fetch_with_failover(
            "dividend_calendar",
            "dividend_calendar",
            region=region,
            from_date=from_date,
            to_date=to_date,
            as_of_date=as_of_date,
        )

    @mcp.tool()
    async def calendar_get_ipo(
        from_date: str | None = None,
        to_date: str | None = None,
        region: str = "US",
    ) -> list[dict]:
        return await fetch_with_failover(
            "ipo_calendar",
            "ipo_calendar",
            region=region,
            from_date=from_date,
            to_date=to_date,
            as_of_date=to_date,
        )

    @mcp.tool()
    async def calendar_get_stock_splits(
        symbol: str | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
        region: str = "US",
    ) -> list[dict]:
        return await fetch_with_failover(
            "stock_split_calendar",
            "stock_split_calendar",
            region=region,
            symbol=symbol,
            from_date=from_date,
            to_date=to_date,
            as_of_date=to_date,
        )

    @mcp.tool()
    async def calendar_get_corporate_actions(
        symbol: str,
        from_date: str | None = None,
        to_date: str | None = None,
        event_types: list[str] | None = None,
        region: str = "US",
    ) -> dict[str, Any]:
        requested = {
            str(item).strip().lower()
            for item in (event_types or ["dividend", "split", "earnings"])
            if str(item).strip()
        }
        if "stock_split" in requested:
            requested.add("split")
        events: list[dict[str, Any]] = []
        missing_event_types: list[str] = []
        if "dividend" in requested:
            dividend_rows = _rows(await fetch_with_failover(
                "dividend_calendar",
                "dividend_historical",
                region=region,
                symbol=symbol,
                from_date=from_date,
                to_date=to_date,
                as_of_date=to_date,
            ))
            if not dividend_rows:
                missing_event_types.append("dividend")
            for row in dividend_rows:
                event_date = _row_date(row, "date", "paymentDate", "payment_date")
                events.append({
                    "event_type": "dividend",
                    "symbol": str(row.get("symbol") or symbol),
                    "date": event_date,
                    "event_date": event_date,
                    "ex_dividend_date": _row_date(row, "exDividendDate", "ex_dividend_date"),
                    "record_date": _row_date(row, "recordDate", "record_date"),
                    "payment_date": _row_date(row, "paymentDate", "payment_date"),
                    "amount": row.get("amount", row.get("dividend")),
                    "currency": row.get("currency", "USD"),
                    "raw": row,
                })
        if "split" in requested:
            split_rows = _rows(await fetch_with_failover(
                "stock_split_calendar",
                "stock_split_calendar",
                region=region,
                symbol=symbol,
                from_date=from_date,
                to_date=to_date,
                as_of_date=to_date,
            ))
            if not split_rows:
                missing_event_types.append("split")
            for row in split_rows:
                event_date = _row_date(row, "date", "executionDate", "effectiveDate")
                events.append({
                    "event_type": "split",
                    "symbol": str(row.get("symbol") or symbol),
                    "date": event_date,
                    "event_date": event_date,
                    "ratio": _split_ratio(row),
                    "raw": row,
                })
        if "earnings" in requested or "earning" in requested:
            earnings_rows = _rows(await fetch_with_failover(
                "earning_calendar",
                "earning_calendar_historical",
                region=region,
                symbol=symbol,
                from_date=from_date,
                to_date=to_date,
                as_of_date=to_date,
            ))
            if not earnings_rows:
                missing_event_types.append("earnings")
            for row in earnings_rows:
                event_date = _row_date(row, "date", "fiscalDateEnding")
                events.append({
                    "event_type": "earnings",
                    "symbol": str(row.get("symbol") or symbol),
                    "date": event_date,
                    "event_date": event_date,
                    "time": row.get("time", ""),
                    "eps_estimated": row.get("epsEstimated", row.get("eps_estimated")),
                    "eps_actual": row.get("eps", row.get("eps_actual")),
                    "raw": row,
                })
        events.sort(key=lambda item: str(item.get("date") or ""))
        return {
            "status": "success" if events else "empty_result",
            "symbol": symbol,
            "events": events,
            "missing_event_types": missing_event_types,
        }

    @mcp.tool()
    async def calendar_get_next_earnings(
        identifier: str,
        region: str = "US",
    ) -> Any:
        require_region(region, allowed={"US"}, data_type="earnings_event")
        return await call_external_tool(
            "spglobal",
            "get_next_earnings_from_identifiers",
            identifier=identifier,
        )

    @mcp.tool()
    async def calendar_get_latest_earnings(
        identifier: str,
        region: str = "US",
    ) -> Any:
        require_region(region, allowed={"US"}, data_type="earnings_event")
        return await call_external_tool(
            "spglobal",
            "get_latest_earnings_from_identifiers",
            identifier=identifier,
        )
