"""Macro tools — macroeconomic time series, economic calendar, treasury yields."""

from __future__ import annotations

from typing import Any

from ..connectors.base import fetch_with_failover, get_source_connector
from ..errors import DataValidationError, SourceUnavailableError
from ..symbols import classify_instrument
from ..utils.macro_coverage import assess_macro_series_coverage
from ._external import call_external_tool
from ._region import require_region


_KNOWN_FREQUENCY_SHORT = {
    "GDP": "Q",
    "GDPC1": "Q",
    "CPIAUCSL": "M",
    "CPILFESL": "M",
    "UNRATE": "M",
    "FEDFUNDS": "M",
    "DGS10": "D",
}

_MACRO_SERIES_ALIASES = {
    "US GDP": "GDP",
    "FEDFEDRATE": "FEDFUNDS",
    "FEDERAL RESERVE INTEREST RATE HIKES": "FEDFUNDS",
    "FEDERAL FUNDS RATE": "FEDFUNDS",
    "FED FUNDS RATE": "FEDFUNDS",
    "FED RATE": "FEDFUNDS",
    "INTEREST RATE HIKES": "FEDFUNDS",
    "CPI": "CPIAUCSL",
    "INFLATION": "CPIAUCSL",
    "CONSUMER PRICE INFLATION": "CPIAUCSL",
    "CONSUMER PRICE INDEX": "CPIAUCSL",
    "PCECTP": "PCECTPI",
    "PCE INFLATION": "PCECTPI",
    "PCE INFLATION INDEX": "PCECTPI",
    "UNEMPLOYMENT": "UNRATE",
    "HOUSING STARTS": "HOUST",
    "HOUST": "HOUST",
    "10Y TREASURY": "DGS10",
    "10-YEAR TREASURY": "DGS10",
}


def resolve_macro_series_id(value: str) -> str:
    text = " ".join(str(value or "").strip().upper().replace("_", " ").split())
    if not text:
        return ""
    return _MACRO_SERIES_ALIASES.get(text, str(value or "").strip().upper())


def _infer_frequency_short(series_id: str, observations: list[dict[str, Any]]) -> str:
    known = _KNOWN_FREQUENCY_SHORT.get(str(series_id or "").upper())
    if known:
        return known
    dates = sorted(str(item.get("date") or "")[:10] for item in observations if isinstance(item, dict) and item.get("date"))
    if len(dates) < 2:
        return ""
    months = []
    for value in dates[:6]:
        try:
            year, month, _ = value.split("-")
            months.append(int(year) * 12 + int(month))
        except ValueError:
            return ""
    deltas = [right - left for left, right in zip(months, months[1:]) if right >= left]
    if not deltas:
        return ""
    median_delta = sorted(deltas)[len(deltas) // 2]
    if median_delta >= 11:
        return "A"
    if median_delta >= 3:
        return "Q"
    if median_delta == 1:
        return "M"
    return "D"


def register(mcp) -> None:

    @mcp.tool()
    async def macro_get_series(
        series_id: str,
        start_date: str | None = None,
        end_date: str | None = None,
        region: str = "US",
    ) -> dict[str, Any]:
        series_id = resolve_macro_series_id(series_id)
        observations = await fetch_with_failover(
            "macro_series", "series", region=region,
            series_id=series_id, start_date=start_date, end_date=end_date,
        )
        if isinstance(observations, dict) and observations.get("status"):
            return {
                "status": str(observations.get("status") or ""),
                "series_id": series_id,
                "region": region,
                "observations": [],
                "rows": [],
                "reason": str(observations.get("reason") or ""),
            }
        observations = [dict(item) for item in observations] if isinstance(observations, list) else []
        frequency_short = _infer_frequency_short(series_id, observations)
        observed_dates = [
            str(item.get("date") or "")[:10]
            for item in observations
            if isinstance(item, dict) and item.get("date")
        ]
        requested_start = start_date or (min(observed_dates) if observed_dates else None)
        requested_end = end_date or (max(observed_dates) if observed_dates else None)
        coverage = (
            assess_macro_series_coverage(
                series_id=series_id,
                requested_start=requested_start,
                requested_end=requested_end,
                observations=observations,
                frequency_short=frequency_short,
            )
            if requested_start and requested_end
            else {
                "series_id": series_id,
                "status": "missing",
                "expected_observation_count": 0,
                "observed_observation_count": len(observations),
                "covered_period_end": None,
            }
        )
        return {
            "status": "success" if observations else "empty_result",
            "series_id": series_id,
            "region": region,
            "start_date": start_date,
            "end_date": end_date,
            "frequency_short": frequency_short,
            "frequency": {
                "D": "Daily",
                "M": "Monthly",
                "Q": "Quarterly",
                "A": "Annual",
            }.get(frequency_short, ""),
            "observations": observations,
            "rows": observations,
            "coverage": coverage,
        }

    @mcp.tool()
    async def macro_get_shipping_index(
        series_id: str = "SCFI",
        start_date: str | None = None,
        end_date: str | None = None,
        region: str = "CN",
    ) -> Any:
        """Fetch a shipping-index observation from the official SSE SCFI page.

        SCFI is a weekly freight index, not an equity ticker.  The official
        page exposes the current composite observation; historical requests
        that cannot be served are returned as a structured, non-retryable
        status so the manager can choose a conditional answer or another
        evidence path instead of sending the symbol through OHLCV failover.
        """
        instrument = classify_instrument(series_id)
        if instrument.kind != "shipping_series" or instrument.provider_symbol != "SCFI":
            raise DataValidationError(
                "series_id",
                "macro_get_shipping_index supports only SCFI shipping series; "
                "use macro_get_series for macro identifiers and market_data_get_history "
                "for tradable instruments",
            )
        if str(region).strip().upper() not in {"CN", "GLOBAL"}:
            raise DataValidationError(
                "region",
                "SCFI is published by the Shanghai Shipping Exchange and is routed in CN/GLOBAL",
            )
        connector = get_source_connector("sse_scfi")
        try:
            return await connector.fetch(
                "series",
                series_id=instrument.provider_symbol,
                start_date=start_date,
                end_date=end_date,
            )
        except SourceUnavailableError as exc:
            return {
                "status": "source_unavailable",
                "series_id": instrument.provider_symbol,
                "retryable": False,
                "reason": exc.reason or str(exc),
            }

    @mcp.tool()
    async def macro_get_economic_calendar(
        from_date: str | None = None,
        to_date: str | None = None,
        region: str = "US",
    ) -> list[dict]:
        return await fetch_with_failover(
            "economic_calendar",
            "economic_calendar",
            region=region,
            from_date=from_date,
            to_date=to_date,
        )

    @mcp.tool()
    async def macro_get_treasury(
        start_date: str | None = None,
        end_date: str | None = None,
        region: str = "US",
    ) -> list[dict]:
        return await fetch_with_failover(
            "treasury_yield",
            "treasury",
            region=region,
            start_date=start_date,
            end_date=end_date,
        )

    @mcp.tool()
    async def macro_get_interest_rate_curve(
        country: str,
        currency: str | None = None,
        curve_name: str | None = None,
        region: str = "US",
    ) -> Any:
        require_region(region, allowed={"US"}, data_type="interest_rate_curve")
        return await call_external_tool(
            "lseg",
            "interest_rate_curve",
            country=country,
            currency=currency,
            curve_name=curve_name,
        )

    @mcp.tool()
    async def macro_get_fx_forward_curve(
        currency_pair: str,
        region: str = "US",
    ) -> Any:
        require_region(region, allowed={"US"}, data_type="fx_forward_curve")
        return await call_external_tool("lseg", "fx_forward_curve", currency_pair=currency_pair)

    @mcp.tool()
    async def macro_get_fx_spot_forward(
        currency_pair: str,
        tenor: str | None = None,
        value_date: str | None = None,
        region: str = "US",
    ) -> dict:
        require_region(region, allowed={"US"}, data_type="fx_spot_forward")
        spot = await call_external_tool("lseg", "fx_spot_price", currency_pair=currency_pair)
        forward = await call_external_tool(
            "lseg",
            "fx_forward_price",
            currency_pair=currency_pair,
            tenor=tenor,
            value_date=value_date,
        )
        return {"spot": spot, "forward": forward}

    @mcp.tool()
    async def macro_get_credit_curve(
        country: str,
        issuer_type: str = "Corporate",
        currency: str | None = None,
        region: str = "US",
    ) -> Any:
        require_region(region, allowed={"US"}, data_type="credit_curve")
        return await call_external_tool(
            "lseg",
            "credit_curve",
            country=country,
            issuer_type=issuer_type,
            currency=currency,
        )

    @mcp.tool()
    async def macro_get_inflation_curve(
        country: str,
        currency: str | None = None,
        region: str = "US",
    ) -> Any:
        require_region(region, allowed={"US"}, data_type="inflation_curve")
        return await call_external_tool(
            "lseg",
            "inflation_curve",
            country=country,
            currency=currency,
        )
