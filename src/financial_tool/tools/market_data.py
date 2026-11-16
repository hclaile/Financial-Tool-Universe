"""Market data tools — quotes, history, realtime, rating, key metrics, technicals, commodity, forex."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pandas as pd

from ..compat.metrics import compute_technical_indicators
from ..compat.temporal_policy import technical_warmup_start_date

from ..connectors.base import fetch_with_failover, get_source_connector
from ..errors import SourceUnavailableError
from ..symbols import classify_instrument
from ._date_window import normalize_market_data_window
from ._external import call_external_tool
from ._index_history import fetch_index_history
from ._region import require_region


def _indicator_key(statistics_type: str, period: int) -> str:
    """Map ``statistics_type`` + ``period`` to the indicator dict key
    produced by ``compute_technical_indicators``.
    """
    st = statistics_type.upper().strip()
    if st in {"SMA", "MA"}:
        return f"sma_{period}"
    if st == "EMA":
        return f"ema_{period}"
    if st == "RSI":
        return f"rsi_{period}"
    if st == "ADX":
        return f"adx_{period}"
    if st == "MACD":
        return "macd"
    if st == "WMA":
        return f"wma_{period}"
    if st == "DEMA":
        return f"dema_{period}"
    if st == "TEMA":
        return f"tema_{period}"
    if st in {"Williams", "WILLR"}:
        return f"willr_{period}"
    # Fallback: lowercase the type name
    return f"{st.lower()}_{period}"


def _market_instrument(symbol: str, region: str) -> tuple[str, str]:
    instrument = classify_instrument(symbol)
    if instrument.kind == "macro_series":
        from ..errors import DataValidationError

        raise DataValidationError(
            "symbol",
            f"{symbol} is macro data; call macro_get_series("
            f"series_id={instrument.provider_symbol!r}, region='US')",
        )
    if instrument.kind == "shipping_series":
        from ..errors import DataValidationError

        raise DataValidationError(
            "symbol",
            f"{symbol} is a shipping index; call a shipping-series tool "
            f"with series_id={instrument.provider_symbol!r}",
        )
    return instrument.provider_symbol, instrument.region or region


def _parse_iso_date(value: str | None) -> date | None:
    try:
        return date.fromisoformat(str(value or "").replace("/", "-")[:10])
    except ValueError:
        return None


def _rows_from_result(result: Any) -> list[dict[str, Any]]:
    if isinstance(result, list):
        return [row for row in result if isinstance(row, dict)]
    if isinstance(result, dict):
        rows = result.get("rows")
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
        data = result.get("data")
        if isinstance(data, list):
            return [row for row in data if isinstance(row, dict)]
    return []


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _select_price_row(
    rows: list[dict[str, Any]],
    target_date: str,
    nearest_policy: str,
) -> tuple[dict[str, Any] | None, str]:
    target = _parse_iso_date(target_date)
    dated_rows: list[tuple[date, dict[str, Any]]] = []
    for row in rows:
        row_date = _parse_iso_date(str(row.get("date") or row.get("period") or ""))
        if row_date is not None:
            dated_rows.append((row_date, row))
    if target is None or not dated_rows:
        return None, "date_parse_failed" if target is None else "no_dated_rows"
    dated_rows.sort(key=lambda item: item[0])
    exact = [item for item in dated_rows if item[0] == target]
    if exact:
        return exact[-1][1], "exact"
    policy = str(nearest_policy or "previous_trading_day").strip().lower()
    previous = [item for item in dated_rows if item[0] <= target]
    next_rows = [item for item in dated_rows if item[0] >= target]
    if policy in {"exact", "same_day"}:
        return None, "exact_date_not_available"
    if policy in {"previous", "previous_trading_day", "prior", "last"}:
        return (previous[-1][1], "previous_trading_day") if previous else (None, "no_previous_trading_day")
    if policy in {"next", "next_trading_day"}:
        return (next_rows[0][1], "next_trading_day") if next_rows else (None, "no_next_trading_day")
    if policy in {"nearest", "nearest_trading_day"}:
        nearest = min(dated_rows, key=lambda item: (abs((item[0] - target).days), item[0] > target))
        return nearest[1], "nearest_trading_day"
    return (previous[-1][1], "previous_trading_day") if previous else (None, "unsupported_nearest_policy")


def _commodity_alias(symbol: str | None) -> tuple[str | None, dict[str, str]]:
    raw = str(symbol or "").strip()
    lowered = raw.lower()
    aliases = {
        "crude oil wti": ("CL=F", "commodity_future", "USD/bbl"),
        "wti": ("CL=F", "commodity_future", "USD/bbl"),
        "wti crude": ("CL=F", "commodity_future", "USD/bbl"),
        "wti crude oil": ("CL=F", "commodity_future", "USD/bbl"),
        "cl=f": ("CL=F", "commodity_future", "USD/bbl"),
        "brent": ("BZ=F", "commodity_future", "USD/bbl"),
        "brent crude": ("BZ=F", "commodity_future", "USD/bbl"),
        "gold": ("GC=F", "commodity_future", "USD/oz"),
        "silver": ("SI=F", "commodity_future", "USD/oz"),
    }
    mapped = aliases.get(lowered)
    if not mapped:
        return symbol, {}
    return mapped[0], {"asset_type": mapped[1], "unit": mapped[2], "requested_symbol": raw}


async def _fetch_index_history_from_fred(
    symbol: str,
    start_date: str | None,
    end_date: str | None,
    as_of_date: str | None,
) -> list[dict] | None:
    """Fetch supported index closes from FRED without treating them as stocks.

    Yahoo's ``Ticker.history`` reports a misleading ``possibly delisted``
    warning when an index request is blocked by a proxy or returns an empty
    payload.  FRED publishes stable daily close series for the common US
    indices below, which is sufficient for historical close/SMA requests.
    ``None`` means no FRED mapping exists and the normal OHLCV failover should
    remain in charge for that index.
    """

    return await fetch_index_history(
        fetch_with_failover,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        as_of_date=as_of_date,
        region="US",
    )


async def _fetch_crypto_history(
    *,
    symbol: str,
    start_date: str | None,
    end_date: str | None,
    interval: str,
    as_of_date: str | None = None,
) -> Any:
    request_kwargs: dict[str, Any] = {
        "symbol": symbol,
        "start_date": start_date,
        "end_date": end_date,
        "interval": interval,
    }
    if as_of_date:
        request_kwargs["as_of_date"] = as_of_date
    return await fetch_with_failover(
        "crypto_ohlcv",
        "ohlcv",
        region="GLOBAL",
        **request_kwargs,
    )


def register(mcp) -> None:

    @mcp.tool()
    async def market_data_get_akshare_stock_price(symbol: str) -> dict:
        """Get an A-share price snapshot directly from AKShare."""
        requested_symbol = str(symbol or "").strip()
        if not requested_symbol:
            raise ValueError("symbol must be a nonempty A-share code")
        quote = await get_source_connector("akshare").fetch("quote", symbol=requested_symbol)
        if not quote:
            return {
                "status": "unavailable",
                "symbol": requested_symbol,
                "source": "akshare",
                "reason": "No usable A-share price in the current AKShare snapshot",
            }
        return {"status": "success", **quote}

    @mcp.tool()
    async def market_data_get_quote(
        symbol: str,
        exchange: str | None = None,
        region: str = "CN",
        start_date: str | None = None,
        end_date: str | None = None,
        as_of_date: str | None = None,
    ) -> dict:
        symbol, region = _market_instrument(symbol, region)
        if start_date or end_date or as_of_date:
            start_date, end_date, as_of_date = normalize_market_data_window(
                start_date=start_date,
                end_date=end_date,
                as_of_date=as_of_date,
            )
        return await fetch_with_failover(
            "quote", "quote", region=region, symbol=symbol, exchange=exchange,
            start_date=start_date, end_date=end_date, as_of_date=as_of_date,
        )

    @mcp.tool()
    async def market_data_get_history(
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
        lookback: int = 14,
        interval: str = "1d",
        region: str = "CN",
        as_of_date: str | None = None,
    ) -> list[dict]:
        instrument = classify_instrument(symbol)
        symbol, region = _market_instrument(symbol, region)
        requested_as_of_date = as_of_date
        start_date, end_date, as_of_date = normalize_market_data_window(
            start_date=start_date,
            end_date=end_date,
            as_of_date=as_of_date,
        )
        if instrument.kind == "crypto":
            return await _fetch_crypto_history(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                interval=interval,
                as_of_date=as_of_date if requested_as_of_date else None,
            )
        if instrument.kind == "index":
            index_rows = await _fetch_index_history_from_fred(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                as_of_date=as_of_date if requested_as_of_date else None,
            )
            if index_rows is not None:
                return index_rows
        request_kwargs = {
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "interval": interval,
        }
        if requested_as_of_date:
            request_kwargs["as_of_date"] = as_of_date
        return await fetch_with_failover(
            "ohlcv", "ohlcv", region=region,
            **request_kwargs,
        )

    @mcp.tool()
    async def market_data_get_exact_price(
        symbol: str,
        target_date: str,
        price_field: str = "close",
        adjustment: str = "raw",
        nearest_policy: str = "previous_trading_day",
        region: str = "US",
        lookaround_days: int = 10,
    ) -> dict:
        target = _parse_iso_date(target_date)
        if target is None:
            return {
                "status": "invalid_request",
                "symbol": symbol,
                "requested_date": target_date,
                "reason": "target_date must use YYYY-MM-DD",
            }
        instrument = classify_instrument(symbol)
        normalized_symbol, normalized_region = _market_instrument(symbol, region)
        days = max(1, min(int(lookaround_days), 30))
        start_date = (target - timedelta(days=days)).isoformat()
        end_date = (target + timedelta(days=days)).isoformat() if str(nearest_policy).lower().startswith("next") else target.isoformat()
        if instrument.kind == "index":
            history = await _fetch_index_history_from_fred(
                symbol=normalized_symbol,
                start_date=start_date,
                end_date=end_date,
                as_of_date=target.isoformat(),
            )
            if history is None:
                history = await fetch_with_failover(
                    "ohlcv",
                    "ohlcv",
                    region=normalized_region,
                    symbol=normalized_symbol,
                    start_date=start_date,
                    end_date=end_date,
                    interval="1d",
                )
        else:
            history = await fetch_with_failover(
                "ohlcv",
                "ohlcv",
                region=normalized_region,
                symbol=normalized_symbol,
                start_date=start_date,
                end_date=end_date,
                interval="1d",
            )
        rows = _rows_from_result(history)
        selected, resolution = _select_price_row(rows, target.isoformat(), nearest_policy)
        if selected is None:
            return {
                "status": "empty_result",
                "symbol": normalized_symbol,
                "requested_date": target.isoformat(),
                "nearest_policy": nearest_policy,
                "reason": resolution,
            }
        actual_date = str(selected.get("date") or "")[:10]
        close = _float_or_none(selected.get("close"))
        adj_close = _float_or_none(
            selected.get("adj_close")
            if "adj_close" in selected
            else selected.get("adjusted_close")
        )
        field = str(price_field or "close").strip().lower()
        if str(adjustment or "raw").strip().lower() in {"adjusted", "adj"} and adj_close is not None:
            selected_price = adj_close
            field = "adj_close"
        else:
            selected_price = _float_or_none(selected.get(field))
            if selected_price is None and field == "adj_close":
                selected_price = adj_close
            if selected_price is None and close is not None:
                selected_price = close
                field = "close"
        return {
            "status": "success",
            "symbol": normalized_symbol,
            "requested_date": target.isoformat(),
            "actual_date": actual_date,
            "is_exact_date": actual_date == target.isoformat(),
            "nearest_policy": nearest_policy,
            "resolution": resolution,
            "price_field": field,
            "adjustment": adjustment,
            "adjustment_policy": adjustment,
            "selected_price": selected_price,
            "open": _float_or_none(selected.get("open")),
            "high": _float_or_none(selected.get("high")),
            "low": _float_or_none(selected.get("low")),
            "close": close,
            "adj_close": adj_close,
            "volume": selected.get("volume"),
            "currency": {"US": "USD", "HK": "HKD", "CN": "CNY"}.get(normalized_region, ""),
            "source": "ohlcv",
            "temporal_metadata": {
                "requested_start": target.isoformat(),
                "requested_end": target.isoformat(),
                "actual_min_date": actual_date,
                "actual_max_date": actual_date,
                "date_dimension": "market_observation",
                "temporal_safe": True,
                "date_verification": "verified",
                "coverage_completeness": "known",
            },
        }

    @mcp.tool()
    async def market_data_get_multi_history(
        symbols: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
        lookback: int = 14,
        interval: str = "1d",
        region: str = "US",
        as_of_date: str | None = None,
    ) -> dict[str, list[dict]]:
        requested_as_of_date = as_of_date
        start_date, end_date, as_of_date = normalize_market_data_window(
            start_date=start_date,
            end_date=end_date,
            as_of_date=as_of_date,
        )
        results: dict[str, list[dict]] = {}
        for symbol in symbols:
            instrument = classify_instrument(symbol)
            normalized_symbol, normalized_region = _market_instrument(symbol, region)
            if instrument.kind == "crypto":
                results[symbol] = await _fetch_crypto_history(
                    symbol=normalized_symbol,
                    start_date=start_date,
                    end_date=end_date,
                    interval=interval,
                    as_of_date=as_of_date if requested_as_of_date else None,
                )
                continue
            if instrument.kind == "index":
                index_rows = await _fetch_index_history_from_fred(
                    symbol=normalized_symbol,
                    start_date=start_date,
                    end_date=end_date,
                    as_of_date=as_of_date if requested_as_of_date else None,
                )
                if index_rows is not None:
                    results[symbol] = index_rows
                    continue
            request_kwargs = {
                "symbol": normalized_symbol,
                "start_date": start_date,
                "end_date": end_date,
                "interval": interval,
            }
            if requested_as_of_date:
                request_kwargs["as_of_date"] = as_of_date
            results[symbol] = await fetch_with_failover(
                "ohlcv", "ohlcv", region=normalized_region,
                **request_kwargs,
            )
        return results

    @mcp.tool()
    async def market_data_get_realtime(
        symbols: list[str],
        region: str = "CN",
    ) -> list[dict]:
        quotes = []
        for symbol in symbols:
            normalized_symbol, normalized_region = _market_instrument(symbol, region)
            quotes.append(await fetch_with_failover(
                "quote",
                "quote",
                region=normalized_region,
                symbol=normalized_symbol,
            ))
        return quotes

    @mcp.tool()
    async def market_data_get_rating(
        symbol: str,
        limit: int = 10,
        region: str = "US",
        start_date: str | None = None,
        end_date: str | None = None,
        as_of_date: str | None = None,
    ) -> dict | list[dict]:
        if start_date or end_date or as_of_date:
            start_date, end_date, as_of_date = normalize_market_data_window(
                start_date=start_date,
                end_date=end_date,
                as_of_date=as_of_date,
            )
        if limit > 1:
            return await fetch_with_failover(
                "rating",
                "rating_historical",
                region=region,
                symbol=symbol,
                limit=limit,
                start_date=start_date,
                end_date=end_date,
                as_of_date=as_of_date,
            )
        return await fetch_with_failover(
            "rating",
            "rating",
            region=region,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            as_of_date=as_of_date,
        )

    @mcp.tool()
    async def market_data_get_key_metrics(
        symbol: str,
        period: str = "annual",
        limit: int = 10,
        region: str = "US",
        start_date: str | None = None,
        end_date: str | None = None,
        as_of_date: str | None = None,
    ) -> dict | list[dict]:
        if start_date or end_date or as_of_date:
            start_date, end_date, as_of_date = normalize_market_data_window(
                start_date=start_date,
                end_date=end_date,
                as_of_date=as_of_date,
            )
        if period == "ttm":
            return await fetch_with_failover(
                "key_metrics",
                "key_metrics_ttm",
                region=region,
                symbol=symbol,
                limit=limit,
                start_date=start_date,
                end_date=end_date,
                as_of_date=as_of_date,
            )
        return await fetch_with_failover(
            "key_metrics",
            "key_metrics",
            region=region,
            symbol=symbol,
            period=period,
            limit=limit,
            start_date=start_date,
            end_date=end_date,
            as_of_date=as_of_date,
        )

    @mcp.tool()
    async def market_data_get_technical(
        symbol: str,
        period: int = 10,
        statistics_type: str = "SMA",
        time_delta: str = "daily",
        region: str = "US",
        start_date: str | None = None,
        end_date: str | None = None,
        as_of_date: str | None = None,
    ) -> list[dict]:
        """Compute technical indicators locally from OHLCV data.

        Priority chain:
        1. Fetch OHLCV via failover (FMP → yfinance) then compute
           indicators locally using TA-Lib / pandas.
        2. If OHLCV is unavailable or computation fails, raise
           SourceUnavailableError.

        The old approach of calling FMP's ``/technical_indicator/daily``
        endpoint has been removed because that endpoint does not exist
        on the FMP stable API.
        """
        instrument = classify_instrument(symbol)
        normalized_symbol, normalized_region = _market_instrument(symbol, region)
        start_date, end_date, as_of_date = normalize_market_data_window(
            start_date=start_date,
            end_date=end_date,
            as_of_date=as_of_date,
        )
        output_start_date = start_date
        calculation_start_date = technical_warmup_start_date(output_start_date, period)
        # ── Priority 1: OHLCV → local talib/pandas computation ──
        if instrument.kind == "crypto":
            history = await _fetch_crypto_history(
                symbol=normalized_symbol,
                start_date=calculation_start_date,
                end_date=end_date,
                as_of_date=as_of_date,
                interval="1d",
            )
        elif instrument.kind == "index":
            history = await _fetch_index_history_from_fred(
                symbol=normalized_symbol,
                start_date=calculation_start_date,
                end_date=end_date,
                as_of_date=as_of_date,
            )
            if history is None:
                history = await fetch_with_failover(
                    "ohlcv",
                    "ohlcv",
                    region=normalized_region,
                    symbol=normalized_symbol,
                    start_date=calculation_start_date,
                    end_date=end_date,
                    as_of_date=as_of_date,
                    interval="1d",
                )
        else:
            history = await fetch_with_failover(
                "ohlcv",
                "ohlcv",
                region=normalized_region,
                symbol=normalized_symbol,
                start_date=calculation_start_date,
                end_date=end_date,
                as_of_date=as_of_date,
                interval="1d",
            )

        rows = history.get("rows", []) if isinstance(history, dict) else history
        if not isinstance(rows, list) or not rows:
            raise SourceUnavailableError(
                "technical_indicator",
                reason=f"empty_ohlcv for {symbol}",
            )
        if len(rows) < period + 1:
            raise SourceUnavailableError(
                "technical_indicator",
                reason=(
                    f"insufficient_history for {statistics_type}({period}) on {symbol}: "
                    f"need >= {period + 1} bars, received {len(rows)}"
                ),
            )

        frame = pd.DataFrame(rows)
        if not {"date", "close"}.issubset(frame.columns):
            raise SourceUnavailableError(
                "technical_indicator",
                reason="schema_invalid: OHLCV requires date and close",
            )
        frame = frame.sort_values("date").reset_index(drop=True)
        try:
            indicators = compute_technical_indicators(frame)
        except (TypeError, ValueError) as exc:
            raise SourceUnavailableError(
                "technical_indicator",
                reason=f"invalid_ohlcv: {type(exc).__name__}",
            ) from exc
        if not indicators:
            raise SourceUnavailableError(
                "technical_indicator",
                reason=f"indicator_empty for {statistics_type}({period})",
            )

        key = _indicator_key(statistics_type, period)
        value = indicators.get(key)
        result: dict[str, object] = {
            "symbol": symbol,
            "date": str(frame["date"].iloc[-1]),
            "source": indicators.get("technical_backend", "local"),
            "temporal_metadata": {
                "requested_start": output_start_date,
                "requested_end": end_date,
                "actual_min_date": str(frame["date"].iloc[0])[:10],
                "actual_max_date": str(frame["date"].iloc[-1])[:10],
                "calculation_start": calculation_start_date,
                "date_dimension": "market_observation",
                "temporal_safe": True,
                "date_verification": "verified",
                "coverage_completeness": "known",
            },
        }
        for indicator_name, indicator_value in indicators.items():
            if indicator_value is not None and indicator_name not in (
                "indicator_sources",
                "technical_backend",
            ):
                result[indicator_name] = indicator_value
        if key not in result and value is not None:
            result[key] = value
        return [result]

    @mcp.tool()
    async def market_data_get_market_status(region: str = "US") -> dict:
        return await fetch_with_failover(
            "market_status",
            "market_status",
            region=region,
        )

    @mcp.tool()
    async def market_data_get_sectors_performance(
        limit: int = 10,
        region: str = "US",
    ) -> list[dict]:
        return await fetch_with_failover(
            "sectors_performance",
            "sectors_performance",
            region=region,
            limit=limit,
        )

    @mcp.tool()
    async def market_data_get_commodity(
        symbol: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        region: str = "US",
        as_of_date: str | None = None,
    ) -> list[dict]:
        normalized_symbol, commodity_metadata = _commodity_alias(symbol)
        if symbol and (start_date or end_date or as_of_date):
            start_date, end_date, as_of_date = normalize_market_data_window(
                start_date=start_date,
                end_date=end_date,
                as_of_date=as_of_date,
            )
            result = await fetch_with_failover(
                "commodity_data",
                "commodity_historical",
                region=region,
                symbol=normalized_symbol,
                start_date=start_date,
                end_date=end_date,
                as_of_date=as_of_date,
            )
        else:
            result = await fetch_with_failover(
                "commodity_data",
                "commodity",
                region=region,
                symbol=normalized_symbol,
            )
        if commodity_metadata:
            rows = result if isinstance(result, list) else result.get("rows") if isinstance(result, dict) else None
            if isinstance(rows, list):
                for row in rows:
                    if isinstance(row, dict):
                        row.update({key: value for key, value in commodity_metadata.items() if key not in row})
        return result

    @mcp.tool()
    async def market_data_get_forex(
        symbol: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        region: str = "US",
        as_of_date: str | None = None,
    ) -> list[dict]:
        if symbol and (start_date or end_date or as_of_date):
            start_date, end_date, as_of_date = normalize_market_data_window(
                start_date=start_date,
                end_date=end_date,
                as_of_date=as_of_date,
            )
            return await fetch_with_failover(
                "forex_data",
                "forex_historical",
                region=region,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                as_of_date=as_of_date,
            )
        return await fetch_with_failover(
            "forex_data",
            "forex",
            region=region,
            symbol=symbol,
        )

    @mcp.tool()
    async def market_data_get_total_return_history(
        identifier: str,
        start_date: str | None = None,
        end_date: str | None = None,
        periodicity: str = "day",
        region: str = "US",
        as_of_date: str | None = None,
    ) -> dict:
        require_region(region, allowed={"US"}, data_type="total_return_history")
        start_date, end_date, as_of_date = normalize_market_data_window(
            start_date=start_date,
            end_date=end_date,
            as_of_date=as_of_date,
        )
        return await call_external_tool(
            "lseg",
            "qa_historical_equity_price",
            identifier=identifier,
            start_date=start_date,
            end_date=end_date,
            as_of_date=as_of_date,
            periodicity=periodicity,
        )

    @mcp.tool()
    async def market_data_get_ric_history(
        ric: str,
        interval: str = "P1D",
        tenor: str | None = None,
        region: str = "US",
    ) -> dict:
        require_region(region, allowed={"US"}, data_type="ric_history")
        return await call_external_tool(
            "lseg",
            "tscc_historical_pricing_summaries",
            ric=ric,
            interval=interval,
            tenor=tenor,
        )
