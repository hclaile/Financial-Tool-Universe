"""Shared historical routing for supported market indices."""

from __future__ import annotations

from typing import Any, Awaitable, Callable


IndexFetcher = Callable[..., Awaitable[Any]]

INDEX_FRED_SERIES: dict[str, str] = {
    "^GSPC": "SP500",
    "^SOX": "NASDAQSOX",
    "^IXIC": "NASDAQCOM",
    "^DJI": "DJIA",
    "^VIX": "VIXCLS",
    "^TNX": "DGS10",
    "DX-Y.NYB": "DTWEXBGS",
}


def _fred_rows(symbol: str, observations: Any, series_id: str) -> list[dict[str, Any]]:
    if not isinstance(observations, list):
        return []
    rows: list[dict[str, Any]] = []
    for observation in observations:
        if not isinstance(observation, dict):
            continue
        date = str(observation.get("date") or "")[:10]
        try:
            value = float(observation.get("value"))
        except (TypeError, ValueError):
            continue
        rows.append(
            {
                "symbol": symbol,
                "date": date,
                "open": value,
                "high": value,
                "low": value,
                "close": value,
                "volume": 0,
                "source_series_id": str(observation.get("series_id") or series_id),
                "proxy_for_symbol": symbol,
            }
        )
    return rows


async def fetch_index_history(
    fetcher: IndexFetcher,
    *,
    symbol: str,
    start_date: str | None,
    end_date: str | None,
    as_of_date: str | None = None,
    region: str = "US",
) -> list[dict[str, Any]] | None:
    """Fetch an index via FRED first, then provider OHLCV as fallback.

    ``None`` means the index has no configured FRED series and the caller
    should use its ordinary route.  A list, including an empty list after the
    fallback, means the index route was selected explicitly.
    """

    series_id = INDEX_FRED_SERIES.get(symbol)
    if not series_id:
        return None

    fred_kwargs: dict[str, Any] = {
        "series_id": series_id,
        "start_date": start_date,
        "end_date": end_date,
    }
    if as_of_date:
        fred_kwargs["as_of_date"] = as_of_date

    try:
        observations = await fetcher(
            "macro_series",
            "series",
            region="US",
            **fred_kwargs,
        )
    except Exception:
        observations = []
    rows = _fred_rows(symbol, observations, series_id)
    if rows:
        return rows

    fallback_kwargs: dict[str, Any] = {
        "symbol": symbol,
        "start_date": start_date,
        "end_date": end_date,
        "interval": "1d",
    }
    if as_of_date:
        fallback_kwargs["as_of_date"] = as_of_date
    try:
        fallback = await fetcher(
            "ohlcv",
            "ohlcv",
            region=region,
            **fallback_kwargs,
        )
    except Exception:
        return []
    if isinstance(fallback, dict):
        return fallback.get("rows", []) if isinstance(fallback.get("rows"), list) else []
    return fallback if isinstance(fallback, list) else []
