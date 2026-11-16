"""Trading analytics tools — correlations and factor research."""

from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd

from ..connectors.base import fetch_with_failover
from ..connectors._numeric import safe_float
from ._external import call_external_tool


_QLIB_STATE: tuple[str, str] | None = None


def preprocess_factor_formula(formula: str) -> str:
    """Normalize the small amount of syntax accepted by the legacy IC service."""

    value = str(formula or "").strip()
    if not value:
        return ""
    value = re.sub(r"Constant\s*\(\s*(-?\d+(?:\.\d*)?)\s*\)", r"\1", value)
    return re.sub(r"\s+", "", value)


def _ic_series(factor_values: pd.Series, labels: pd.Series, method: str) -> pd.Series:
    frame = pd.concat([factor_values, labels], axis=1, keys=["factor", "label"]).dropna()
    if frame.empty:
        return pd.Series(dtype=float)
    if "datetime" in frame.index.names:
        group_level: int | str = "datetime"
    else:
        group_level = 0
    values: list[float] = []
    dates: list[Any] = []
    for timestamp, group in frame.groupby(level=group_level, sort=True):
        if len(group) < 3:
            continue
        correlation = group["factor"].corr(group["label"], method=method)
        if pd.notna(correlation):
            dates.append(timestamp)
            values.append(float(correlation))
    return pd.Series(values, index=pd.Index(dates, name=str(group_level)), dtype=float)


def compute_factor_ic_metrics(factor_values: pd.Series, labels: pd.Series) -> dict[str, Any]:
    """Compute cross-sectional IC and RankIC statistics without Qlib state."""

    ic_series = _ic_series(factor_values, labels, method="pearson")
    rank_ic_series = _ic_series(factor_values, labels, method="spearman")
    ic_mean = float(ic_series.mean()) if not ic_series.empty else 0.0
    ic_std = float(ic_series.std()) if not ic_series.empty else 0.0
    rank_ic_mean = float(rank_ic_series.mean()) if not rank_ic_series.empty else 0.0
    rank_ic_std = float(rank_ic_series.std()) if not rank_ic_series.empty else 0.0
    return {
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "ir": ic_mean / ic_std if ic_std > 1e-8 else 0.0,
        "rank_ic_mean": rank_ic_mean,
        "rank_ic_std": rank_ic_std,
        "rank_ir": rank_ic_mean / rank_ic_std if rank_ic_std > 1e-8 else 0.0,
        "ic_win_rate": float((ic_series > 0).mean()) if not ic_series.empty else 0.0,
        "sample_count": int(len(ic_series)),
    }


def calculate_factor_ic_from_qlib(
    *,
    data_api: Any,
    formula: str,
    instruments: str,
    start_date: str,
    end_date: str,
    label_expr: str,
) -> dict[str, Any]:
    """Evaluate a factor and forward-return label through a Qlib data API."""

    normalized_formula = preprocess_factor_formula(formula)
    normalized_label = preprocess_factor_formula(label_expr)
    if not normalized_formula or not normalized_label:
        raise ValueError("formula and label_expr are required")
    instrument_pool = data_api.instruments(instruments)
    instrument_list = list(
        data_api.list_instruments(
            instruments=instrument_pool,
            start_time=start_date,
            end_time=end_date,
            as_list=True,
        )
        or []
    )
    if not instrument_list:
        return {
            "status": "insufficient_data",
            "reason": f"instrument pool '{instruments}' is empty",
            "metrics": {},
        }
    factor_frame = data_api.features(
        instrument_list,
        [normalized_formula],
        start_time=start_date,
        end_time=end_date,
    )
    label_frame = data_api.features(
        instrument_list,
        [normalized_label],
        start_time=start_date,
        end_time=end_date,
    )
    factor_values = factor_frame.iloc[:, 0] if isinstance(factor_frame, pd.DataFrame) else factor_frame
    labels = label_frame.iloc[:, 0] if isinstance(label_frame, pd.DataFrame) else label_frame
    aligned = pd.concat([factor_values, labels], axis=1, keys=["factor", "label"]).dropna()
    if aligned.empty:
        return {
            "status": "insufficient_data",
            "reason": "factor and label have no overlapping observations",
            "metrics": {},
        }
    metrics = compute_factor_ic_metrics(aligned["factor"], aligned["label"])
    return {
        "status": "success",
        "success": True,
        "metrics": metrics,
        "details": {
            "formula": normalized_formula,
            "instruments": instruments,
            "start_date": start_date,
            "end_date": end_date,
            "label_expr": normalized_label,
            "stock_count": len(instrument_list),
            "sample_count": metrics["sample_count"],
            "data_records": int(len(aligned)),
        },
    }


def _qlib_provider_uri(provider_uri: str | None) -> str:
    return str(
        provider_uri
        or os.getenv("QLIB_PROVIDER_URI")
        or os.getenv("FTU_QLIB_CN_DATA")
        or "data/qlib/cn_data"
    )


def _load_qlib_data_api(provider_uri: str, region: str) -> Any:
    global _QLIB_STATE
    import qlib

    state = (str(Path(provider_uri).expanduser().resolve()), region.lower())
    if _QLIB_STATE != state:
        qlib.init(provider_uri=state[0], region=state[1])
        _QLIB_STATE = state
    from qlib.data import D

    return D


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(ys) < 2 or len(xs) != len(ys):
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    denom = math.sqrt(var_x * var_y)
    if denom == 0:
        return None
    return cov / denom


def _returns_by_date(history: list[dict]) -> dict[str, float]:
    rows = sorted(
        (row for row in history if row.get("date") and row.get("close") is not None),
        key=lambda row: str(row["date"]),
    )
    returns: dict[str, float] = {}
    for prev, cur in zip(rows, rows[1:]):
        prev_close = safe_float(prev.get("close"))
        cur_close = safe_float(cur.get("close"))
        if prev_close and cur_close:
            returns[str(cur["date"])] = (cur_close - prev_close) / prev_close
    return returns


def summarize_market_history(
    rows: list[dict[str, Any]],
    *,
    symbol: str = "",
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """Return deterministic volume, volatility and drawdown metrics for OHLCV."""

    frame = pd.DataFrame(rows or [])
    if frame.empty or "date" not in frame or "close" not in frame:
        return {
            "status": "insufficient_data",
            "symbol": symbol,
            "observations": 0,
            "annualized_volatility": None,
        }
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in ("close", "volume"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date")
    if start_date:
        frame = frame[frame["date"] >= pd.Timestamp(start_date)]
    if end_date:
        frame = frame[frame["date"] <= pd.Timestamp(end_date)]
    if frame.empty:
        return {
            "status": "insufficient_data",
            "symbol": symbol,
            "observations": 0,
            "annualized_volatility": None,
        }
    closes = frame["close"].astype(float)
    log_returns = (closes / closes.shift(1)).where(closes.shift(1) > 0)
    log_returns = log_returns.apply(lambda value: math.log(value) if pd.notna(value) and value > 0 else float("nan")).dropna()
    volume = frame["volume"].dropna() if "volume" in frame else pd.Series(dtype=float)
    running_max = closes.cummax()
    drawdown = (closes / running_max - 1.0).min()
    return {
        "status": "success" if len(frame) >= 2 else "insufficient_data",
        "symbol": symbol,
        "observations": int(len(frame)),
        "average_volume": float(volume.mean()) if not volume.empty else None,
        "median_volume": float(volume.median()) if not volume.empty else None,
        "annualized_volatility": float(log_returns.std(ddof=1) * math.sqrt(252)) if len(log_returns) >= 2 else None,
        "max_drawdown": float(drawdown) if pd.notna(drawdown) else None,
        "first_close": float(closes.iloc[0]),
        "last_close": float(closes.iloc[-1]),
        "actual_min_date": frame["date"].iloc[0].date().isoformat(),
        "actual_max_date": frame["date"].iloc[-1].date().isoformat(),
    }


def analytics_find_price_extreme(
    rows: list[dict[str, object]],
    *,
    symbol: str,
    start_date: str,
    end_date: str,
    field: str = "high",
    direction: str = "max",
) -> dict[str, object]:
    """Find a deterministic price extreme from already-loaded OHLCV rows."""

    normalized_field = str(field or "high")
    normalized_direction = str(direction or "max").lower()
    usable_rows = [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("date")
        and row.get(normalized_field) is not None
        and start_date <= str(row.get("date"))[:10] <= end_date
        and safe_float(row.get(normalized_field)) is not None
    ]
    if not usable_rows:
        return {
            "status": "unavailable",
            "symbol": symbol,
            "field": normalized_field,
            "direction": normalized_direction,
            "reason": "no rows in requested window",
        }

    def key(row: dict[str, object]) -> float:
        value = safe_float(row.get(normalized_field))
        return float(value) if value is not None else float("nan")

    selected = (
        min(usable_rows, key=key)
        if normalized_direction == "min"
        else max(usable_rows, key=key)
    )
    return {
        "status": "success",
        "symbol": symbol,
        "field": normalized_field,
        "direction": normalized_direction,
        "start_date": start_date,
        "end_date": end_date,
        "actual_date": str(selected["date"])[:10],
        "value": float(key(selected)),
        "source": "derived_from_ohlcv",
    }


def _history_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        return _history_rows(payload.get("rows") or payload.get("data") or payload.get("result"))
    return []


def register(mcp) -> None:

    @mcp.tool()
    async def analytics_get_correlation(
        symbols: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
        lookback: int = 14,
        interval: str = "1d",
        region: str = "US",
    ) -> dict:
        histories: dict[str, list[dict]] = {}
        returns: dict[str, dict[str, float]] = {}
        for symbol in symbols:
            history = await fetch_with_failover(
                "ohlcv",
                "ohlcv",
                region=region,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                interval=interval,
            )
            histories[symbol] = history
            returns[symbol] = _returns_by_date(history)

        matrix: dict[str, dict[str, float | None]] = {}
        observations: dict[str, dict[str, int]] = {}
        for left in symbols:
            matrix[left] = {}
            observations[left] = {}
            for right in symbols:
                common_dates = sorted(set(returns[left]) & set(returns[right]))
                xs = [returns[left][d] for d in common_dates]
                ys = [returns[right][d] for d in common_dates]
                corr = _pearson(xs, ys)
                matrix[left][right] = None if corr is None else round(corr, 6)
                observations[left][right] = len(common_dates)

        return {
            "symbols": symbols,
            "start_date": start_date,
            "end_date": end_date,
            "interval": interval,
            "region": region,
            "matrix": matrix,
            "observations": observations,
        }

    @mcp.tool()
    async def analytics_summarize_market_series(
        symbols: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
        region: str = "US",
    ) -> dict[str, Any]:
        summaries: list[dict[str, Any]] = []
        for symbol in symbols:
            payload = await fetch_with_failover(
                "ohlcv",
                "ohlcv",
                region=region,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                interval="1d",
            )
            summaries.append(
                summarize_market_history(
                    _history_rows(payload),
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                )
            )
        statuses = [str(item.get("status")) for item in summaries]
        return {
            "status": "success" if summaries and all(status == "success" for status in statuses) else "partial",
            "region": region,
            "start_date": start_date,
            "end_date": end_date,
            "summaries": summaries,
        }

    @mcp.tool()
    async def analytics_calculate_ic(
        formula: str,
        instruments: str = "csi300",
        start_date: str = "2020-01-01",
        end_date: str = "2023-12-31",
        label_expr: str = "Ref($close, -2)/Ref($close, -1) - 1",
        provider_uri: str | None = None,
        region: str = "CN",
    ) -> dict[str, Any]:
        """Calculate factor IC, RankIC, IR and win rate from Qlib data."""

        normalized_region = str(region or "CN").upper()
        if normalized_region != "CN":
            return {
                "status": "unsupported_region",
                "reason": "Qlib IC calculation currently uses the CN provider",
                "region": normalized_region,
            }
        resolved_provider = _qlib_provider_uri(provider_uri)
        if not Path(resolved_provider).expanduser().exists():
            return {
                "status": "data_unavailable",
                "reason": "Qlib provider directory does not exist",
                "provider_uri": resolved_provider,
            }
        try:
            data_api = _load_qlib_data_api(resolved_provider, normalized_region)
            return calculate_factor_ic_from_qlib(
                data_api=data_api,
                formula=formula,
                instruments=instruments,
                start_date=start_date,
                end_date=end_date,
                label_expr=label_expr,
            )
        except ImportError:
            return {
                "status": "dependency_unavailable",
                "reason": "Qlib is required for analytics_calculate_ic",
            }
        except Exception as exc:
            return {
                "status": "error",
                "error_type": type(exc).__name__,
                "reason": str(exc),
            }

    @mcp.tool()
    async def analytics_factor_analysis(
        codes: list[str],
        factor_name: str,
        start_date: str,
        end_date: str,
        source: str = "auto",
        top_n: int = 10,
        bottom_n: int = 10,
        region: str = "US",
    ) -> Any:
        return await call_external_tool(
            "vibe_trading",
            "factor_analysis",
            codes=codes,
            factor_name=factor_name,
            start_date=start_date,
            end_date=end_date,
            source=source,
            top_n=top_n,
            bottom_n=bottom_n,
        )
