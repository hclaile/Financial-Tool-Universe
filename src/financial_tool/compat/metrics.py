"""Deterministic technical and portfolio metric calculations."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

try:
    import talib
except Exception:  # pragma: no cover - environment-dependent optional backend
    talib = None


def compute_technical_indicators(price_history: pd.DataFrame) -> dict[str, Any]:
    if price_history.empty:
        return {}
    frame = price_history.sort_values("date")
    close = pd.to_numeric(frame["close"], errors="coerce")
    volume = frame["volume"].astype(float) if "volume" in frame else pd.Series(dtype=float)
    returns = close.pct_change()
    running_max = close.cummax()
    drawdown = close / running_max - 1.0
    base_indicators = {
        "return_1d": _pct_return(close, 1),
        "return_5d": _pct_return(close, 5),
        "return_20d": _pct_return(close, 20),
        "volatility_20d": _std_annualized(returns.tail(20)),
        "drawdown": _safe_float(drawdown.iloc[-1]),
        "volume_zscore": _volume_zscore(volume),
    }
    pandas_indicators = _pandas_technical_indicators(frame, close)
    talib_indicators = _talib_technical_indicators(frame, close)
    merged_indicators, indicator_sources = _merge_indicator_priority(
        primary=talib_indicators,
        fallback=pandas_indicators,
    )
    if merged_indicators:
        backend = "talib" if talib_indicators and all(
            source == "talib" for source in indicator_sources.values()
        ) else "talib+pandas" if talib_indicators else "pandas"
        return {
            "technical_backend": backend,
            **base_indicators,
            **merged_indicators,
            "indicator_sources": indicator_sources,
        }


def compute_technical_indicator_frame(
    price_history: pd.DataFrame,
    *,
    sma_windows: list[int] | tuple[int, ...] | None = None,
    ema_windows: list[int] | tuple[int, ...] | None = None,
) -> pd.DataFrame:
    """Return deterministic per-observation technical indicators for an OHLCV frame."""
    if price_history.empty or not {"date", "close"}.issubset(price_history.columns):
        return pd.DataFrame()
    frame = price_history.copy().sort_values("date").reset_index(drop=True)
    close = pd.to_numeric(frame["close"], errors="coerce")
    high = pd.to_numeric(frame.get("high", close), errors="coerce")
    low = pd.to_numeric(frame.get("low", close), errors="coerce")
    volume = pd.to_numeric(frame.get("volume", pd.Series(index=frame.index, dtype=float)), errors="coerce")
    returns = close.pct_change()
    frame["return_1d"] = returns
    frame["return_5d"] = close.pct_change(5)
    frame["return_20d"] = close.pct_change(20)
    frame["volatility_20d"] = returns.rolling(20).std(ddof=1) * math.sqrt(252)
    frame["drawdown"] = close / close.cummax() - 1.0
    requested_sma_windows = _normalized_indicator_windows(
        sma_windows,
        defaults=(5, 10, 20, 50, 60, 100, 200),
    )
    requested_ema_windows = _normalized_indicator_windows(
        ema_windows,
        defaults=(9, 12, 20, 26, 50),
    )
    for window in requested_sma_windows:
        frame[f"sma_{window}"] = close.rolling(window).mean()
    for span in requested_ema_windows:
        frame[f"ema_{span}"] = close.ewm(span=span, adjust=False).mean()
    for required_span in (12, 26):
        if f"ema_{required_span}" not in frame:
            frame[f"ema_{required_span}"] = close.ewm(span=required_span, adjust=False).mean()
    if "sma_20" not in frame:
        frame["sma_20"] = close.rolling(20).mean()
    macd = frame["ema_12"] - frame["ema_26"]
    frame["macd"] = macd
    frame["macd_signal"] = macd.ewm(span=9, adjust=False).mean()
    frame["macd_hist"] = frame["macd"] - frame["macd_signal"]
    delta = close.diff()
    gains = delta.clip(lower=0).rolling(14).mean()
    losses = (-delta.clip(upper=0)).rolling(14).mean()
    relative_strength = gains / losses.replace(0, math.nan)
    frame["rsi_14"] = 100 - 100 / (1 + relative_strength)
    frame.loc[(losses == 0) & gains.notna(), "rsi_14"] = 100.0
    rolling_std = close.rolling(20).std(ddof=1)
    frame["bollinger_mid_20"] = frame["sma_20"]
    frame["bollinger_upper_20"] = frame["sma_20"] + 2 * rolling_std
    frame["bollinger_lower_20"] = frame["sma_20"] - 2 * rolling_std
    true_range = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1
    ).max(axis=1)
    frame["atr_14"] = true_range.rolling(14).mean()
    volume_std = volume.rolling(20).std(ddof=1)
    frame["volume_zscore_20"] = (volume - volume.rolling(20).mean()) / volume_std.replace(0, math.nan)
    return frame.replace([math.inf, -math.inf], math.nan)


def _normalized_indicator_windows(
    windows: list[int] | tuple[int, ...] | None,
    *,
    defaults: tuple[int, ...],
) -> tuple[int, ...]:
    values = [*defaults, *(windows or [])]
    normalized: list[int] = []
    for value in values:
        try:
            window = int(value)
        except (TypeError, ValueError):
            continue
        if 2 <= window <= 500 and window not in normalized:
            normalized.append(window)
    return tuple(normalized or defaults)


def _pandas_technical_indicators(frame: pd.DataFrame, close: pd.Series) -> dict[str, Any]:
    ma5 = close.rolling(5).mean().iloc[-1]
    ma20 = close.rolling(20).mean().iloc[-1]
    ma60 = close.rolling(60).mean().iloc[-1] if len(close) >= 60 else None
    ema12 = close.ewm(span=12, adjust=False).mean().iloc[-1] if len(close) >= 12 else None
    ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1] if len(close) >= 20 else None
    ema26 = close.ewm(span=26, adjust=False).mean().iloc[-1] if len(close) >= 26 else None
    macd, macd_signal, macd_hist = _macd_components(close)
    return {
        "moving_average_5": _safe_float(ma5),
        "moving_average_20": _safe_float(ma20),
        "moving_average_60": _safe_float(ma60),
        "sma_5": _safe_float(ma5),
        "sma_20": _safe_float(ma20),
        "sma_60": _safe_float(ma60),
        "ema_12": _safe_float(ema12),
        "ema_20": _safe_float(ema20),
        "ema_26": _safe_float(ema26),
        "price_to_ma20": _safe_float(close.iloc[-1] / ma20) if pd.notna(ma20) and ma20 else None,
        "rsi_14": _rsi(close, 14),
        "adx_14": _adx(frame, close, 14),
        "macd": macd,
        "macd_signal": macd_signal,
        "macd_hist": macd_hist,
    }


def _merge_indicator_priority(
    *,
    primary: dict[str, Any],
    fallback: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    keys = sorted(set(primary) | set(fallback))
    merged: dict[str, Any] = {}
    sources: dict[str, str] = {}
    for key in keys:
        primary_value = primary.get(key)
        if primary_value is not None:
            merged[key] = primary_value
            sources[key] = "talib"
            continue
        fallback_value = fallback.get(key)
        if fallback_value is not None:
            merged[key] = fallback_value
            sources[key] = "pandas"
        else:
            merged[key] = None
    return merged, sources


def _talib_technical_indicators(frame: pd.DataFrame, close: pd.Series) -> dict[str, Any]:
    if talib is None or len(close.dropna()) < 2:
        return {}
    close_values = close.astype(float).to_numpy()
    high_values = (
        pd.to_numeric(frame["high"], errors="coerce").astype(float).to_numpy()
        if "high" in frame
        else close_values
    )
    low_values = (
        pd.to_numeric(frame["low"], errors="coerce").astype(float).to_numpy()
        if "low" in frame
        else close_values
    )
    ma5 = talib.SMA(close_values, timeperiod=5)[-1]
    ma20 = talib.SMA(close_values, timeperiod=20)[-1]
    ma60 = talib.SMA(close_values, timeperiod=60)[-1] if len(close_values) >= 60 else math.nan
    ema12 = talib.EMA(close_values, timeperiod=12)[-1]
    ema20 = talib.EMA(close_values, timeperiod=20)[-1]
    ema26 = talib.EMA(close_values, timeperiod=26)[-1]
    macd, macd_signal, macd_hist = talib.MACD(
        close_values,
        fastperiod=12,
        slowperiod=26,
        signalperiod=9,
    )
    rsi14 = talib.RSI(close_values, timeperiod=14)[-1]
    adx14 = talib.ADX(high_values, low_values, close_values, timeperiod=14)[-1]
    return {
        "moving_average_5": _safe_float(ma5),
        "moving_average_20": _safe_float(ma20),
        "moving_average_60": _safe_float(ma60),
        "sma_5": _safe_float(ma5),
        "sma_20": _safe_float(ma20),
        "sma_60": _safe_float(ma60),
        "ema_12": _safe_float(ema12),
        "ema_20": _safe_float(ema20),
        "ema_26": _safe_float(ema26),
        "price_to_ma20": _safe_float(close.iloc[-1] / ma20) if pd.notna(ma20) and ma20 else None,
        "rsi_14": _safe_float(rsi14),
        "adx_14": _safe_float(adx14),
        "macd": _safe_float(macd[-1]),
        "macd_signal": _safe_float(macd_signal[-1]),
        "macd_hist": _safe_float(macd_hist[-1]),
    }


def compute_performance_metrics(
    portfolio_values: pd.Series,
    weights: dict[str, float] | None = None,
    periods_per_year: int = 252,
) -> dict[str, float | None]:
    values = portfolio_values.dropna().astype(float)
    if values.empty:
        return {}
    returns = values.pct_change().dropna()
    total_return = values.iloc[-1] / values.iloc[0] - 1.0
    n_days = max(len(values) - 1, 1)
    running_max = values.cummax()
    drawdown = values / running_max - 1.0
    mdd = drawdown.min()
    vol = returns.std(ddof=1) * math.sqrt(periods_per_year) if len(returns) > 1 else 0.0
    mean = returns.mean() if len(returns) else 0.0
    sharpe = mean / returns.std(ddof=1) * math.sqrt(periods_per_year) if len(returns) > 1 and returns.std(ddof=1) else None
    downside = returns[returns < 0]
    sortino = mean / downside.std(ddof=1) * math.sqrt(periods_per_year) if len(downside) > 1 and downside.std(ddof=1) else None
    arr = (1.0 + total_return) ** (periods_per_year / n_days) - 1.0
    calmar = arr / abs(mdd) if mdd < 0 else None
    entropy = compute_entropy(weights or {})
    return {
        "TR": round(float(total_return), 6),
        "ARR": round(float(arr), 6),
        "SR": _round_or_none(sharpe),
        "CR": _round_or_none(calmar),
        "SoR": _round_or_none(sortino),
        "MDD": round(float(mdd), 6),
        "Vol": round(float(vol), 6),
        "ENT": _round_or_none(entropy),
        "ENB": _round_or_none(math.exp(entropy) if entropy is not None else None),
    }


def compute_entropy(weights: dict[str, float]) -> float | None:
    positive = [float(weight) for weight in weights.values() if float(weight) > 0]
    total = sum(positive)
    if total <= 0:
        return 0.0
    probs = [weight / total for weight in positive]
    return -sum(prob * math.log(prob) for prob in probs)


def _pct_return(close: pd.Series, periods: int) -> float | None:
    if len(close) <= periods:
        return None
    return _safe_float(close.iloc[-1] / close.iloc[-periods - 1] - 1.0)


def _std_annualized(values: pd.Series) -> float | None:
    clean = values.dropna()
    if len(clean) < 2:
        return None
    return _safe_float(clean.std(ddof=1) * math.sqrt(252))


def _rsi(close: pd.Series, window: int) -> float | None:
    if len(close) <= window:
        return None
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean().iloc[-1]
    loss = (-delta.clip(upper=0)).rolling(window).mean().iloc[-1]
    if pd.isna(loss) or loss == 0:
        return 100.0
    return _safe_float(100 - 100 / (1 + gain / loss))


def _macd(close: pd.Series) -> float | None:
    macd, _, _ = _macd_components(close)
    return macd


def _macd_components(close: pd.Series) -> tuple[float | None, float | None, float | None]:
    if len(close) < 26:
        return None, None, None
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return (
        _safe_float(macd.iloc[-1]),
        _safe_float(signal.iloc[-1]),
        _safe_float(hist.iloc[-1]),
    )


def _adx(frame: pd.DataFrame, close: pd.Series, window: int) -> float | None:
    if len(close) <= window * 2 or not {"high", "low"}.issubset(frame.columns):
        return None
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    true_range = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(window).mean()
    plus_di = 100 * plus_dm.rolling(window).mean() / atr
    minus_di = 100 * minus_dm.rolling(window).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return _safe_float(dx.rolling(window).mean().iloc[-1])


def _volume_zscore(volume: pd.Series) -> float | None:
    if len(volume) < 20:
        return None
    window = volume.tail(20)
    std = window.std(ddof=1)
    if not std:
        return None
    return _safe_float((window.iloc[-1] - window.mean()) / std)


def _safe_float(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), 6)


def _round_or_none(value) -> float | None:
    return None if value is None else round(float(value), 6)
