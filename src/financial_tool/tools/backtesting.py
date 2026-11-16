"""Deterministic OHLCV rule-strategy backtesting plus the legacy run-dir path.

The local engine deliberately keeps the small, transparent Top-K-style metric
surface used by the Qlib helper, while making rule based benchmark questions
executable without a separate trading MCP service.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from ..connectors.base import fetch_with_failover
from ..errors import SourceUnavailableError, UnsupportedRegionError
from ._external import call_external_tool


RULE_STRATEGIES = frozenset(
    {
        "buy_and_hold",
        "moving_average_crossover",
        "bollinger_breakout",
        "bollinger_mean_reversion",
        "momentum_rebalance",
    }
)


def _normalized_strategy(strategy: str) -> str:
    normalized = str(strategy or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "buyhold": "buy_and_hold",
        "ma_crossover": "moving_average_crossover",
        "moving_average": "moving_average_crossover",
        "bollinger_reversion": "bollinger_mean_reversion",
        "momentum": "momentum_rebalance",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in RULE_STRATEGIES:
        raise ValueError(f"unsupported strategy '{strategy}'; choose one of {sorted(RULE_STRATEGIES)}")
    return normalized


def _positive_int(value: int, name: str, *, minimum: int = 1) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}") from exc
    if parsed < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return parsed


def _parse_ohlcv(
    ohlcv_rows: list[dict[str, Any]], start_date: str | None, end_date: str | None
) -> tuple[pd.DataFrame, pd.Timestamp | None, pd.Timestamp | None]:
    if not isinstance(ohlcv_rows, list) or not ohlcv_rows:
        raise ValueError("ohlcv_rows must contain at least one OHLCV row")
    frame = pd.DataFrame(ohlcv_rows)
    if "date" not in frame.columns:
        raise ValueError("ohlcv_rows requires a date field")
    if "close" not in frame.columns:
        raise ValueError("ohlcv_rows requires a close field")
    if "symbol" not in frame.columns:
        frame["symbol"] = "ASSET"
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if frame["date"].isna().any():
        raise ValueError("ohlcv_rows contains an invalid date")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    if frame["close"].isna().any() or (frame["close"] <= 0).any():
        raise ValueError("ohlcv_rows close values must be positive numbers")
    frame["symbol"] = frame["symbol"].astype(str).str.strip()
    if not frame["symbol"].ne("").all():
        raise ValueError("ohlcv_rows symbol values must be non-empty")

    def parse_boundary(value: str | None, name: str) -> pd.Timestamp | None:
        if value is None:
            return None
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            raise ValueError(f"{name} must be a valid YYYY-MM-DD date")
        return pd.Timestamp(parsed)

    start = parse_boundary(start_date, "start_date")
    end = parse_boundary(end_date, "end_date")
    if start is not None and end is not None and start > end:
        raise ValueError("start_date must be on or before end_date")
    # Keep pre-start rows as indicator warm-up.  Evaluation is sliced only
    # after signals and the one-bar execution lag have been calculated.
    if end is not None:
        frame = frame[frame["date"] <= end]
    if frame.empty:
        raise ValueError("no OHLCV rows remain in the requested date range")
    return (
        frame.sort_values(["date", "symbol"]).drop_duplicates(["date", "symbol"], keep="last"),
        start,
        end,
    )


def _weights_from_signal(signal: pd.DataFrame) -> pd.DataFrame:
    active_count = signal.sum(axis=1).replace(0.0, np.nan)
    return signal.div(active_count, axis=0).fillna(0.0)


def _bollinger_signal(
    prices: pd.DataFrame, *, window: int, multiplier: float, mean_reversion: bool
) -> pd.DataFrame:
    middle = prices.rolling(window=window, min_periods=window).mean()
    deviation = prices.rolling(window=window, min_periods=window).std(ddof=0)
    upper = middle + multiplier * deviation
    lower = middle - multiplier * deviation
    signal = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for symbol in prices.columns:
        position = 0.0
        for timestamp in prices.index:
            close = prices.at[timestamp, symbol]
            mid = middle.at[timestamp, symbol]
            if not (pd.notna(close) and pd.notna(mid)):
                signal.at[timestamp, symbol] = position
                continue
            if mean_reversion:
                if close < lower.at[timestamp, symbol]:
                    position = 1.0
                elif close >= mid:
                    position = 0.0
            else:
                if close > upper.at[timestamp, symbol]:
                    position = 1.0
                elif close < mid:
                    position = 0.0
            signal.at[timestamp, symbol] = position
    return signal


def _momentum_weights(
    prices: pd.DataFrame, *, lookback: int, top_k: int, rebalance_frequency: int
) -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    momentum = prices.pct_change(periods=lookback, fill_method=None)
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    current = pd.Series(0.0, index=prices.columns)
    rebalance_dates: list[pd.Timestamp] = []
    for index, timestamp in enumerate(prices.index):
        if index >= lookback and (index - lookback) % rebalance_frequency == 0:
            scores = momentum.loc[timestamp].dropna().sort_values(ascending=False)
            selected = list(scores.head(min(top_k, len(scores))).index)
            current = pd.Series(0.0, index=prices.columns)
            if selected:
                current.loc[selected] = 1.0 / len(selected)
            rebalance_dates.append(pd.Timestamp(timestamp))
        weights.loc[timestamp] = current
    return weights, rebalance_dates


def _metrics(portfolio_returns: pd.Series, weight_changes: pd.DataFrame) -> dict[str, Any]:
    returns = portfolio_returns.astype(float).fillna(0.0)
    observations = len(returns)
    wealth = (1.0 + returns).cumprod()
    cumulative_return = float(wealth.iloc[-1] - 1.0) if observations else 0.0
    annual_return = float((1.0 + cumulative_return) ** (252.0 / max(1, observations)) - 1.0)
    annual_volatility = float(returns.std(ddof=0) * math.sqrt(252.0)) if observations > 1 else 0.0
    drawdown = wealth.div(wealth.cummax()).sub(1.0)
    max_drawdown = float(drawdown.min()) if observations else 0.0
    turnover = float(weight_changes.abs().sum(axis=1).sum() / 2.0)
    trades = int((weight_changes.abs().sum(axis=1) > 1e-12).sum())
    sharpe = float(annual_return / annual_volatility) if annual_volatility > 1e-12 else 0.0
    return {
        "cumulative_return": cumulative_return,
        "cum_return": cumulative_return,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "annual_vol": annual_volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "trades": trades,
        "turnover": turnover,
        "observations": int(observations),
    }


def run_rule_strategy_backtest(
    *,
    ohlcv_rows: list[dict[str, Any]],
    strategy: str,
    start_date: str | None = None,
    end_date: str | None = None,
    fast_window: int = 20,
    slow_window: int = 50,
    bollinger_window: int = 20,
    bollinger_stddev: float = 2.0,
    momentum_lookback: int = 20,
    top_k: int = 1,
    rebalance_frequency: int = 5,
) -> dict[str, Any]:
    """Backtest a transparent long-only rule using close-to-close OHLCV returns.

    Signals are calculated at each close and returns apply the *previous*
    session's target weights, so neither the local engine nor benchmark tests
    can accidentally trade on a same-bar signal.
    """

    normalized_strategy = _normalized_strategy(strategy)
    strategy_parameters: dict[str, Any] = {"execution_lag_bars": 1}
    fast = slow = bollinger = momentum = selected_count = rebalance = 0
    stddev = 0.0
    if normalized_strategy == "moving_average_crossover":
        fast = _positive_int(fast_window, "fast_window")
        slow = _positive_int(slow_window, "slow_window")
        if fast >= slow:
            raise ValueError("fast_window must be smaller than slow_window")
        strategy_parameters.update({"fast_window": fast, "slow_window": slow})
    elif normalized_strategy in {"bollinger_breakout", "bollinger_mean_reversion"}:
        bollinger = _positive_int(bollinger_window, "bollinger_window")
        try:
            stddev = float(bollinger_stddev)
        except (TypeError, ValueError) as exc:
            raise ValueError("bollinger_stddev must be a positive number") from exc
        if not math.isfinite(stddev) or stddev <= 0:
            raise ValueError("bollinger_stddev must be a positive number")
        strategy_parameters.update({"bollinger_window": bollinger, "bollinger_stddev": stddev})
    elif normalized_strategy == "momentum_rebalance":
        momentum = _positive_int(momentum_lookback, "momentum_lookback")
        selected_count = _positive_int(top_k, "top_k")
        rebalance = _positive_int(rebalance_frequency, "rebalance_frequency")
        strategy_parameters.update(
            {
                "momentum_lookback": momentum,
                "top_k": selected_count,
                "rebalance_frequency": rebalance,
            }
        )

    frame, evaluation_start, _evaluation_end = _parse_ohlcv(ohlcv_rows, start_date, end_date)
    prices = frame.pivot(index="date", columns="symbol", values="close").sort_index().ffill()
    if prices.empty:
        raise ValueError("no usable close prices are available")
    returns = prices.pct_change(fill_method=None).fillna(0.0)

    rebalance_dates: list[pd.Timestamp] = []
    if normalized_strategy == "buy_and_hold":
        target_weights = pd.DataFrame(1.0 / len(prices.columns), index=prices.index, columns=prices.columns)
    elif normalized_strategy == "moving_average_crossover":
        target_weights = _weights_from_signal(
            (prices.rolling(fast, min_periods=fast).mean() > prices.rolling(slow, min_periods=slow).mean()).astype(float)
        )
    elif normalized_strategy == "bollinger_breakout":
        target_weights = _weights_from_signal(
            _bollinger_signal(prices, window=bollinger, multiplier=stddev, mean_reversion=False)
        )
    elif normalized_strategy == "bollinger_mean_reversion":
        target_weights = _weights_from_signal(
            _bollinger_signal(prices, window=bollinger, multiplier=stddev, mean_reversion=True)
        )
    else:
        target_weights, rebalance_dates = _momentum_weights(
            prices, lookback=momentum, top_k=selected_count, rebalance_frequency=rebalance
        )

    # The one-bar shift is the no-lookahead invariant.
    realized_returns = (target_weights.shift(1).fillna(0.0) * returns).sum(axis=1)
    evaluation_mask = pd.Series(True, index=prices.index)
    if evaluation_start is not None:
        evaluation_mask &= prices.index >= evaluation_start
    evaluation_returns = realized_returns.loc[evaluation_mask]
    if evaluation_returns.empty:
        raise ValueError("no OHLCV rows remain in the requested evaluation date range")
    all_weight_changes = target_weights.diff()
    all_weight_changes.iloc[0] = target_weights.iloc[0]
    metrics = _metrics(evaluation_returns, all_weight_changes.loc[evaluation_mask])
    return {
        "status": "success",
        "success": True,
        "strategy": normalized_strategy,
        "symbols": [str(symbol) for symbol in prices.columns],
        "start_date": str(evaluation_returns.index.min().date()),
        "end_date": str(prices.index.max().date()),
        "warmup_observations": int(len(prices) - len(evaluation_returns)),
        "parameters": strategy_parameters,
        "metrics": metrics,
        "rebalance_dates": [str(timestamp.date()) for timestamp in rebalance_dates if timestamp in evaluation_returns.index],
    }


def warmup_start_date(
    *,
    strategy: str,
    start_date: str | None,
    fast_window: int = 20,
    slow_window: int = 50,
    bollinger_window: int = 20,
    momentum_lookback: int = 20,
) -> str | None:
    """Return a conservative calendar start date for indicator warm-up."""

    if not start_date:
        return None
    parsed = pd.to_datetime(start_date, errors="coerce")
    if pd.isna(parsed):
        return start_date
    normalized_strategy = _normalized_strategy(strategy)
    if normalized_strategy == "moving_average_crossover":
        bars = max(1, int(fast_window), int(slow_window))
    elif normalized_strategy in {"bollinger_breakout", "bollinger_mean_reversion"}:
        bars = max(1, int(bollinger_window))
    elif normalized_strategy == "momentum_rebalance":
        bars = max(1, int(momentum_lookback))
    else:
        return str(pd.Timestamp(parsed).date())
    # Three calendar days per bar safely spans weekends and common holidays.
    return str((pd.Timestamp(parsed) - pd.Timedelta(days=bars * 3 + 7)).date())


async def _fetch_ohlcv_rows(
    symbols: list[str], *, start_date: str | None, end_date: str | None, region: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for symbol in symbols:
        history = await fetch_with_failover(
            "ohlcv",
            "ohlcv",
            region=region,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval="1d",
        )
        source_rows = history.get("rows", []) if isinstance(history, dict) else history
        for row in source_rows or []:
            if isinstance(row, dict):
                rows.append({**row, "symbol": row.get("symbol") or symbol})
    return rows


def register(mcp) -> None:

    @mcp.tool()
    async def backtest_run_strategy(
        run_dir: str | None = None,
        strategy: str = "buy_and_hold",
        symbols: list[str] | None = None,
        ohlcv_rows: list[dict[str, Any]] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        fast_window: int = 20,
        slow_window: int = 50,
        bollinger_window: int = 20,
        bollinger_stddev: float = 2.0,
        momentum_lookback: int = 20,
        top_k: int = 1,
        rebalance_frequency: int = 5,
        region: str = "US",
    ) -> Any:
        """Run a local OHLCV rule backtest, or retain legacy run-dir execution."""

        if ohlcv_rows is not None:
            try:
                return run_rule_strategy_backtest(
                    ohlcv_rows=ohlcv_rows,
                    strategy=strategy,
                    start_date=start_date,
                    end_date=end_date,
                    fast_window=fast_window,
                    slow_window=slow_window,
                    bollinger_window=bollinger_window,
                    bollinger_stddev=bollinger_stddev,
                    momentum_lookback=momentum_lookback,
                    top_k=top_k,
                    rebalance_frequency=rebalance_frequency,
                )
            except ValueError as exc:
                return {"status": "invalid_request", "success": False, "reason": str(exc)}
        normalized_symbols = [str(symbol).strip() for symbol in (symbols or []) if str(symbol).strip()]
        if normalized_symbols:
            try:
                fetch_start_date = warmup_start_date(
                    strategy=strategy,
                    start_date=start_date,
                    fast_window=fast_window,
                    slow_window=slow_window,
                    bollinger_window=bollinger_window,
                    momentum_lookback=momentum_lookback,
                )
                rows = await _fetch_ohlcv_rows(
                    normalized_symbols, start_date=fetch_start_date, end_date=end_date, region=region
                )
                return run_rule_strategy_backtest(
                    ohlcv_rows=rows,
                    strategy=strategy,
                    start_date=start_date,
                    end_date=end_date,
                    fast_window=fast_window,
                    slow_window=slow_window,
                    bollinger_window=bollinger_window,
                    bollinger_stddev=bollinger_stddev,
                    momentum_lookback=momentum_lookback,
                    top_k=top_k,
                    rebalance_frequency=rebalance_frequency,
                )
            except ValueError as exc:
                return {"status": "invalid_request", "success": False, "reason": str(exc)}
            except (SourceUnavailableError, UnsupportedRegionError) as exc:
                return {
                    "status": "unavailable",
                    "success": False,
                    "reason": str(exc),
                    "symbols": normalized_symbols,
                }
            except Exception as exc:
                return {
                    "status": "failed",
                    "success": False,
                    "reason": str(exc),
                    "symbols": normalized_symbols,
                }
        if run_dir:
            # Compatibility for existing Vibe-Trading/Top-K prepared runs.
            return await call_external_tool("vibe_trading", "backtest", run_dir=run_dir)
        return {
            "status": "invalid_request",
            "success": False,
            "reason": "provide ohlcv_rows, symbols, or a legacy run_dir",
        }
