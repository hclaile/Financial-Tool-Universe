"""Risk tools — VaR, Altman Z-Score, and risk metrics."""

from __future__ import annotations

import math

from ..connectors.base import fetch_with_failover
from ..connectors._numeric import safe_float


def register(mcp) -> None:

    @mcp.tool()
    async def risk_get_var(
        portfolio: list[str],
        confidence: float = 0.95,
        horizon_days: int = 1,
        start_date: str | None = None,
        end_date: str | None = None,
        lookback: int = 14,
        region: str = "US",
    ) -> dict:
        z_scores = {0.90: 1.282, 0.95: 1.645, 0.99: 2.326}
        if confidence not in z_scores:
            raise ValueError("confidence must be one of 0.90, 0.95, 0.99")
        if lookback < 5:
            raise ValueError("lookback must be at least 5")

        # Fetch historical returns for each position
        returns_data: dict[str, list[float]] = {}
        for symbol in portfolio:
            try:
                history = await fetch_with_failover(
                    "ohlcv", "ohlcv", region=region,
                    symbol=symbol, start_date=start_date, end_date=end_date, interval="1d",
                )
                if isinstance(history, list) and len(history) >= 2:
                    closes = [safe_float(h.get("close")) for h in history if safe_float(h.get("close"))]
                    daily_returns = [
                        (closes[i] - closes[i - 1]) / closes[i - 1]
                        for i in range(1, len(closes))
                        if closes[i - 1] != 0
                    ]
                    returns_data[symbol] = daily_returns[-lookback:]
            except Exception:
                returns_data[symbol] = []

        # Only symbols that actually returned data participate.  A symbol that
        # failed to fetch is excluded rather than silently treated as a zero-
        # return position, which previously under-weighted the whole portfolio.
        active_symbols = [symbol for symbol in portfolio if returns_data.get(symbol)]
        if not active_symbols:
            return {
                "portfolio": portfolio,
                "confidence": confidence,
                "horizon_days": horizon_days,
                "lookback": lookback,
                "var": None,
                "status": "insufficient_data",
                "message": "No historical data available for portfolio",
            }

        # Align returns by minimum common length across active symbols.
        min_len = min(len(returns_data[symbol]) for symbol in active_symbols)
        if min_len < 5:
            return {
                "portfolio": portfolio,
                "confidence": confidence,
                "horizon_days": horizon_days,
                "lookback": lookback,
                "var": None,
                "status": "insufficient_data",
                "message": f"Only {min_len} observations, need at least 5",
            }

        # Equal weight over the symbols that actually have data on that day.
        portfolio_returns = []
        for i in range(min_len):
            day_returns = [returns_data[symbol][i] for symbol in active_symbols]
            portfolio_returns.append(sum(day_returns) / len(day_returns))

        # Parametric VaR (variance-covariance method)
        mean_return = sum(portfolio_returns) / len(portfolio_returns)
        variance = sum((r - mean_return) ** 2 for r in portfolio_returns) / (len(portfolio_returns) - 1)
        std_dev = math.sqrt(variance)

        # Z-score for confidence level
        z = z_scores[confidence]

        # VaR = (mean - z * sigma) * sqrt(horizon) * portfolio_value
        # Assuming unit portfolio value (1.0), return as percentage
        daily_var_pct = mean_return - z * std_dev
        var_pct = daily_var_pct * math.sqrt(horizon_days)

        return {
            "portfolio": portfolio,
            "confidence": confidence,
            "horizon_days": horizon_days,
            "lookback": lookback,
            "var": round(abs(var_pct) * 100, 4),  # expressed as percentage loss
            "var_pct": round(var_pct, 6),
            "daily_std_dev": round(std_dev, 6),
            "daily_mean_return": round(mean_return, 6),
            "observations": min_len,
            "method": "parametric_variance_covariance",
            "status": "success",
        }

    @mcp.tool()
    async def risk_get_altman_z(
        symbol: str,
        period: str = "annual",
        limit: int = 10,
        region: str = "US",
        start_date: str | None = None,
        end_date: str | None = None,
        as_of_date: str | None = None,
    ) -> list[dict]:
        return await fetch_with_failover(
            "altman_z_score",
            "altman_z",
            region=region,
            symbol=symbol,
            period=period,
            limit=limit,
            start_date=start_date,
            end_date=end_date,
            as_of_date=as_of_date,
        )
