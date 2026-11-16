"""Derivatives tools — local option pricing and remote volatility surfaces."""

from __future__ import annotations

import math
from typing import Any

from ._external import call_external_tool


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def register(mcp) -> None:

    @mcp.tool()
    async def derivatives_price_option(
        spot: float,
        strike: float,
        expiry_days: int,
        risk_free_rate: float = 0.03,
        volatility: float = 0.25,
        option_type: str = "call",
    ) -> dict:
        t = max(expiry_days, 0) / 365.0
        if spot <= 0 or strike <= 0 or volatility <= 0 or t <= 0:
            intrinsic = max(spot - strike, 0.0) if option_type == "call" else max(strike - spot, 0.0)
            return {
                "price": round(intrinsic, 6),
                "delta": None,
                "gamma": None,
                "theta": None,
                "vega": None,
                "rho": None,
                "method": "intrinsic_value",
            }

        d1 = (math.log(spot / strike) + (risk_free_rate + 0.5 * volatility**2) * t) / (
            volatility * math.sqrt(t)
        )
        d2 = d1 - volatility * math.sqrt(t)
        pdf_d1 = math.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi)

        if option_type == "put":
            price = strike * math.exp(-risk_free_rate * t) * _norm_cdf(-d2) - spot * _norm_cdf(-d1)
            delta = _norm_cdf(d1) - 1
            theta = (
                -spot * pdf_d1 * volatility / (2 * math.sqrt(t))
                + risk_free_rate * strike * math.exp(-risk_free_rate * t) * _norm_cdf(-d2)
            ) / 365
            rho = -strike * t * math.exp(-risk_free_rate * t) * _norm_cdf(-d2) / 100
        else:
            price = spot * _norm_cdf(d1) - strike * math.exp(-risk_free_rate * t) * _norm_cdf(d2)
            delta = _norm_cdf(d1)
            theta = (
                -spot * pdf_d1 * volatility / (2 * math.sqrt(t))
                - risk_free_rate * strike * math.exp(-risk_free_rate * t) * _norm_cdf(d2)
            ) / 365
            rho = strike * t * math.exp(-risk_free_rate * t) * _norm_cdf(d2) / 100

        gamma = pdf_d1 / (spot * volatility * math.sqrt(t))
        vega = spot * pdf_d1 * math.sqrt(t) / 100
        return {
            "price": round(price, 6),
            "delta": round(delta, 6),
            "gamma": round(gamma, 6),
            "theta": round(theta, 6),
            "vega": round(vega, 6),
            "rho": round(rho, 6),
            "method": "black_scholes",
        }

    @mcp.tool()
    async def derivatives_get_vol_surface(
        identifier: str,
        asset_class: str = "equity",
        currency_pair: str | None = None,
    ) -> Any:
        if asset_class == "fx":
            return await call_external_tool(
                "lseg",
                "fx_vol_surface",
                currency_pair=currency_pair or identifier,
            )
        return await call_external_tool("lseg", "equity_vol_surface", identifier=identifier)
