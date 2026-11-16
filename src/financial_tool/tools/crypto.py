"""Crypto tools — cryptocurrency price data with failover."""

from __future__ import annotations

from types import SimpleNamespace

from ..connectors.base import fetch_with_failover
from ._blocking import run_blocking


def register(mcp) -> None:

    @mcp.tool()
    async def crypto_list_assets(output_path: str, limit: int = 100) -> dict:
        def execute():
            from . import crypto_assets as engine
            return engine.run(SimpleNamespace(output_path=output_path, limit=limit))
        return await run_blocking(execute, name="crypto_list_assets", output_path=output_path)

    @mcp.tool()
    async def crypto_get_price(symbol: str, vs_currency: str = "USDT") -> dict:
        return await fetch_with_failover(
            "crypto_price", "price", symbol=symbol, vs_currency=vs_currency,
        )
