"""On-chain tools — blockchain analytics."""

from __future__ import annotations

from ..connectors.base import fetch_with_failover


def register(mcp) -> None:

    @mcp.tool()
    async def onchain_get_balance(address: str, chain: str = "eth") -> dict:
        return await fetch_with_failover(
            "onchain_balance", "balance", address=address, chain=chain,
        )
