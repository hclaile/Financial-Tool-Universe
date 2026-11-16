"""On-chain connector — blockchain data access via public RPC and block explorers."""

from __future__ import annotations

from typing import Any

import httpx

from .base import BaseConnector, register_connector

# Public RPC endpoints (no API key required for basic queries)
_CHAIN_RPC = {
    "eth": "https://eth.llamarpc.com",
    "btc": "https://blockstream.info/api",
    "sol": "https://api.mainnet-beta.solana.com",
}


class OnchainConnector(BaseConnector):
    name = "onchain"
    actions = frozenset({"balance"})

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._rpc_overrides = self._config.get("rpc_urls", {})

    def _rpc_url(self, chain: str) -> str:
        return self._rpc_overrides.get(chain, _CHAIN_RPC.get(chain, ""))

    async def fetch(self, action: str, **kwargs: Any) -> Any:
        dispatch = {
            "balance": self._balance,
        }
        handler = dispatch.get(action)
        if handler is None:
            raise ValueError(f"Onchain: unsupported action '{action}'")
        return await handler(**kwargs)

    async def _balance(self, address: str, chain: str = "eth", **_: Any) -> dict:
        rpc = self._rpc_url(chain)
        if not rpc:
            return {"address": address, "chain": chain, "balance": 0, "token": "native", "timestamp": ""}

        try:
            if chain == "eth":
                return await self._eth_balance(address, rpc)
            elif chain == "btc":
                return await self._btc_balance(address, rpc)
            elif chain == "sol":
                return await self._sol_balance(address, rpc)
            else:
                return {"address": address, "chain": chain, "balance": 0, "token": "native", "timestamp": ""}
        except Exception as exc:
            return {"address": address, "chain": chain, "balance": 0, "token": "native", "timestamp": "", "error": str(exc)}

    async def _eth_balance(self, address: str, rpc: str) -> dict:
        payload = {
            "jsonrpc": "2.0",
            "method": "eth_getBalance",
            "params": [address, "latest"],
            "id": 1,
        }
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(rpc, json=payload)
            resp.raise_for_status()
            data = resp.json()
        hex_balance = data.get("result", "0x0")
        wei = int(hex_balance, 16)
        eth_balance = wei / 1e18
        return {
            "address": address,
            "chain": "eth",
            "balance": eth_balance,
            "token": "ETH",
            "timestamp": "",
        }

    async def _btc_balance(self, address: str, rpc: str) -> dict:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(f"{rpc}/address/{address}")
            resp.raise_for_status()
            data = resp.json()
        sats = data.get("chain_stats", {}).get("funded_txo_sum", 0) - data.get("chain_stats", {}).get("spent_txo_sum", 0)
        return {
            "address": address,
            "chain": "btc",
            "balance": sats / 1e8,
            "token": "BTC",
            "timestamp": "",
        }

    async def _sol_balance(self, address: str, rpc: str) -> dict:
        payload = {
            "jsonrpc": "2.0",
            "method": "getBalance",
            "params": [address],
            "id": 1,
        }
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(rpc, json=payload)
            resp.raise_for_status()
            data = resp.json()
        lamports = data.get("result", {}).get("value", 0)
        return {
            "address": address,
            "chain": "sol",
            "balance": lamports / 1e9,
            "token": "SOL",
            "timestamp": "",
        }

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    _CHAIN_RPC["eth"],
                    json={"jsonrpc": "2.0", "method": "eth_blockNumber", "params": [], "id": 1},
                )
                return resp.status_code == 200
        except Exception:
            return False


register_connector("onchain", OnchainConnector)
