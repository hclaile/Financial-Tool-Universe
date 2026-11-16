"""External MCP connector — proxy requests to other MCP servers."""

from __future__ import annotations

import os
from typing import Any

import httpx

from .base import BaseConnector, register_connector


class ExternalMCPConnector(BaseConnector):
    name = "external_mcp"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._server_url = self._config.get("url", "")
        self._auth_token = self._config.get("auth_token", "")

    async def fetch(self, action: str, **kwargs: Any) -> Any:
        server = kwargs.pop("server", None)
        server_url, auth_token = self._resolve_server(server)
        if not server_url:
            raise RuntimeError("External MCP: no server URL configured")

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"

        # Build MCP tool call request
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": action,
                "arguments": kwargs,
            },
            "id": 1,
        }

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                self._mcp_endpoint(server_url),
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()

        if "error" in data:
            raise RuntimeError(f"External MCP error: {data['error']}")

        return data.get("result", {})

    def _resolve_server(self, server: str | None) -> tuple[str, str]:
        if not server:
            return os.path.expandvars(self._server_url), self._auth_token

        servers = self._config.get("servers", {})
        cfg = servers.get(server, {}) if isinstance(servers, dict) else {}
        url = cfg.get("url", "")
        auth_token = cfg.get("auth_token", "")
        auth_token_env = cfg.get("auth_token_env", "")
        if auth_token_env:
            auth_token = os.getenv(auth_token_env, auth_token)
        return os.path.expandvars(url), auth_token

    @staticmethod
    def _mcp_endpoint(server_url: str) -> str:
        url = server_url.rstrip("/")
        if url.endswith("/mcp"):
            return url
        return f"{url}/mcp"

    async def health_check(self, server: str | None = None) -> bool:
        server_url, auth_token = self._resolve_server(server)
        if not server_url:
            return False
        headers = {"Content-Type": "application/json"}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "params": {},
            "id": 1,
        }
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    self._mcp_endpoint(server_url),
                    json=payload,
                    headers=headers,
                )
                return resp.status_code == 200
        except httpx.HTTPError:
            return False


register_connector("external_mcp", ExternalMCPConnector)
