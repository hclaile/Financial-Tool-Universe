"""One MCP endpoint for the financial tool catalog."""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import json
import sys
from typing import Any

from .config import apply_config, config_path, example_path, load_config
from .catalog import tools as catalog


def create_server():
    config = load_config()
    apply_config(config)
    from mcp.server import Server
    from mcp.types import Tool

    # Third-party imports may log to stdout; MCP stdio reserves it for JSON-RPC.
    with contextlib.redirect_stdout(sys.stderr):
        from financial_tool.registration import create_tool_server
        fin = create_tool_server()
    registered = {tool.name: tool for tool in fin._tool_manager.list_tools()}
    records = catalog()
    expected = {t["name"] for t in records}
    if set(registered) != expected:
        raise ValueError(
            f"Catalog/runtime tool mismatch: missing={expected-set(registered)}, "
            f"extra={set(registered)-expected}"
        )
    for entry in records:
        if registered[entry["name"]].parameters != entry["input_schema"]:
            raise ValueError(f"Catalog/runtime input schema mismatch: {entry['name']}")
    server = Server(str((config.get("server") or {}).get("name") or "finance-tool-universe"))
    role = str((config.get("permissions") or {}).get("default_role") or "admin")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name=entry["name"],
                description=entry["description"],
                inputSchema=entry["input_schema"],
                outputSchema=entry["output_schema"] or None,
            )
            for entry in records
            if fin._auth.allowed(role, entry["name"])
        ]

    @server.call_tool(validate_input=True)
    async def call_tool(name: str, arguments: dict[str, Any] | None):
        args = arguments or {}
        fin._auth.require(role, name)
        fin._rate_limiter.check("per_tool", name)
        result = await fin._tool_manager.call_tool(name, args, convert_result=False)
        serialized = json.loads(json.dumps(result, ensure_ascii=False, default=str))
        return serialized if isinstance(serialized, dict) else {"result": serialized}

    return server


async def run_stdio(server) -> None:
    from mcp.server import NotificationOptions
    from mcp.server.stdio import stdio_server

    async with stdio_server() as streams:
        await server.run(
            streams[0], streams[1],
            server.create_initialization_options(notification_options=NotificationOptions()),
        )


def create_http_app(server, config: dict[str, Any]):
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    from starlette.applications import Starlette
    from starlette.responses import PlainTextResponse
    from starlette.routing import Route

    settings = config.get("server") or {}
    host = str(settings.get("host") or "127.0.0.1")
    allowed = [ipaddress.ip_network(str(ip), strict=False) for ip in settings.get("allowed_ips") or []]
    if host not in {"127.0.0.1", "::1", "localhost"} and not allowed:
        raise ValueError("Non-loopback HTTP binding requires server.allowed_ips")
    manager = StreamableHTTPSessionManager(server, stateless=True)

    class MCPEndpoint:
        async def __call__(self, scope, receive, send):
            address = ipaddress.ip_address(scope["client"][0])
            permitted = (any(address in network for network in allowed) if allowed else address.is_loopback)
            if not permitted:
                await PlainTextResponse("Forbidden", status_code=403)(scope, receive, send)
                return
            await manager.handle_request(scope, receive, send)

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app):
        async with manager.run():
            yield

    app = Starlette(routes=[Route("/mcp", endpoint=MCPEndpoint(), methods=["GET", "POST", "DELETE"])], lifespan=lifespan)
    return app


def run_http(server, config: dict[str, Any]) -> None:
    import uvicorn
    settings = config.get("server") or {}
    host = str(settings.get("host") or "127.0.0.1")
    port = int(settings.get("port") or 8000)
    app = create_http_app(server, config)
    uvicorn.run(app, host=host, port=port)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Finance Tool Universe MCP server")
    parser.add_argument("--check", action="store_true", help="validate config and tool registration")
    parser.add_argument("--list-tools", action="store_true", help="print tool names to stdout")
    parser.add_argument("--init-config", action="store_true", help="copy the blank config template")
    args = parser.parse_args()
    if args.init_config:
        destination = config_path()
        with destination.open("x", encoding="utf-8") as config_file:
            config_file.write(example_path().read_text(encoding="utf-8"))
        destination.chmod(0o600)
        print(f"Created {destination}")
        return
    config = load_config()
    apply_config(config)
    server = create_server()
    transport = str((config.get("server") or {}).get("transport") or "stdio")
    if transport not in {"stdio", "http"}:
        raise ValueError("server.transport must be stdio or http")
    if args.check:
        if transport == "http":
            create_http_app(server, config)
        print(f"OK: {len(catalog())} tools registered", file=sys.stderr)
    elif args.list_tools:
        for tool in catalog():
            print(tool["name"])
    elif transport == "http":
        run_http(server, config)
    else:
        asyncio.run(run_stdio(server))


if __name__ == "__main__":
    main()
