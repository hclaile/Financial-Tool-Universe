"""Exercise tool families through the public MCP stdio protocol."""

import asyncio
import os
import sys
from pathlib import Path

import httpx
import pytest
import jsonschema
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from finance_tool_universe.server import create_http_app, create_server

ROOT = Path(__file__).resolve().parents[1]


def test_unified_mcp_catalog_and_local_calls(tmp_path):
    csv_path = tmp_path / "prices.csv"
    parquet_path = tmp_path / "prices.parquet"
    csv_path.write_text("date,symbol,open,high,low,close,volume\n2024-01-02,AAPL,10,11,9,10,100\n")

    async def run():
        env = dict(os.environ)
        env["FTU_CONFIG"] = str(ROOT / "config.example.yaml")
        env["PYTHONPATH"] = str(ROOT / "src")
        params = StdioServerParameters(command=sys.executable, args=["-m", "finance_tool_universe.server"], env=env)
        async with stdio_client(params) as streams:
            async with ClientSession(*streams) as session:
                initialized = await session.initialize()
                assert initialized.capabilities.resources is None
                assert initialized.capabilities.prompts is None
                listed = await session.list_tools()
                names = {tool.name for tool in listed.tools}
                assert len(names) == 116
                assert {"backtest_run_strategy", "backtest_model_predictions", "import_data", "market_data_get_history", "market_data_get_akshare_stock_price", "crypto_list_assets", "crawler_reddit_extractor", "training_get_task", "training_list_tasks"} <= names
                assert "derivatives_get_options" not in names
                assert "crypto_get_history" not in names
                assert "fmp_company_profile" not in names
                assert "calculate_ic" not in names
                assert all(
                    tool.description and tool.inputSchema.get("type") == "object"
                    and tool.outputSchema and tool.outputSchema.get("type") == "object"
                    for tool in listed.tools
                )
                schemas = {tool.name: tool.outputSchema for tool in listed.tools}
                assert schemas["market_data_get_akshare_stock_price"]["properties"]["price"]["type"] == "number"
                native = await session.call_tool("backtest_run_strategy", {
                    "strategy": "buy_and_hold",
                    "ohlcv_rows": [
                        {"date": "2024-01-02", "symbol": "AAPL", "close": 10},
                        {"date": "2024-01-03", "symbol": "AAPL", "close": 11},
                    ],
                })
                assert not native.isError, native
                jsonschema.validate(native.structuredContent, schemas["backtest_run_strategy"])

                imported = await session.call_tool("import_data", {
                    "input": str(csv_path), "output": str(parquet_path),
                })
                assert not imported.isError, imported
                jsonschema.validate(imported.structuredContent, schemas["import_data"])
                assert parquet_path.is_file()

                invalid_price = await session.call_tool("market_data_get_exact_price", {
                    "symbol": "AAPL", "target_date": "not-a-date",
                })
                assert not invalid_price.isError, invalid_price
                jsonschema.validate(invalid_price.structuredContent, schemas["market_data_get_exact_price"])

                empty_risk = await session.call_tool("risk_get_var", {"portfolio": []})
                assert not empty_risk.isError, empty_risk
                jsonschema.validate(empty_risk.structuredContent, schemas["risk_get_var"])

                missing_key = await session.call_tool("crypto_list_assets", {
                    "output_path": str(tmp_path / "assets.json"),
                })
                assert missing_key.isError
                assert "FMP" in missing_key.content[0].text

    asyncio.run(run())


def test_http_ip_allowlist(monkeypatch):
    with pytest.raises(ValueError, match="server.allowed_ips"):
        create_http_app(None, {"server": {"host": "0.0.0.0", "allowed_ips": []}})

    app = create_http_app(None, {
        "server": {"host": "0.0.0.0", "allowed_ips": ["192.0.2.0/24"]},
    })

    async def run():
        transport = httpx.ASGITransport(app=app, client=("203.0.113.7", 50000))
        async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
            response = await client.get("/mcp")
            assert response.status_code == 403

    asyncio.run(run())

    # Allowed clients reach the MCP transport (the bare GET lacks required
    # Accept headers and is rejected by MCP, after passing the IP guard).
    monkeypatch.setenv("FTU_CONFIG", str(ROOT / "config.example.yaml"))
    active = create_http_app(create_server(), {
        "server": {"host": "0.0.0.0", "allowed_ips": ["192.0.2.0/24"]},
    })

    async def allowed():
        async with active.router.lifespan_context(active):
            transport = httpx.ASGITransport(app=active, client=("192.0.2.7", 50000))
            async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
                response = await client.get("/mcp")
                assert response.status_code == 406

    asyncio.run(allowed())
