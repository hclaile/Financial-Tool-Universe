"""Check catalog contracts and registered tool coverage."""

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import jsonschema
import pandas as pd

from finance_tool_universe.catalog import groups, load_catalog
from finance_tool_universe.server import catalog
from financial_tool.registration import create_tool_server
from financial_tool.connectors.base import resolve_sources
from financial_tool.connectors.fmp import FMPConnector
from financial_tool.connectors import akshare as akshare_module
from financial_tool.tools import crypto_assets, data_download, reddit
from financial_tool.tools import market_data as market_data_module
from financial_tool.tools._blocking import run_blocking
from financial_tool.tools.resolve_symbol import _match_alias

ROOT = Path(__file__).resolve().parents[1]


def test_catalog_replacements_and_shared_registry(monkeypatch):
    monkeypatch.setenv("FTU_CONFIG", str(ROOT / "config.example.yaml"))
    records = catalog()
    names = {entry["name"] for entry in records}
    unified = load_catalog()
    assert unified == json.loads((ROOT / "catalogs/tool_catalog.json").read_text())
    replacements = unified["replacements"]
    assert len(records) == len(names) == 116
    assert len(replacements) == 42
    assert "dispatch" not in unified
    registered = {tool.name: tool for tool in create_tool_server()._tool_manager.list_tools()}
    assert set(registered) == names
    assert all(registered[entry["name"]].parameters == entry["input_schema"] for entry in records)
    for entry in records:
        jsonschema.Draft202012Validator.check_schema(entry["input_schema"])
        jsonschema.Draft202012Validator.check_schema(entry["output_schema"])
        jsonschema.validate(entry["example"], entry["input_schema"])
        assert entry["output_schema"]["type"] == "object"
        assert entry["output_schema"]["description"]
        assert set(entry) == {"name", "description", "group", "input_schema", "output_schema", "example"}
    assert all(old not in names and set(new) <= names for old, new in replacements.items())
    assert "calendar_get_economic_events" not in names
    assert not (ROOT / "src/finance_tool_universe/workflows.py").exists()
    assert not (ROOT / "src/financial_tool/workflows").exists()
    membership = {name for group in groups() for name in group["tools"]}
    assert membership == names
    assert sum(len(group["tools"]) for group in groups()) == len(records)
    assert all(entry["group"] in unified["groups"] for entry in records)
    assert all(entry["group"] not in {"qlib", "\u6570\u636e\u83b7\u53d6"} for entry in records)
    functional_groups = {entry["name"]: entry["group"] for entry in records}
    assert functional_groups["calendar_get_earnings_transcript"] == "documents"
    assert functional_groups["calendar_get_analyst_estimates"] == "fundamentals"
    assert functional_groups["market_data_get_technical"] == "technical"
    assert functional_groups["fundamentals_get_enterprise_value"] == "valuation"
    assert functional_groups["flows_get_market_cap"] == "market_data"
    assert functional_groups["feature"] == "data_pipeline"


def test_crypto_history_uses_canonical_market_data_route(monkeypatch):
    calls = []
    async def fake_get(endpoint, params):
        calls.append((endpoint, params))
        return [{"date": "2024-01-02", "open": 1, "high": 2, "low": 1, "close": 2, "volume": 3}] * 600

    connector = FMPConnector({"api_key": "test-key"})
    monkeypatch.setattr(connector, "_get", fake_get)
    rows = asyncio.run(connector.fetch(
        "ohlcv", symbol="BTC-USD", start_date="2024-01-01", end_date="2024-01-03"
    ))
    assert calls == [("/historical-price-eod/full", {"symbol": "BTCUSD", "from": "2024-01-01", "to": "2024-01-03"})]
    assert len(rows) == 600
    assert rows[0]["symbol"] == "BTC-USD"
    assert resolve_sources("crypto_ohlcv", "GLOBAL") == ["fmp", "ccxt", "yfinance"]


def test_akshare_stock_price_uses_a_share_snapshot(monkeypatch):
    monkeypatch.setenv("FTU_CONFIG", str(ROOT / "config.example.yaml"))
    calls = []

    def snapshot():
        calls.append(1)
        return pd.DataFrame([{
            "\u4ee3\u7801": "600519", "\u540d\u79f0": "Moutai",
            "\u6700\u65b0\u4ef7": 1688.5, "\u6628\u6536": 1675.0,
            "\u4eca\u5f00": 1680.0, "\u6700\u9ad8": 1690.0,
            "\u6700\u4f4e": 1678.0, "\u6210\u4ea4\u91cf": 1234,
            "\u6da8\u8dcc\u989d": 13.5, "\u6da8\u8dcc\u5e45": 0.81,
        }])

    monkeypatch.setattr(akshare_module.ak, "stock_zh_a_spot_em", snapshot)
    async def inline_call(operation, *args, **kwargs):
        return operation(*args, **kwargs)

    monkeypatch.setattr(akshare_module.asyncio, "to_thread", inline_call)
    manager = create_tool_server()._tool_manager
    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(manager.call_tool(
            "market_data_get_akshare_stock_price", {"symbol": "600519.SH"},
            convert_result=False,
        ))
    finally:
        loop.close()
    assert calls == [1]
    assert result["status"] == "success"
    assert result["symbol"] == "600519.SH"
    assert result["provider_symbol"] == "600519"
    assert result["price"] == result["close"] == 1688.5
    assert result["source"] == "akshare"
    schema = next(t["output_schema"] for t in catalog() if t["name"] == "market_data_get_akshare_stock_price")
    jsonschema.validate(result, schema)


def test_akshare_stock_price_handles_missing_provider_values(monkeypatch):
    monkeypatch.setenv("FTU_CONFIG", str(ROOT / "config.example.yaml"))
    frame = pd.DataFrame([{
        "\u4ee3\u7801": "000001", "\u6700\u65b0\u4ef7": 12.34,
        "\u4eca\u5f00": "-", "\u6210\u4ea4\u91cf": None,
    }]).astype({"\u6700\u65b0\u4ef7": object})
    monkeypatch.setattr(akshare_module.ak, "stock_zh_a_spot_em", lambda: frame)

    async def inline_call(operation, *args, **kwargs):
        return operation(*args, **kwargs)

    monkeypatch.setattr(akshare_module.asyncio, "to_thread", inline_call)
    manager = create_tool_server()._tool_manager
    schema = next(t["output_schema"] for t in catalog() if t["name"] == "market_data_get_akshare_stock_price")
    loop = asyncio.new_event_loop()
    try:
        quote = loop.run_until_complete(manager.call_tool(
            "market_data_get_akshare_stock_price", {"symbol": "000001.SZ"},
            convert_result=False,
        ))
        missing = loop.run_until_complete(manager.call_tool(
            "market_data_get_akshare_stock_price", {"symbol": "600519.SH"},
            convert_result=False,
        ))
        frame.loc[0, "\u6700\u65b0\u4ef7"] = "-"
        no_price = loop.run_until_complete(manager.call_tool(
            "market_data_get_akshare_stock_price", {"symbol": "000001.SZ"},
            convert_result=False,
        ))
    finally:
        loop.close()

    assert quote["price"] == 12.34
    assert quote["open"] is None
    assert quote["volume"] is None
    assert missing["status"] == no_price["status"] == "unavailable"
    for result in (quote, missing, no_price):
        jsonschema.validate(result, schema)


def test_file_tool_schemas_match_completion_envelope():
    schemas = {entry["name"]: entry["output_schema"] for entry in catalog()}
    names = (
        "import_data", "feature", "train", "backtest_model_predictions",
        "talib_technical_indicators", "report_generate",
    )
    for name in names:
        result = asyncio.run(run_blocking(lambda: None, name=name, output_path=None))
        jsonschema.validate(result, schemas[name])


def test_launch_tools_return_task_id_and_status_endpoints_are_idempotent(monkeypatch):
    from financial_tool.tools import training

    # Missing repository fails loudly instead of leaving a dangling task.
    monkeypatch.setenv("FTU_ALPHAQCM_REPO", "/nonexistent/AlphaQCM")
    with pytest.raises(FileNotFoundError, match="AlphaQCM"):
        training.launch_training("train_qcm", model="qrdqn")

    # Status endpoints behave without any launched task.
    assert training.get_training_status("does-not-exist")["status"] == "not_found"
    assert training.list_training_tasks() == []


def test_training_command_builders_preserve_external_cli(monkeypatch, tmp_path):
    from financial_tool.tools import training

    repo = tmp_path / "AlphaSAGE"
    repo.mkdir()
    monkeypatch.setenv("FTU_ALPHASAGE_REPO", str(repo))

    cmd = training._build_gp_command(repo, {"config_params": {
        "instruments": "csi300", "seed": 0, "train_end_year": 2020,
        "freq": "day", "use_gpu": True,
    }})
    assert cmd[1:] == [
        str(repo / "train_GP.py"),
        "--instruments", "csi300",
        "--seed", "0",
        "--train-end-year", "2020",
        "--freq", "day",
        "--use-gpu",
    ]

    cmd = training._build_ppo_command(repo, {"config_params": {
        "seed": 0, "instruments": "csi300", "pool": 20, "steps": 200000,
    }})
    assert cmd[1:] == [
        str(repo / "train_ppo.py"),
        "--seed", "0",
        "--instruments", "csi300",
        "--pool", "20",
        "--steps", "200000",
    ]

    cmd = training._build_qcm_command(tmp_path / "qcm", {
        "model": "iqn", "seed": 7, "pool": 20, "std_lam": 0.5,
    })
    assert cmd[1:] == [
        str(tmp_path / "qcm" / "train_qcm.py"),
        "--model", "iqn",
        "--seed", "7",
        "--pool", "20",
        "--std-lam", "0.5",
    ]


def test_exact_price_success_matches_output_schema(monkeypatch):
    monkeypatch.setenv("FTU_CONFIG", str(ROOT / "config.example.yaml"))
    async def history(*args, **kwargs):
        return [{"date": "2024-01-02", "open": 10, "high": 12, "low": 9,
                 "close": 11, "volume": "1000"}]

    monkeypatch.setattr(market_data_module, "fetch_with_failover", history)
    manager = create_tool_server()._tool_manager
    result = asyncio.run(manager.call_tool(
        "market_data_get_exact_price",
        {"symbol": "AAPL", "target_date": "2024-01-02", "region": "US"},
        convert_result=False,
    ))
    schema = next(t["output_schema"] for t in catalog() if t["name"] == "market_data_get_exact_price")
    jsonschema.validate(result, schema)


def test_translated_company_names_keep_chinese_alias_lookup():
    matches = _match_alias("\u5e73\u5b89\u94f6\u884c")
    assert matches[0]["symbol"] == "000001.SZ"
    assert matches[0]["name"] == "Ping An Bank"


def test_crypto_list_preserves_old_listing_mode(monkeypatch, tmp_path):
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return [{"symbol": "BTCUSD"}, {"symbol": "ETHUSD"}]

    calls = []
    monkeypatch.setenv("FMP_API_KEY", "test-key")
    monkeypatch.setenv("FTU_FMP_BASE_URL", "https://example.test/stable")
    monkeypatch.setattr(crypto_assets.requests, "get", lambda url, **kwargs: (calls.append(url), Response())[1])
    output = tmp_path / "assets.json"
    result = crypto_assets.run(SimpleNamespace(limit=1, output_path=str(output)))
    assert calls == ["https://example.test/stable/cryptocurrency-list"]
    assert result["count"] == 1
    assert json.loads(output.read_text()) == [{"symbol": "BTCUSD"}]
    schema = next(t["output_schema"] for t in catalog() if t["name"] == "crypto_list_assets")
    jsonschema.validate(result, schema)


def test_qlib_dataset_tool_rejects_company_price_mode():
    with pytest.raises(ValueError, match="market_data_get_history"):
        data_download.run(SimpleNamespace(market="us", company="AAPL"))


def test_reddit_requires_credentials_before_loading_praw(monkeypatch, tmp_path):
    monkeypatch.delenv("REDDIT_CLIENT_ID", raising=False)
    monkeypatch.delenv("REDDIT_CLIENT_SECRET", raising=False)
    with pytest.raises(ValueError, match="Reddit credentials"):
        reddit.run(SimpleNamespace(keyword="bitcoin", output_path=str(tmp_path / "posts.json")))


def test_reddit_keeps_posts_and_top_comments(monkeypatch, tmp_path):
    comment = SimpleNamespace(author="reader", score=8, body="  useful   comment ", created_utc=1704153600)

    class Comments:
        def replace_more(self, **kwargs):
            assert kwargs == {"limit": 0}

        def list(self):
            return [comment]

    post = SimpleNamespace(
        title="Bitcoin", author="author", score=12, url="https://example.test/post",
        created_utc=datetime(2024, 1, 2, tzinfo=timezone.utc).timestamp(),
        subreddit=SimpleNamespace(display_name="finance"), selftext="post body",
        comments=Comments(),
    )
    calls = []

    class FakeReddit:
        def subreddit(self, name):
            assert name == "all"
            return self

        def search(self, **kwargs):
            calls.append(kwargs)
            return [post]

    monkeypatch.setenv("REDDIT_CLIENT_ID", "test-id")
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", "test-secret")
    monkeypatch.setenv("FTU_USER_AGENT", "test-agent")
    monkeypatch.setitem(sys.modules, "praw", SimpleNamespace(Reddit=lambda **kwargs: FakeReddit()))
    output = tmp_path / "reddit.json"
    result = reddit.run(SimpleNamespace(
        keyword="bitcoin", start_time="2024-01-01", end_time="2024-01-31",
        limit=10, output_path=str(output),
    ))
    assert calls[0]["time_filter"] == "all"
    assert result["count"] == 1
    assert result["posts"][0]["top_comments"][0]["text"] == "useful comment"
    assert json.loads(output.read_text()) == result["posts"]
    schema = next(t["output_schema"] for t in catalog() if t["name"] == "crawler_reddit_extractor")
    jsonschema.validate(result, schema)


def test_risk_var_excludes_missing_symbols_and_bounds_lookback(monkeypatch):
    """Symbols with no data must not drag down the equal-weight portfolio, and
    ``lookback`` must actually bound the return window used."""
    import asyncio as _asyncio

    from financial_tool.tools import risk as risk_module
    from financial_tool.registration import create_tool_server

    async def fake_fetch(data_type, action, region, symbol, **kwargs):
        if symbol in ("AAPL", "MSFT"):
            return [{"date": f"2024-01-{d:02d}", "close": 100 + d} for d in range(1, 21)]
        raise Exception("no data")

    monkeypatch.setattr(risk_module, "fetch_with_failover", fake_fetch)
    manager = create_tool_server()._tool_manager
    result = _asyncio.run(manager.call_tool(
        "risk_get_var",
        {"portfolio": ["AAPL", "MSFT", "UNKNOWN"], "lookback": 6, "confidence": 0.95},
        convert_result=False,
    ))
    assert result["status"] == "success"
    # The window used is 6 dates (lookback), not the full 19 returns.
    assert result["observations"] == 6
    assert result["method"] == "parametric_variance_covariance"


def test_risk_var_rejects_bad_confidence(monkeypatch):
    import asyncio as _asyncio
    import pytest as _pytest

    from financial_tool.registration import create_tool_server
    from mcp.server.fastmcp.exceptions import ToolError

    manager = create_tool_server()._tool_manager
    with _pytest.raises(ToolError, match="confidence must be one of"):
        _asyncio.run(manager.call_tool(
            "risk_get_var", {"portfolio": ["AAPL"], "confidence": 0.999},
            convert_result=False,
        ))


def test_derivatives_price_option_rejects_invalid_type_and_date():
    import asyncio as _asyncio
    import pytest as _pytest

    from financial_tool.registration import create_tool_server
    from mcp.server.fastmcp.exceptions import ToolError

    manager = create_tool_server()._tool_manager
    with _pytest.raises(ToolError, match="option_type must be"):
        _asyncio.run(manager.call_tool(
            "derivatives_price_option",
            {"spot": 100, "strike": 100, "expiry_days": 30, "option_type": "straddle"},
            convert_result=False,
        ))
    with _pytest.raises(ToolError, match="expiry_days must be"):
        _asyncio.run(manager.call_tool(
            "derivatives_price_option",
            {"spot": 100, "strike": 100, "expiry_days": 0},
            convert_result=False,
        ))


def test_derivatives_price_option_matches_call_put_parity():
    import asyncio as _asyncio

    from financial_tool.registration import create_tool_server

    manager = create_tool_server()._tool_manager
    args = {"spot": 100, "strike": 100, "expiry_days": 30,
            "risk_free_rate": 0.03, "volatility": 0.25}
    call = _asyncio.run(manager.call_tool(
        "derivatives_price_option", {**args, "option_type": "call"}, convert_result=False))
    put = _asyncio.run(manager.call_tool(
        "derivatives_price_option", {**args, "option_type": "put"}, convert_result=False))
    # Put-call parity: C - P = S - K*e^(-rT)
    import math
    T = 30 / 365.0
    parity = call["price"] - put["price"] - (100 - 100 * math.exp(-0.03 * T))
    assert abs(parity) < 1e-4


def test_search_read_url_guards_block_private_targets():
    import asyncio as _asyncio

    from financial_tool.tools import search

    for bad in (
        "http://127.0.0.1/x",
        "http://[::1]/x",
        "http://10.0.0.1/x",
        "http://192.168.1.1/x",
        "http://169.254.1.1/x",
        "http://localhost/x",
        "http://user:pass@example.com/x",
        "ftp://example.com/x",
    ):
        allowed, _ = search._url_allowed(bad)
        assert not allowed, f"{bad} must be rejected"

    assert search._url_allowed("https://example.com/x")[0] is True
    assert _asyncio.run(search._host_resolves_to_public("localhost")) is False
    assert _asyncio.run(search._host_resolves_to_public("127.0.0.1")) is False


def test_search_read_url_rejects_private_dns(monkeypatch):
    import asyncio as _asyncio

    from financial_tool.tools import search

    def fake_getaddrinfo(host, port, *args, **kwargs):
        return [(search.socket.AF_INET, search.socket.SOCK_STREAM, 6, "", ("192.168.1.10", 0))]

    monkeypatch.setattr(search.socket, "getaddrinfo", fake_getaddrinfo)
    assert _asyncio.run(search._host_resolves_to_public("internal.example")) is False
