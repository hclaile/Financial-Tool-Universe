# Finance Tool Universe

One MCP service exposes **116 financial tools**. All tools register through `financial_tool.registry` with one FastMCP manager, and their implementations live in `financial_tool.tools`. The [tool catalog](catalogs/tool_catalog.json) defines their public descriptions, schemas, examples, and functional groups. See the [catalog guide](docs/CATALOG.md) for response conventions and tool-name mappings.

## Get Started

Requires Python 3.10+. From the repository root, run:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e .
cp config.example.yaml config.yaml
# Edit config.yaml with only the credentials and paths you need
chmod 600 config.yaml
.venv/bin/finance-tool-universe --check
.venv/bin/finance-tool-universe
```

`config.yaml` is the user configuration file and is ignored by Git. Empty credential fields do not inherit keys from the environment. Local tools can run with the default configuration; tools that call external providers need the corresponding credentials.

Example MCP client configuration for stdio (replace `<PROJECT_DIR>` with the installation directory):

```json
{
  "mcpServers": {
    "finance-tool-universe": {
      "command": "<PROJECT_DIR>/.venv/bin/finance-tool-universe",
      "env": {"FTU_CONFIG": "<PROJECT_DIR>/config.yaml"}
    }
  }
}
```

`FTU_CONFIG` can point to another configuration file. Relative data paths resolve from that file's directory. Use `--list-tools` to list public tool names. Set `server.transport: http` to enable Streamable HTTP at `http://<host>:<port>/mcp`. Binding to a non-loopback address requires `server.allowed_ips` with client CIDR ranges. The default binds to `127.0.0.1`. If you use a reverse proxy, enforce access control there; the service checks the directly connected IP.

For an A-share stock price, call `market_data_get_akshare_stock_price` with `{"symbol":"600519.SH"}` or a bare six-digit code. The result includes `price`, `previous_close`, daily price fields, `volume`, and `retrieved_at`. This AKShare snapshot depends on upstream availability and may be delayed; `retrieved_at` records collection time.

After installation via wheel, you can also run `finance-tool-universe --init-config` in the working directory to generate a configuration template.

## Configuration fields

| Location | Purpose |
| --- | --- |
| `credentials` | API credentials for FMP, FRED, Tavily, Google, Reddit, MinerU, and other providers; empty values mean unconfigured |
| `network.user_agent` | Application identity for SEC, Wikipedia, and Reddit; use your own application name and contact address |
| `network.proxy_url` | Optional HTTP/HTTPS proxy; default is direct connection |
| `network.allowed_ips` | Provider-facing IP settings; also configure any key restrictions in the provider's console |
| `server.allowed_ips` | Client IP allowlist for MCP HTTP; unused by stdio |
| `paths` | Directories for data, outputs, Qlib data, and optional training backend |
| `sources` | Provider priorities, API URLs, and environment variable mappings |
| `permissions`, `cache`, `rate_limits` | Service role permissions, caching, and rate limiting; `default_role` is the execution role for the entire service, with the template defaulting to `admin` |

For FMP and SEC access, set `credentials.FMP_API_KEY` and `network.user_agent` in `config.yaml`, then enable the relevant sources. Keep a configuration containing real credentials readable only by its owner, and do not add it to Git.

## Optional tool dependencies

The core installation registers all 116 tools and supports local CSV/JSON-to-Parquet import. Some tools need optional SDKs, model libraries, or research projects when called:

```bash
.venv/bin/python -m pip install -e '.[qlib,crawlers,research,training]'
```

The `train_qcm`, `train_AFF`, `train_*_AlphaSAGE`, and `qlib_benchmark_runner` tools launch external research projects configured under `paths` as background tasks. Each launch returns a `task_id`; poll it with `training_get_task` and list all tasks with `training_list_tasks`. Those projects, model weights, and market data are not bundled, so clone each project into the configured path (or set its `FTU_*_REPO` variable) before calling its trainer. The former `backtest` tool is now `backtest_model_predictions`.

## Structure and validation

| Path | Content |
| --- | --- |
| `src/finance_tool_universe` | MCP entry point, configuration, and catalog loading |
| `src/financial_tool` | Unified implementation package for financial tools, connectors, and research algorithms |
| `src/financial_tool/tools` | Registration functions and implementations for all 116 tools; optional dependencies loaded at call time |
| `catalogs/tool_catalog.json` | Metadata for 116 tools, groups, and name migrations |
| `catalogs/data_types.json`, `company_aliases.json` | Data schemas and company aliases |
| `catalogs/source_priority.yaml`, `source_action_contract.yaml` | Provider priorities and supported actions |
| `docs/CATALOG.md` | Tool metadata, response conventions, and name mappings |
| `config.example.yaml` | Complete configuration template without actual credentials |

Provider column names and company aliases needed to match Chinese market data
are stored with Unicode escapes. Public descriptions and display names are in English.

`build/` is a temporary build output ignored by Git. Runtime code comes from `src/` or the installed wheel.

The integration tests connect through MCP stdio, check the 116-tool registry, call representative local tools, and verify missing-credential errors. Run:

```bash
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest -q
```

## Open source statement

The repository currently declares an MIT license. Confirm redistribution rights for included third-party modules before public release. `external/` is not versioned. Keep credentials out of Git and rotate any credentials that appeared in earlier repository history.
