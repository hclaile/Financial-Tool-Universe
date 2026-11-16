"""Tool registry — loads tool definitions from catalog and wires them to the MCP server."""

from __future__ import annotations

from typing import Any

from .errors import ToolNotFoundError

from finance_tool_universe.catalog import groups as catalog_groups, tools as catalog_tools


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, dict[str, Any]] = {}
        self._groups: dict[str, dict[str, Any]] = {}

    @classmethod
    def from_catalog(cls) -> ToolRegistry:
        reg = cls()
        reg._load_tools()
        reg._load_groups()
        return reg

    def _load_tools(self) -> None:
        for t in catalog_tools():
            self._tools[t["name"]] = t

    def _load_groups(self) -> None:
        for g in catalog_groups():
            self._groups[g["name"]] = g

    def get_tool(self, name: str) -> dict[str, Any]:
        if name not in self._tools:
            raise ToolNotFoundError(name)
        return self._tools[name]

    def get_group(self, name: str) -> dict[str, Any]:
        if name not in self._groups:
            raise ToolNotFoundError(name)
        return self._groups[name]

    def tool_metadata(self, name: str) -> dict[str, Any]:
        """Return the selection and schema metadata for a registered catalog tool."""

        tool = self.get_tool(name)
        return dict(tool)

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    def list_groups(self) -> list[str]:
        return list(self._groups.keys())

    def tools_for_group(self, group_name: str) -> list[dict[str, Any]]:
        group = self.get_group(group_name)
        return [self._tools[n] for n in group["tools"] if n in self._tools]

    def register_all(self, mcp) -> None:
        """Register all catalog tools with the FastMCP instance."""
        from .tools import market_data, fundamentals, valuation, filings, ownership
        from .tools import calendar, macro, news
        from .tools import funds_etf, derivatives, crypto, onchain, flows
        from .tools import risk, documents, search
        from .tools import analytics, backtesting, technical, fixed_income, rates
        from .tools import sentiment, resolve_symbol, crawler
        from .tools import data_pipeline, model_training, forecasting, benchmarks

        _modules = [
            market_data, fundamentals, valuation, filings, ownership,
            calendar, macro, news,
            funds_etf, derivatives, crypto, onchain, flows,
            risk, documents, search,
            analytics, backtesting, technical, fixed_income, rates,
            sentiment, resolve_symbol, crawler,
            data_pipeline, model_training, forecasting, benchmarks,
        ]
        for mod in _modules:
            if hasattr(mod, "register"):
                mod.register(mcp)
        self._apply_catalog_descriptions(mcp)

    def _apply_catalog_descriptions(self, mcp) -> None:
        """Keep FastMCP tool descriptions aligned with the canonical catalog.

        FastMCP has kept the tool manager private across releases.  The
        metadata update is therefore best-effort; schema parity is checked
        independently against the registered callables in tests.
        """

        manager = getattr(mcp, "_tool_manager", None)
        registered = getattr(manager, "_tools", None)
        if not isinstance(registered, dict):
            return
        for name, tool in registered.items():
            metadata = self._tools.get(name)
            if metadata is None:
                continue
            description = str(metadata.get("description") or "").strip()
            if description and hasattr(tool, "description"):
                tool.description = description
            # The callable owns parameter types and validation. Apply catalog
            # help consistently to every registered tool.
            documented = metadata["input_schema"].get("properties", {})
            actual = tool.parameters.get("properties", {})
            for argument, field in actual.items():
                help_text = documented.get(argument, {}).get("description")
                if help_text:
                    field["description"] = help_text
