"""Local dataset import, initialization, and feature construction tools."""

from __future__ import annotations

from types import SimpleNamespace

from ._blocking import run_blocking


def register(mcp) -> None:
    @mcp.tool()
    async def import_data(
        input: str,
        output: str | None = None,
        datetime_col: str = "date",
        instrument_col: str = "symbol",
        columns: list[str] | None = None,
        tz: str | None = None,
    ) -> dict:
        def execute():
            from . import import_data as engine
            return engine.run(SimpleNamespace(
                input=input, output=output, datetime_col=datetime_col,
                instrument_col=instrument_col,
                columns=columns if columns is not None else ["open", "high", "low", "close", "volume"], tz=tz,
            ))
        return await run_blocking(execute, name="import_data", output_path=output)

    @mcp.tool()
    async def data_download(
        market: str = "cn", provider_uri: str | None = None, force: bool = False,
    ) -> dict:
        def execute():
            from . import data_download as engine
            return engine.run(SimpleNamespace(market=market, provider_uri=provider_uri, force=force))
        return await run_blocking(execute, name="data_download", output_path=provider_uri)

    @mcp.tool()
    async def feature(config: str, output: str) -> dict:
        def execute():
            from . import feature as engine
            return engine.run(SimpleNamespace(config=config, output=output))
        return await run_blocking(execute, name="feature", output_path=output)
