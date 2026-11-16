"""Document tools — local file reading."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from ..connectors.base import get_connector
from ._external import call_external_tool
from ._region import require_region
from ._blocking import run_blocking


def register(mcp) -> None:

    @mcp.tool()
    async def mineru_pdf_to_json(
        pdf_path: str, output_path: str | None = None,
        api_key: str | None = None, endpoint: str | None = None,
    ) -> dict:
        def execute():
            from . import mineru_pdf_to_json as engine
            arguments = {"pdf_path": pdf_path, "output_path": output_path, "api_key": api_key}
            if endpoint is not None:
                arguments["endpoint"] = endpoint
            return engine.run(SimpleNamespace(**arguments))
        return await run_blocking(execute, name="mineru_pdf_to_json", output_path=output_path)

    @mcp.tool()
    async def report_generate(
        llm_json: dict[str, Any], output_path: str,
        data_files: list[str] | None = None, chart_config: dict[str, Any] | None = None,
        template_path: str | None = None, report_dir: str | None = None,
    ) -> dict:
        def execute():
            from . import report as engine
            return engine.run_generate_report(SimpleNamespace(
                llm_json=llm_json, output_path=output_path, data_files=data_files,
                chart_config=chart_config, template_path=template_path, report_dir=report_dir,
            ))
        return await run_blocking(execute, name="report_generate", output_path=output_path)

    @mcp.tool()
    async def documents_read(path: str, format: str | None = None) -> dict:
        conn = get_connector("documents")
        return await conn.fetch("read", path=path, format=format)

    @mcp.tool()
    async def documents_get_earnings_transcript(
        identifier: str,
        key_dev_id: str | None = None,
        region: str = "US",
    ) -> Any:
        require_region(region, allowed={"US"}, data_type="earnings_transcript")
        return await call_external_tool(
            "spglobal",
            "get_earnings_transcript_from_identifiers",
            identifier=identifier,
            key_dev_id=key_dev_id,
        )
