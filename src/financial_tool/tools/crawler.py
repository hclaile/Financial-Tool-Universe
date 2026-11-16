"""MCP tools migrated from the Qlib crawler module."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ..connectors.base import fetch_with_failover, get_source_connector
from ..errors import SourceUnavailableError
from ._region import require_region
from ._blocking import run_blocking


_FILING_DOWNLOAD_CACHE: dict[tuple[Any, ...], list[dict[str, Any]]] = {}


def _filing_download_cache_key(
    *,
    company_name: str,
    start_date: str,
    end_date: str,
    form_types: list[str] | None,
    limit: int,
    output_dir: str | None,
) -> tuple[Any, ...]:
    return (
        str(company_name),
        str(start_date),
        str(end_date),
        tuple(str(item).upper() for item in (form_types or [])),
        int(limit),
        str(output_dir or ""),
    )


async def _convert_downloaded_filings(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert downloaded filings while preserving the original path on failure."""
    sec_api = get_source_connector("sec_api")
    converted: list[dict[str, Any]] = []
    for record in records:
        item = dict(record)
        original_path = str(item.get("path") or "")
        item["original_path"] = original_path
        if not original_path:
            item["path"] = original_path
            item["conversion_status"] = "original"
            item["conversion_error"] = "Downloaded record has no file path"
            converted.append(item)
            continue

        json_path = str(Path(original_path).with_suffix(".json"))
        try:
            payload = await sec_api.fetch(
                "xbrl_to_json_file",
                input_path=original_path,
            )
            Path(json_path).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            item["path"] = json_path
            item["conversion_status"] = "converted"
            item["conversion_error"] = ""
        except Exception as exc:
            item["path"] = original_path
            item["conversion_status"] = "original"
            item["conversion_error"] = str(exc)
        converted.append(item)
    return converted


def register(mcp) -> None:
    @mcp.tool()
    async def crawler_reddit_extractor(
        keyword: str, output_path: str, start_time: str | None = None,
        end_time: str | None = None, limit: int = 100,
    ) -> dict:
        def execute():
            from . import reddit as engine
            return engine.run(SimpleNamespace(
                keyword=keyword, output_path=output_path,
                start_time=start_time, end_time=end_time, limit=limit,
            ))
        return await run_blocking(execute, name="crawler_reddit_extractor", output_path=output_path)

    @mcp.tool()
    async def macro_search_fred(
        query: str,
        limit: int = 10,
        search_type: str = "search",
        region: str = "US",
    ) -> Any:
        """Search FRED series metadata, or use its legacy note/code modes."""
        require_region(region, allowed={"US"}, data_type="macro_search_results")
        if search_type not in {"search", "search_note", "search_code"}:
            raise ValueError("search_type must be search, search_note, or search_code")
        if search_type != "search":
            return await get_source_connector("fred").fetch(
                search_type,
                query=query,
                limit=limit,
            )
        return await fetch_with_failover(
            "macro_search_results",
            "search",
            region=region,
            query=query,
            limit=limit,
        )

    @mcp.tool()
    async def search_scrape_urls(
        urls: list[str],
    ) -> dict:
        """Fetch public URLs and extract readable headings and text sections."""
        return await get_source_connector("web_scraper").fetch(
            "scrape_urls",
            urls=urls,
        )

    @mcp.tool()
    async def search_knowledge_graph(
        query: str,
        limit: int = 1,
    ) -> Any:
        """Search Wikipedia first, with Google Knowledge Graph as a fallback."""
        try:
            return await fetch_with_failover(
                "knowledge_entity",
                "search",
                region="GLOBAL",
                query=query,
                limit=limit,
            )
        except SourceUnavailableError as exc:
            return {"query": query, "results": [], "source": "none", "error": str(exc)}

    @mcp.tool()
    async def filings_download(
        company_name: str,
        start_date: str,
        end_date: str,
        form_types: list[str] | None = None,
        limit: int = 10,
        output_dir: str | None = None,
        region: str = "US",
    ) -> list[dict]:
        """Download SEC filings via SEC API, then fall back to official EDGAR."""
        require_region(region, allowed={"US"}, data_type="filing_downloads")
        download_kwargs = {
            "company_name": company_name,
            "start_date": start_date,
            "end_date": end_date,
            "form_types": form_types,
            "limit": limit,
            "output_dir": output_dir,
        }
        cache_key = _filing_download_cache_key(
            company_name=company_name,
            start_date=start_date,
            end_date=end_date,
            form_types=form_types,
            limit=limit,
            output_dir=output_dir,
        )
        if cache_key in _FILING_DOWNLOAD_CACHE:
            records = [dict(item) for item in _FILING_DOWNLOAD_CACHE[cache_key]]
        else:
            records = await fetch_with_failover(
                "filing_downloads",
                "download",
                region=region,
                **download_kwargs,
            )
            _FILING_DOWNLOAD_CACHE[cache_key] = [
                dict(item) for item in records if isinstance(item, dict)
            ]
        return await _convert_downloaded_filings(records)

    @mcp.tool()
    async def reference_search_wikipedia(
        query: str,
        limit: int = 3,
    ) -> dict:
        """Search Wikipedia and return short page extracts."""
        return await get_source_connector("wikipedia").fetch(
            "search",
            query=query,
            limit=limit,
        )
