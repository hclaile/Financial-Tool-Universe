"""Search tools — cross-source data search, Tavily web search, Google web search."""

from __future__ import annotations

import ipaddress
import logging
from datetime import datetime, timezone
from urllib.parse import urlsplit

import httpx

from ..connectors.base import fetch_with_failover
from ..errors import SourceUnavailableError

logger = logging.getLogger("financial_tool.tools.search")

_JINA_PREFIX = "https://r.jina.ai/"
_READ_TIMEOUT = 30
_MAX_PAGE_CHARS = 8000


def _url_allowed(url: str) -> tuple[bool, str]:
    try:
        parsed = urlsplit(url.strip())
    except ValueError:
        return False, "target URL is not allowed"

    if parsed.scheme.lower() not in {"http", "https"}:
        return False, "target URL is not allowed"
    if not parsed.hostname:
        return False, "target URL is not allowed"
    if parsed.username or parsed.password:
        return False, "target URL is not allowed"

    host = parsed.hostname.rstrip(".").lower()
    if host == "localhost" or host.endswith(".localhost") or host.endswith(".local"):
        return False, "target URL is not allowed"

    ip_host = host.split("%", 1)[0]
    try:
        ip = ipaddress.ip_address(ip_host)
    except ValueError:
        return True, ""

    if (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
        or not ip.is_global
    ):
        return False, "target URL is not allowed"
    return True, ""


def _extract_title(markdown: str) -> str:
    for line in markdown.splitlines():
        if line.startswith("Title:"):
            return line[6:].strip()
    return ""


async def _read_url_as_markdown(url: str, max_chars: int = _MAX_PAGE_CHARS) -> dict:
    target_url = url.strip()
    allowed, error = _url_allowed(target_url)
    if not allowed:
        return {"status": "error", "url": target_url, "error": error}

    try:
        async with httpx.AsyncClient(timeout=_READ_TIMEOUT) as client:
            resp = await client.get(
                f"{_JINA_PREFIX}{target_url}",
                headers={"Accept": "text/markdown"},
            )
        if resp.status_code != 200:
            return {
                "status": "error",
                "url": target_url,
                "error": f"Jina Reader returned {resp.status_code}: {resp.text[:500]}",
            }

        text = resp.text
        original_length = len(text)
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n\n... (truncated, total {original_length} chars)"

        return {
            "status": "ok",
            "title": _extract_title(resp.text),
            "url": target_url,
            "content": text,
            "length": original_length,
        }
    except httpx.TimeoutException:
        return {"status": "error", "url": target_url, "error": f"Request timed out ({_READ_TIMEOUT}s)"}
    except Exception as exc:
        return {"status": "error", "url": target_url, "error": str(exc)}


def register(mcp) -> None:

    @mcp.tool()
    async def search_data(
        query: str,
        data_types: list[str] | None = None,
        limit: int = 10,
        region: str = "CN",
    ) -> list[dict]:
        return await fetch_with_failover(
            "search_results", "search", region=region, query=query, limit=limit,
        )

    @mcp.tool()
    async def search_web(
        query: str,
        max_results: int = 5,
        search_depth: str = "advanced",
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
    ) -> dict:
        try:
            return await fetch_with_failover(
                "web_search_results",
                "web_search",
                region="GLOBAL",
                query=query,
                max_results=max_results,
                search_depth=search_depth,
                include_domains=include_domains,
                exclude_domains=exclude_domains,
            )
        except SourceUnavailableError as exc:
            logger.error("All web search sources failed: %s", exc)
            return {
                "query": query,
                "answer": "",
                "results": [],
                "source": "none",
                "error": f"All web search sources failed. {exc}",
            }

    @mcp.tool()
    async def search_web_read(
        query: str,
        max_results: int = 5,
        read_top_n: int = 3,
        max_chars_per_page: int = _MAX_PAGE_CHARS,
        search_depth: str = "advanced",
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
    ) -> dict:
        search_result = await search_web(
            query=query,
            max_results=max_results,
            search_depth=search_depth,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
        )

        pages = []
        for item in search_result.get("results", [])[: max(read_top_n, 0)]:
            url = item.get("url", "")
            if not url:
                continue
            page = await _read_url_as_markdown(url, max_chars=max_chars_per_page)
            parsed_url = urlsplit(url)
            page["read_status"] = page.get("status", "")
            page["source_url"] = page.get("url") or url
            page["source_domain"] = (parsed_url.hostname or "").lower()
            page["retrieved_at"] = datetime.now(timezone.utc).isoformat()
            page["search_title"] = item.get("title", "")
            page["search_content"] = item.get("content", "")
            page["search_score"] = item.get("score")
            pages.append(page)

        return {
            "query": query,
            "answer": search_result.get("answer", ""),
            "search_source": search_result.get("source", ""),
            "results_considered": len(search_result.get("results", [])),
            "pages_read": pages,
            "status": "ok" if pages else "no_pages_read",
            "search_error": search_result.get("error", ""),
        }
