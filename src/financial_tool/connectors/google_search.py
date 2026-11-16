"""Google search connector — web search via SerpAPI/Custom Search JSON API."""

from __future__ import annotations

import json
import os
from typing import Any
from urllib.parse import quote_plus

import httpx

from .base import BaseConnector, register_connector
from ..network_config import DEFAULT_USER_AGENT

# Supports Google Custom Search JSON API (requires API key + CX)
_GOOGLE_CSE_URL = "https://www.googleapis.com/customsearch/v1"


class GoogleSearchConnector(BaseConnector):
    name = "google_search"
    actions = frozenset({"web_search"})

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._api_key = self._config.get("api_key", "") or os.getenv("GOOGLE_SEARCH_API_KEY", "")
        self._cx = self._config.get("cx", "") or os.getenv("GOOGLE_SEARCH_CX", "")

    async def fetch(self, action: str, **kwargs: Any) -> Any:
        dispatch = {
            "web_search": self._web_search,
        }
        handler = dispatch.get(action)
        if handler is None:
            raise ValueError(f"GoogleSearch: unsupported action '{action}'")
        return await handler(**kwargs)

    async def _web_search(
        self,
        query: str,
        max_results: int = 5,
        **_: Any,
    ) -> dict:
        if self._api_key and self._cx:
            return await self._google_cse(query, max_results)
        # Fallback: use DuckDuckGo HTML search (no API key needed)
        return await self._ddg_fallback(query, max_results)

    async def _google_cse(self, query: str, max_results: int) -> dict:
        params: dict[str, Any] = {
            "key": self._api_key,
            "cx": self._cx,
            "q": query,
            "num": str(min(max_results, 10)),
        }
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(_GOOGLE_CSE_URL, params=params)
            resp.raise_for_status()
            data = resp.json()

        results = []
        for item in data.get("items", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "content": item.get("snippet", ""),
                "score": 1.0,
            })

        return {
            "query": query,
            "answer": "",
            "results": results,
            "source": "google_cse",
        }

    async def _ddg_fallback(self, query: str, max_results: int) -> dict:
        """DuckDuckGo Lite HTML search as free fallback (no key required).

        The Instant Answer API returns only abstract summaries, not real
        search result lists.  The Lite HTML endpoint yields actual results
        with titles and URLs, which is what benchmark queries need.
        """
        headers = {"User-Agent": DEFAULT_USER_AGENT}
        results = []
        try:
            # Use DuckDuckGo Lite HTML search for actual result links
            async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
                resp = await client.post(
                    "https://lite.duckduckgo.com/lite/",
                    data={"q": query, "kl": "us-en"},
                    headers=headers,
                )
                if resp.status_code == 200:
                    import re
                    # Parse the HTML response for result links and snippets
                    # DDG Lite format: <a class="result__a" href="...">Title</a>
                    # followed by <a class="result__snippet">Snippet</a>
                    text = resp.text
                    link_pattern = re.compile(
                        r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
                        re.DOTALL,
                    )
                    snippet_pattern = re.compile(
                        r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
                        re.DOTALL,
                    )
                    links = link_pattern.findall(text)
                    snippets = snippet_pattern.findall(text)
                    for i, (url, title) in enumerate(links[:max_results]):
                        content = snippets[i][0] if i < len(snippets) else ""
                        # Strip HTML tags from title and snippet
                        title = re.sub(r"<[^>]+>", "", title).strip()
                        content = re.sub(r"<[^>]+>", "", content).strip()
                        results.append({
                            "title": title[:100],
                            "url": url,
                            "content": content[:300],
                            "score": 1.0,
                        })

            if results:
                return {
                    "query": query,
                    "answer": "",
                    "results": results,
                    "source": "duckduckgo_lite",
                }

            # If DDG Lite yielded nothing, try the Instant Answer API as
            # last resort (returns only abstract, no result list).
            params: dict[str, Any] = {
                "q": query,
                "format": "json",
                "no_html": 1,
                "skip_disambig": 1,
            }
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(
                    "https://api.duckduckgo.com/",
                    params=params,
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()

            answer = data.get("AbstractText", "")
            for topic in data.get("RelatedTopics", [])[:max_results]:
                if isinstance(topic, dict) and topic.get("Text"):
                    results.append({
                        "title": topic.get("Text", "")[:100],
                        "url": topic.get("FirstURL", ""),
                        "content": topic.get("Text", ""),
                        "score": 1.0,
                    })
                    if len(results) >= max_results:
                        break

            return {
                "query": query,
                "answer": answer,
                "results": results,
                "source": "duckduckgo",
            }
        except Exception:
            return {
                "query": query,
                "answer": "",
                "results": [],
                "source": "google_search",
                "error": "No search backend available (configure Google CSE API key or Tavily)",
            }

    async def health_check(self) -> bool:
        if self._api_key and self._cx:
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    resp = await client.get(_GOOGLE_CSE_URL, params={
                        "key": self._api_key,
                        "cx": self._cx,
                        "q": "test",
                        "num": "1",
                    })
                    return resp.status_code == 200
            except Exception:
                return False
        # DDG fallback is always "available" but health is uncertain
        return True


register_connector("google_search", GoogleSearchConnector)
