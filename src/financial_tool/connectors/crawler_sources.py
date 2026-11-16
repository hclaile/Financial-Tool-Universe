"""Connectors migrated from the Qlib crawler module.

The crawler historically bundled several unrelated HTTP clients in one file.
These connectors keep the public capabilities while making credentials,
failover, and response shapes visible to the MCP catalog.
"""

from __future__ import annotations

import ipaddress
import os
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from .base import BaseConnector, decode_json_response, register_connector
from ..network_config import DEFAULT_USER_AGENT


def _config_value(config: dict[str, Any], key: str, env_key: str, default: str = "") -> str:
    return str(config.get(key, "") or os.getenv(env_key, "") or default)


def _raise_api_error(response: httpx.Response, source: str) -> None:
    if response.is_success:
        return
    headers = getattr(response, "headers", {}) or {}
    content_type = str(headers.get("content-type", "")) or "unknown"
    if getattr(response, "status_code", None) == 429:
        retry_after = str(headers.get("retry-after", "") or "").strip()
        retry_text = f" retry_after_seconds={retry_after}" if retry_after else ""
        raise RuntimeError(
            f"{source} HTTP error rate_limited status=429 "
            f"content_type={content_type}{retry_text}"
        )
    raise RuntimeError(
        f"{source} HTTP error status={response.status_code} content_type={content_type}"
    )


class SerpAPIConnector(BaseConnector):
    name = "serpapi"
    actions = frozenset({"web_search"})

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._api_key = _config_value(self._config, "api_key", "SERPAPI_API_KEY")
        self._base_url = self._config.get("base_url", "https://serpapi.com/search.json")

    async def fetch(self, action: str, **kwargs: Any) -> Any:
        if action != "web_search":
            raise ValueError(f"SerpAPI: unsupported action '{action}'")
        if not self._api_key:
            raise RuntimeError("SerpAPI API key is not configured")

        query = str(kwargs.get("query", "")).strip()
        if not query:
            raise ValueError("SerpAPI web_search requires query")
        max_results = max(1, min(int(kwargs.get("max_results", 5)), 100))
        params: dict[str, Any] = {
            "api_key": self._api_key,
            "engine": "google",
            "q": query,
            "num": max_results,
            "google_domain": kwargs.get("google_domain", "google.com"),
            "gl": kwargs.get("gl", "us"),
            "hl": kwargs.get("hl", "en"),
        }
        search_date = kwargs.get("search_date")
        if search_date:
            params["tbs"] = f"cdr:1,cd_min:{search_date},cd_max:{search_date}"

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(self._base_url, params=params)
        _raise_api_error(response, "SerpAPI")
        data = response.json()
        if data.get("error"):
            raise RuntimeError(f"SerpAPI error: {data['error']}")

        results = [
            {
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "content": item.get("snippet", ""),
                "score": 1.0 / max(1, int(item.get("position", 1))),
            }
            for item in data.get("organic_results", [])[:max_results]
        ]
        return {
            "query": query,
            "answer": data.get("answer_box", {}).get("answer", "")
            if isinstance(data.get("answer_box"), dict)
            else "",
            "results": results,
            "source": "serpapi",
        }

    async def health_check(self) -> bool:
        return bool(self._api_key)


class GoogleKnowledgeGraphConnector(BaseConnector):
    name = "google_knowledge_graph"
    actions = frozenset({"search"})

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._api_key = _config_value(self._config, "api_key", "GOOGLE_KG_API_KEY")
        self._base_url = self._config.get(
            "base_url", "https://kgsearch.googleapis.com/v1/entities:search"
        )

    async def fetch(self, action: str, **kwargs: Any) -> Any:
        if action != "search":
            raise ValueError(f"GoogleKnowledgeGraph: unsupported action '{action}'")
        if not self._api_key:
            raise RuntimeError("Google Knowledge Graph API key is not configured")
        query = str(kwargs.get("query", "")).strip()
        limit = max(1, min(int(kwargs.get("limit", 1)), 20))
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                self._base_url,
                params={"query": query, "limit": limit, "indent": True, "key": self._api_key},
            )
        _raise_api_error(response, "Google Knowledge Graph")
        data = response.json()
        results = []
        for item in data.get("itemListElement", [])[:limit]:
            entity = item.get("result", {})
            description = entity.get("detailedDescription", {})
            entity_types = entity.get("@type", ["Unknown"])
            results.append(
                {
                    "name": entity.get("name", "N/A"),
                    "type": entity_types[0] if entity_types else "Unknown",
                    "description": description.get("articleBody", "No description"),
                    "url": description.get("url", "#"),
                    "score": item.get("resultScore", 0),
                    "image": entity.get("image", {}).get("contentUrl", ""),
                }
            )
        return results

    async def health_check(self) -> bool:
        return bool(self._api_key)


class WikipediaConnector(BaseConnector):
    name = "wikipedia"
    actions = frozenset({"search"})

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._base_url = self._config.get(
            "base_url", "https://api.wikimedia.org/core/v1/wikipedia/en"
        )
        self._api_base_url = self._config.get("api_base_url", "https://en.wikipedia.org/w/api.php")
        self._user_agent = _config_value(
            self._config, "app_name", "WIKIPEDIA_USER_AGENT", DEFAULT_USER_AGENT
        )

    def _headers(self) -> dict[str, str]:
        return {"User-Agent": self._user_agent} if self._user_agent else {}

    async def _page_content(self, client: httpx.AsyncClient, title: str) -> str:
        response = await client.get(
            self._api_base_url,
            params={
                "action": "query",
                "prop": "extracts",
                "titles": title,
                "explaintext": True,
                "exsectionformat": "plain",
                "format": "json",
            },
            headers=self._headers(),
        )
        _raise_api_error(response, "Wikipedia")
        pages = response.json().get("query", {}).get("pages", {})
        page = next(iter(pages.values()), {})
        return re.sub(r"\n=+\s*(References|Bibliography|External links|Sources|Notes)\s*=+.*", "", page.get("extract", ""), flags=re.I | re.S).strip()

    async def fetch(self, action: str, **kwargs: Any) -> Any:
        if action != "search":
            raise ValueError(f"Wikipedia: unsupported action '{action}'")
        query = str(kwargs.get("query", "")).strip()
        limit = max(1, min(int(kwargs.get("limit", 3)), 20))
        headers = self._headers()
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                f"{self._base_url}/search/page",
                params={"q": query, "limit": min(20, limit)},
                headers=headers,
            )
            _raise_api_error(response, "Wikipedia")
            pages = response.json().get("pages", [])[:limit]
            results = []
            for page in pages:
                title = page.get("title", "")
                results.append(
                    {
                        "title": title,
                        "excerpt": (await self._page_content(client, title))[:8000],
                        "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                        "description": page.get("description", ""),
                    }
                )
        return {"query": query, "results": results, "source": "wikipedia"}

    async def health_check(self) -> bool:
        return bool(self._base_url)


def _allowed_url(url: str) -> bool:
    try:
        parsed = urlsplit(url.strip())
        host = (parsed.hostname or "").rstrip(".").lower()
        if parsed.scheme.lower() not in {"http", "https"} or not host:
            return False
        if host == "localhost" or host.endswith(".localhost") or host.endswith(".local"):
            return False
        try:
            ip = ipaddress.ip_address(host.split("%", 1)[0])
        except ValueError:
            return True
        return bool(ip.is_global)
    except ValueError:
        return False


class _PageParser(HTMLParser):
    _TEXT_TAGS = {"p", "li", "td", "th"}
    _HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}

    def __init__(self) -> None:
        super().__init__()
        self.title = ""
        self._in_title = False
        self._tag: str | None = None
        self._buffer: list[str] = []
        self._section = {"heading": "", "content": []}
        self.sections: list[dict[str, Any]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "title":
            self._in_title = True
        if tag in self._TEXT_TAGS or tag in self._HEADING_TAGS:
            self._tag = tag
            self._buffer = []

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        if self._tag:
            self._buffer.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        if tag != self._tag:
            return
        text = " ".join("".join(self._buffer).split())
        if text and tag in self._HEADING_TAGS:
            if self._section["content"]:
                self.sections.append(self._section)
            self._section = {"heading": text, "content": []}
        elif text:
            self._section["content"].append(text)
        self._tag = None
        self._buffer = []

    def result(self) -> dict[str, Any]:
        if self._section["content"]:
            self.sections.append(self._section)
        return {
            "title": " ".join(self.title.split()),
            "sections": self.sections,
            "word_count": sum(len(text.split()) for section in self.sections for text in section["content"]),
        }


class WebScraperConnector(BaseConnector):
    name = "web_scraper"
    actions = frozenset({"scrape_urls"})

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._user_agent = _config_value(
            self._config, "user_agent", "WEB_SCRAPER_USER_AGENT", DEFAULT_USER_AGENT
        )

    async def _scrape(self, client: httpx.AsyncClient, url: str) -> dict[str, Any]:
        if not _allowed_url(url):
            return {"url": url, "error": "target URL is not allowed"}
        try:
            response = await client.get(url, follow_redirects=True)
            if not _allowed_url(str(response.url)):
                return {"url": url, "error": "redirect target URL is not allowed"}
            response.raise_for_status()
            parser = _PageParser()
            parser.feed(response.text)
            return {"url": url, **parser.result()}
        except Exception as exc:
            return {"url": url, "error": str(exc)}

    async def fetch(self, action: str, **kwargs: Any) -> Any:
        if action != "scrape_urls":
            raise ValueError(f"WebScraper: unsupported action '{action}'")
        urls = [str(url) for url in kwargs.get("urls", [])]
        headers = {"User-Agent": self._user_agent} if self._user_agent else {}
        async with httpx.AsyncClient(timeout=30, headers=headers) as client:
            results = [await self._scrape(client, url) for url in urls]
        return {"results": results, "source": "web_scraper"}

    async def health_check(self) -> bool:
        return True


class SecAPIConnector(BaseConnector):
    name = "sec_api"
    actions = frozenset({"download", "xbrl_to_json_url", "xbrl_to_json_file"})

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._api_key = _config_value(self._config, "api_key", "SEC_API_KEY")
        self._base_url = self._config.get("base_url", "https://api.sec-api.io")

    async def fetch(self, action: str, **kwargs: Any) -> Any:
        if action not in self.actions:
            raise ValueError(f"SEC API: unsupported action '{action}'")
        if not self._api_key:
            raise RuntimeError("SEC API key is not configured")
        if action == "download":
            return await self._download(**kwargs)
        headers = {"Authorization": self._api_key}
        endpoint = f"{self._base_url}/xbrl-to-json"
        async with httpx.AsyncClient(timeout=120) as client:
            if action == "xbrl_to_json_url":
                params = {"token": self._api_key}
                for key in ("htm-url", "xbrl-url", "accession-no"):
                    if kwargs.get(key.replace("-", "_")):
                        params[key] = kwargs[key.replace("-", "_")]
                response = await client.get(endpoint, params=params, headers=headers)
            else:
                input_path = str(kwargs["input_path"])
                with open(input_path, "rb") as handle:
                    response = await client.post(
                        endpoint,
                        params={"token": self._api_key},
                        headers=headers,
                        files={"file": (os.path.basename(input_path), handle, "text/html")},
                    )
        _raise_api_error(response, "SEC API")
        return decode_json_response(response, source=self.name)

    @staticmethod
    def _filing_query(company_name: str, start_date: str, end_date: str, form_types: list[str]) -> str:
        identifier = company_name.strip()
        if not identifier:
            raise ValueError("SEC API download requires company_name")
        if re.fullmatch(r"\d{1,10}", identifier):
            company_clause = f"cik:{int(identifier)}"
        elif re.fullmatch(r"[A-Za-z][A-Za-z0-9.-]{0,9}", identifier):
            company_clause = f"ticker:{identifier.upper()}"
        else:
            escaped_name = identifier.replace('"', r'\"')
            company_clause = f'companyName:"{escaped_name}"'

        forms = [str(form).strip().upper() for form in form_types if str(form).strip()]
        if not forms:
            forms = ["10-K", "10-Q"]
        form_clause = " OR ".join(f'formType:"{form}"' for form in forms)
        return (
            f"({company_clause}) AND filedAt:[{start_date} TO {end_date}] "
            f"AND ({form_clause})"
        )

    @staticmethod
    def _archive_url(filing_url: str) -> str:
        parsed = urlsplit(filing_url)
        marker = "/Archives/edgar/data/"
        if (
            parsed.scheme != "https"
            or parsed.hostname != "www.sec.gov"
            or not parsed.path.startswith(marker)
        ):
            raise ValueError("SEC API filing URL must be an SEC EDGAR archive URL")
        relative_path = parsed.path[len(marker):]
        if not relative_path or any(part in {"", ".", ".."} for part in relative_path.split("/")):
            raise ValueError("SEC API filing URL has an invalid EDGAR archive path")
        return f"https://archive.sec-api.io/{relative_path}"

    async def _download(
        self,
        company_name: str,
        start_date: str,
        end_date: str,
        form_types: list[str] | None = None,
        limit: int = 10,
        output_dir: str | None = None,
        **_: Any,
    ) -> list[dict[str, Any]]:
        requested_forms = form_types or ["10-K", "10-Q"]
        request_limit = max(1, min(int(limit), 50))
        headers = {"Authorization": self._api_key}
        query_payload = {
            "query": self._filing_query(
                company_name, start_date, end_date, requested_forms,
            ),
            "from": "0",
            "size": str(request_limit),
            "sort": [{"filedAt": {"order": "asc"}}],
        }
        target_dir = Path(output_dir or f"data/sec_filings/{company_name}")
        target_dir.mkdir(parents=True, exist_ok=True)

        async with httpx.AsyncClient(timeout=120) as client:
            query_response = await client.post(
                self._base_url,
                json=query_payload,
                headers=headers,
            )
            _raise_api_error(query_response, "SEC API")
            filings = decode_json_response(query_response, source=self.name).get("filings", [])
            if not isinstance(filings, list):
                return []

            results: list[dict[str, Any]] = []
            for filing in filings[:request_limit]:
                if not isinstance(filing, dict):
                    continue
                filing_url = str(filing.get("linkToFilingDetails", "")).strip()
                accession = str(filing.get("accessionNo", "")).strip()
                filename = Path(urlsplit(filing_url).path).name
                if not filing_url or not accession or not filename:
                    continue
                archive_url = self._archive_url(filing_url)
                response = await client.get(archive_url, headers=headers)
                _raise_api_error(response, "SEC API")

                filed_date = str(filing.get("filedAt", ""))[:10]
                path = target_dir / f"{filed_date}_{accession.replace('/', '_')}_{filename}"
                if not path.exists():
                    path.write_bytes(response.content)
                results.append(
                    {
                        "company_name": company_name,
                        "cik": str(filing.get("cik", "")),
                        "accession_number": accession,
                        "form_type": str(filing.get("formType", "")),
                        "filed_date": filed_date,
                        "url": filing_url,
                        "path": str(path),
                        "source": "sec_api",
                    }
                )
            return results

    async def health_check(self) -> bool:
        return bool(self._api_key)


register_connector("serpapi", SerpAPIConnector)
register_connector("google_knowledge_graph", GoogleKnowledgeGraphConnector)
register_connector("wikipedia", WikipediaConnector)
register_connector("web_scraper", WebScraperConnector)
register_connector("sec_api", SecAPIConnector)
