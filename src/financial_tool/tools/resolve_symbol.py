"""Symbol resolution tool — maps company names/aliases to ticker symbols.

Priority chain:
1. Static alias table (``catalogs/company_aliases.json``) — instant, no network.
2. FMP ``/search-name`` API — US + HK companies.
3. akshare A-share name search — CN companies via ``stock_zh_a_spot_em``.

Use this tool FIRST when a benchmark query mentions a company by name
without an explicit ticker symbol, so downstream data tools receive the
correct symbol instead of a guess from the LLM's training data.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from ..connectors.base import fetch_with_failover
from finance_tool_universe.config import catalog_dir

logger = logging.getLogger("financial_tool.tools.resolve_symbol")

_CATALOG_DIR = catalog_dir()
_ALIAS_PATH = _CATALOG_DIR / "company_aliases.json"

# Module-level cache — load the alias table once per process.
_ALIAS_CACHE: dict[str, Any] | None = None
_GROUP_CACHE: dict[str, Any] | None = None

# Detect CJK characters to decide whether to try akshare A-share search.
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def _load_aliases() -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Load and cache the company alias table.

    Returns a tuple ``(aliases, groups)`` where ``aliases`` maps lowercased
    names to symbol records and ``groups`` maps group names (e.g.
    "magnificent seven") to lists of symbol records.
    """
    global _ALIAS_CACHE, _GROUP_CACHE
    if _ALIAS_CACHE is not None and _GROUP_CACHE is not None:
        return _ALIAS_CACHE, _GROUP_CACHE
    try:
        raw = json.loads(_ALIAS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not load company_aliases.json: %s", exc)
        _ALIAS_CACHE, _GROUP_CACHE = {}, {}
        return _ALIAS_CACHE, _GROUP_CACHE

    aliases: dict[str, dict[str, Any]] = {}
    groups: dict[str, Any] = {}
    for key, value in raw.items():
        if key.startswith("_"):
            if key == "_groups":
                groups = value if isinstance(value, dict) else {}
            continue
        if isinstance(value, dict) and "symbol" in value:
            aliases[key.lower()] = value
    _ALIAS_CACHE, _GROUP_CACHE = aliases, groups
    return _ALIAS_CACHE, _GROUP_CACHE


def _match_alias(query: str) -> list[dict[str, Any]]:
    """Return alias-table matches for *query* (exact + substring)."""
    aliases, groups = _load_aliases()
    matches: list[dict[str, Any]] = []
    seen_symbols: set[str] = set()

    normalized = query.strip().lower()

    # 1. Group aliases (e.g. "magnificent seven")
    for group_name, group_def in groups.items():
        if group_name in normalized or normalized in group_name:
            for sym in group_def.get("symbols", []):
                if sym["symbol"] not in seen_symbols:
                    matches.append({**sym, "source": "alias_table", "query_match": group_name})
                    seen_symbols.add(sym["symbol"])

    # 2. Exact alias match
    if normalized in aliases:
        rec = aliases[normalized]
        if rec["symbol"] not in seen_symbols:
            matches.append({**rec, "source": "alias_table", "query_match": "exact"})
            seen_symbols.add(rec["symbol"])

    # 3. Substring alias match (e.g. "ping an bank" matches "ping an")
    if not matches:
        for alias_key, rec in aliases.items():
            if not alias_key:
                continue
            # Match if the alias is a substring of the query, or vice versa.
            # Skip very short aliases (< 3 chars) to avoid false positives.
            if len(alias_key) < 3:
                continue
            if alias_key in normalized or normalized in alias_key:
                if rec["symbol"] not in seen_symbols:
                    matches.append({**rec, "source": "alias_table", "query_match": "substring"})
                    seen_symbols.add(rec["symbol"])

    return matches


async def _search_fmp(query: str, limit: int) -> list[dict[str, Any]]:
    """Search via FMP ``/search-name`` through the failover engine."""
    matches: list[dict[str, Any]] = []
    try:
        results = await fetch_with_failover(
            "search_results", "search", region="US", query=query, limit=limit,
        )
        for item in (results or [])[:limit]:
            if not isinstance(item, dict):
                continue
            payload = item.get("payload", {})
            symbol = payload.get("symbol", "") or item.get("symbol", "")
            name = payload.get("name", "") or item.get("name", "")
            if not symbol:
                continue
            matches.append({
                "symbol": symbol,
                "name": name,
                "exchange": "",
                "region": "US",
                "source": "fmp",
            })
    except Exception as exc:
        logger.debug("FMP symbol search failed for '%s': %s", query, exc)
    return matches


async def _search_akshare(query: str, limit: int) -> list[dict[str, Any]]:
    """Search A-share companies by name via akshare ``stock_zh_a_spot_em``."""
    matches: list[dict[str, Any]] = []
    try:
        import akshare as ak
        import pandas as pd
        df = ak.stock_zh_a_spot_em()
        if df is None or df.empty:
            return matches
        name_col = "\u540d\u79f0" if "\u540d\u79f0" in df.columns else None
        code_col = "\u4ee3\u7801" if "\u4ee3\u7801" in df.columns else None
        if not name_col or not code_col:
            return matches
        # Substring match on the company name column
        mask = df[name_col].str.contains(query, case=False, na=False)
        for _, row in df[mask].head(limit).iterrows():
            code = str(row[code_col])
            # Build the .SZ/.SH suffix from the code prefix
            if code.startswith(("6", "9")):
                suffix = ".SH"
            elif code.startswith(("0", "3")):
                suffix = ".SZ"
            elif code.startswith(("8", "4")):
                suffix = ".BJ"
            else:
                suffix = ".SZ"
            matches.append({
                "symbol": f"{code}{suffix}",
                "name": str(row[name_col]),
                "exchange": "SZSE" if suffix == ".SZ" else ("SSE" if suffix == ".SH" else "BSE"),
                "region": "CN",
                "source": "akshare",
            })
    except Exception as exc:
        logger.debug("akshare A-share name search failed for '%s': %s", query, exc)
    return matches


def register(mcp) -> None:

    @mcp.tool()
    async def resolve_symbol(
        query: str,
        region: str = "US",
        limit: int = 10,
    ) -> dict:
        """Resolve a company name or alias to stock symbol(s).

        Use this FIRST when the query mentions a company by name without an
        explicit ticker (e.g. "Ping An Bank", "Netflix", "Tencent"). Returns
        candidate symbols with exchange/region so downstream data tools
        receive the correct ticker.

        Priority chain:
        1. Static alias table (instant, covers US/A-share/HK + group aliases)
        2. FMP ``/search-name`` (US + international companies)
        3. akshare A-share name search (CN companies, when query has CJK chars)
        """
        normalized = (query or "").strip()
        if not normalized:
            return {"query": query, "matches": [], "status": "invalid_request"}

        matches: list[dict[str, Any]] = []
        seen_symbols: set[str] = set()

        def _add(items: list[dict[str, Any]]) -> None:
            for item in items:
                sym = item.get("symbol", "")
                if sym and sym not in seen_symbols:
                    matches.append(item)
                    seen_symbols.add(sym)

        # Priority 1: alias table
        _add(_match_alias(normalized))

        # Priority 2: FMP search (always try for US/HK companies)
        if len(matches) < limit:
            fmp_matches = await _search_fmp(normalized, limit - len(matches))
            _add(fmp_matches)

        # Priority 3: akshare A-share search (CJK queries or CN region)
        is_cjk = bool(_CJK_RE.search(normalized))
        if (len(matches) < limit) and (is_cjk or region.upper() == "CN"):
            akshare_matches = await _search_akshare(normalized, limit - len(matches))
            _add(akshare_matches)

        status = "success" if matches else "no_match"
        return {
            "query": query,
            "region": region,
            "matches": matches[:limit],
            "status": status,
            "sources_checked": ["alias_table", "fmp", "akshare"] if is_cjk or region.upper() == "CN"
                              else ["alias_table", "fmp"],
        }
