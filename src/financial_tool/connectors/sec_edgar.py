"""SEC EDGAR connector — US regulatory filings search and retrieval."""

from __future__ import annotations

import os
import re
import time
from html import unescape
from pathlib import Path
from typing import Any

import httpx

from .base import BaseConnector, decode_json_response, register_connector
from ._numeric import safe_float
from ..network_config import sec_user_agent

_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
_FILING_URL = "https://www.sec.gov/Archives/edgar/data"
_CIK_CACHE: dict[str, str] = {}
_COMPANYFACTS_CACHE: dict[str, dict[str, Any]] = {}
_SUBMISSIONS_CACHE: dict[str, dict[str, Any]] = {}
_CONTENT_CACHE: dict[str, str] = {}


def extract_sec_archive_links_from_html(html: str) -> list[str]:
    links: list[str] = []
    for match in re.finditer(r"href=[\"'](?P<href>/Archives/edgar/data/[^\"']+)[\"']", str(html or "")):
        url = f"https://www.sec.gov{match.group('href')}"
        if url not in links:
            links.append(url)
    return links


_FILING_FRAGMENT_KEYWORDS = (
    "management discussion",
    "liquidity",
    "cash flow",
    "debt",
    "maturit",
    "segment",
    "revenue",
    "outstanding share",
    "contractual obligation",
)


def _html_to_plain_text(value: str) -> str:
    text = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", str(value or ""))
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</(h[1-6]|p|div|li|tr|table|section)>", "\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = unescape(text)
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def _filing_evidence_items(
    filing_id: str,
    text: str,
    *,
    max_items: int = 12,
    max_chars: int = 1800,
) -> list[dict[str, str]]:
    plain = _html_to_plain_text(text)
    blocks = [block for block in plain.splitlines() if block.strip()]
    items: list[dict[str, str]] = []
    seen: set[str] = set()

    def add_item(kind: str, snippet: str) -> None:
        normalized = " ".join(str(snippet or "").split())
        if len(normalized) < 20:
            return
        normalized = normalized[:max_chars]
        key = normalized.lower()
        if key in seen or len(items) >= max_items:
            return
        seen.add(key)
        fragment_id = f"{filing_id}:fragment:{len(items)}"
        items.append(
            {
                "source_id": filing_id,
                "fragment_id": fragment_id,
                "kind": kind,
                "text": normalized,
            }
        )

    for table_html in re.findall(r"(?is)<table\b.*?</table>", str(text or ""))[:max_items]:
        add_item("table", _html_to_plain_text(table_html))

    for index, block in enumerate(blocks):
        lower = block.lower()
        if not any(keyword in lower for keyword in _FILING_FRAGMENT_KEYWORDS):
            continue
        window = " ".join(blocks[max(0, index - 1) : min(len(blocks), index + 4)])
        add_item("section", window)

    if not items:
        add_item("content", plain)
    return items


_FIELD_ALIASES = {
    "assets": "total_assets",
    "total assets": "total_assets",
    "total_assets": "total_assets",
    "current assets": "current_assets",
    "current_assets": "current_assets",
    "liabilities": "total_liabilities",
    "total liabilities": "total_liabilities",
    "total_liabilities": "total_liabilities",
    "current liabilities": "current_liabilities",
    "current_liabilities": "current_liabilities",
    "stockholders equity": "stockholders_equity",
    "stockholders' equity": "stockholders_equity",
    "shareholders equity": "stockholders_equity",
    "shareholders' equity": "stockholders_equity",
    "stockholders_equity": "stockholders_equity",
    "revenue": "revenue",
    "revenues": "revenue",
    "sales": "revenue",
    "operating income": "operating_income",
    "operating_income": "operating_income",
    "net income": "net_income",
    "net_income": "net_income",
    "operating cash flow": "operating_cash_flow",
    "operating_cash_flow": "operating_cash_flow",
    "cash flow from operations": "operating_cash_flow",
    "capital expenditures": "capital_expenditures",
    "capital_expenditures": "capital_expenditures",
    "capex": "capital_expenditures",
    "free cash flow": "free_cash_flow",
    "free_cash_flow": "free_cash_flow",
    "research and development": "research_and_development_expense",
    "r&d": "research_and_development_expense",
    "r&d expense": "research_and_development_expense",
    "retained earnings": "retained_earnings",
    "long term debt": "long_term_debt",
    "long-term debt": "long_term_debt",
    "cash and cash equivalents": "cash_and_cash_equivalents",
    "cash equivalents": "cash_and_cash_equivalents",
    "short term investments": "short_term_investments",
    "short-term investments": "short_term_investments",
    "interest expense": "interest_expense",
    "provision for income taxes": "income_tax_expense",
    "income tax expense": "income_tax_expense",
    "earnings per share": "earnings_per_share",
    "eps": "earnings_per_share",
}

_STATEMENT_FIELD_TAGS = {
    "revenue": (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ),
    "cogs": ("CostOfRevenue", "CostOfGoodsAndServicesSold"),
    "gross_profit": ("GrossProfit",),
    "operating_income": ("OperatingIncomeLoss",),
    "net_income": ("NetIncomeLoss", "ProfitLoss"),
    "total_assets": ("Assets",),
    "current_assets": ("AssetsCurrent",),
    "total_liabilities": ("Liabilities",),
    "current_liabilities": ("LiabilitiesCurrent",),
    "stockholders_equity": (
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ),
    "operating_cash_flow": (
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ),
    "investing_cash_flow": ("NetCashProvidedByUsedInInvestingActivities",),
    "financing_cash_flow": ("NetCashProvidedByUsedInFinancingActivities",),
    "capital_expenditures": (
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "CapitalExpenditures",
    ),
    "free_cash_flow": ("FreeCashFlow",),
    "research_and_development_expense": (
        "ResearchAndDevelopmentExpense",
        "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
    ),
    "retained_earnings": (
        "RetainedEarningsAccumulatedDeficit",
        "RetainedEarningsAppropriated",
        "RetainedEarningsUnappropriated",
    ),
    "long_term_debt": (
        "LongTermDebt",
        "LongTermDebtAndFinanceLeaseObligations",
        "LongTermDebtNoncurrent",
    ),
    "cash_and_cash_equivalents": (
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ),
    "short_term_investments": (
        "ShortTermInvestments",
        "MarketableSecuritiesCurrent",
    ),
    "interest_expense": (
        "InterestExpenseNonOperating",
        "InterestExpense",
    ),
    "income_tax_expense": (
        "IncomeTaxExpenseBenefit",
        "CurrentIncomeTaxExpenseBenefit",
    ),
    "earnings_per_share": (
        "EarningsPerShareBasic",
        "EarningsPerShareDiluted",
    ),
}
_COMPOSITE_FIELDS = {
    "cash_and_short_term_investments": (
        "cash_and_cash_equivalents",
        "short_term_investments",
    ),
}

_SECTION_DEFAULT_FIELDS = {
    "income": ("revenue", "cogs", "gross_profit", "operating_income", "net_income"),
    "balance": ("total_assets", "total_liabilities", "stockholders_equity"),
    "cashflow": ("operating_cash_flow", "investing_cash_flow", "financing_cash_flow"),
}


def _canonical_field(field: Any) -> str:
    text = " ".join(str(field or "").strip().lower().replace("-", " ").replace("_", " ").split())
    return _FIELD_ALIASES.get(text, text.replace(" ", "_"))


def _symbol_cache_key(symbol: Any) -> str:
    return str(symbol or "").strip().upper()


def _statement_for_fields(fields: list[str], requested_statement: str | None) -> str:
    statement = str(requested_statement or "").strip().lower()
    if statement in {"income", "balance", "cashflow"}:
        return statement
    balance = {"total_assets", "current_assets", "total_liabilities", "current_liabilities", "stockholders_equity"}
    cashflow = {"operating_cash_flow", "investing_cash_flow", "financing_cash_flow", "capital_expenditures", "free_cash_flow"}
    field_set = set(fields)
    if field_set and field_set <= balance:
        return "balance"
    if field_set and field_set <= cashflow:
        return "cashflow"
    return "income"


def _field_units(fact: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    units = fact.get("units") if isinstance(fact, dict) else {}
    if not isinstance(units, dict):
        return "", []
    for unit_name in ("USD", "USD/shares", "shares"):
        rows = units.get(unit_name)
        if isinstance(rows, list) and rows:
            return unit_name, rows
    for unit_name, rows in units.items():
        if isinstance(rows, list) and rows:
            return str(unit_name), rows
    return "", []


def _period_key(entry: dict[str, Any]) -> tuple[str, int | None, str, str, str]:
    fy_value = entry.get("fy")
    try:
        fiscal_year = int(fy_value) if fy_value not in (None, "") else None
    except (TypeError, ValueError):
        fiscal_year = None
    return (
        str(entry.get("end") or ""),
        fiscal_year,
        str(entry.get("fp") or ""),
        str(entry.get("form") or ""),
        str(entry.get("filed") or ""),
    )


def _period_from_key(key: tuple[str, int | None, str, str, str]) -> dict[str, Any]:
    period_end, fiscal_year, fiscal_period, form, filed = key
    return {
        "period_end": period_end,
        "fiscal_year": fiscal_year,
        "fiscal_period": fiscal_period,
        "form": form,
        "filed": filed,
    }


def _series_rows_for_periods(
    field_candidates: dict[str, list[tuple[tuple[str, int | None, str, str, str], str, str, dict[str, Any]]]],
    requested_fields: list[str],
    period_keys: list[tuple[str, int | None, str, str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in period_keys:
        period = _period_from_key(key)
        for field in requested_fields:
            for candidate_key, tag, unit, entry in field_candidates.get(field, []):
                if candidate_key != key:
                    continue
                rows.append(
                    {
                        "field": field,
                        "value": safe_float(entry.get("val")),
                        "period_label": period.get("fiscal_period"),
                        "period_end": period.get("period_end"),
                        "fiscal_year": period.get("fiscal_year"),
                        "fiscal_period": period.get("fiscal_period"),
                        "form": period.get("form"),
                        "filed": period.get("filed"),
                        "unit": unit,
                        "tag": tag,
                    }
                )
                break
    return rows


class SECEdgarConnector(BaseConnector):
    name = "sec_edgar"
    actions = frozenset({"search", "content", "sec_filings", "download", "income", "balance", "cashflow", "statement_fields"})

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._last_request_time = 0.0
        user_agent_env = self._config.get("user_agent_env", "SEC_USER_AGENT")
        configured_user_agent = str(self._config.get("user_agent", "") or "").strip()
        if not configured_user_agent:
            configured_user_agent = os.getenv(user_agent_env, "").strip()
        if not configured_user_agent:
            configured_user_agent = sec_user_agent()
        if not configured_user_agent:
            raise ValueError("SEC requires network.user_agent in config.yaml")
        self._headers = {
            "User-Agent": configured_user_agent,
            "Accept": "application/json",
        }

    async def _rate_limit(self) -> None:
        """SEC asks for max 10 req/sec; enforce ~100ms gap."""
        elapsed = time.time() - self._last_request_time
        if elapsed < 0.12:
            import asyncio
            await asyncio.sleep(0.12 - elapsed)
        self._last_request_time = time.time()

    async def _get_json(self, url: str, params: dict | None = None) -> Any:
        await self._rate_limit()
        async with httpx.AsyncClient(timeout=30, headers=self._headers) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            return decode_json_response(resp, source=self.name)

    async def _get_text(self, url: str) -> str:
        await self._rate_limit()
        async with httpx.AsyncClient(timeout=30, headers=self._headers) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.text

    async def fetch(self, action: str, **kwargs: Any) -> Any:
        dispatch = {
            "search": self._search,
            "content": self._content,
            "sec_filings": self._sec_filings,
            "download": self._download,
            "income": self._income,
            "balance": self._balance,
            "cashflow": self._cashflow,
            "statement_fields": self._statement_fields,
        }
        handler = dispatch.get(action)
        if handler is None:
            raise ValueError(f"SEC EDGAR: unsupported action '{action}'")
        return await handler(**kwargs)

    # ── Search ───────────────────────────────────────────

    async def _search(
        self,
        query: str,
        form_type: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        **_: Any,
    ) -> list[dict]:
        params: dict[str, Any] = {"q": query, "dateRange": "custom"}
        if form_type:
            params["forms"] = form_type
        if date_from:
            params["startdt"] = date_from
        if date_to:
            params["enddt"] = date_to
        data = await self._get_json(
            "https://efts.sec.gov/LATEST/search-index", params=params,
        )
        hits = data.get("hits", {}).get("hits", [])
        results = []
        for hit in hits[:20]:
            src = hit.get("_source", {})
            hit_id = str(hit.get("_id", "")).strip()
            accession_from_id, separator, document = hit_id.partition(":")
            accession = str(src.get("adsh") or accession_from_id).strip()
            cik_values = src.get("ciks") or []
            cik = str(cik_values[0]).strip() if isinstance(cik_values, list) and cik_values else ""
            try:
                archive_cik = str(int(cik))
            except (TypeError, ValueError):
                archive_cik = cik
            accession_clean = accession.replace("-", "")
            filing_id = ""
            document_url = ""
            if archive_cik and accession_clean and separator and document:
                filing_id = f"{archive_cik}/{accession_clean}/{document}"
                document_url = f"{_FILING_URL}/{filing_id}"
            display_name = " ".join(str(item) for item in (src.get("display_names") or []))
            symbol_match = re.search(
                r"\(([A-Z][A-Z0-9.-]{0,9})\)\s+\(CIK\b", display_name
            )
            form_values = src.get("root_forms") or []
            results.append({
                "filing_id": filing_id or accession,
                "accession_number": accession,
                "document_url": document_url,
                "symbol": symbol_match.group(1) if symbol_match else "",
                "form_type": form_values[0] if form_values else src.get("form", ""),
                "filed_date": src.get("file_date", ""),
                "title": src.get("file_description", src.get("display_names", [""])[0] if src.get("display_names") else ""),
            })
        return results

    # ── Content ──────────────────────────────────────────

    async def _content(self, filing_id: str, **_: Any) -> dict:
        url = f"https://www.sec.gov/Archives/edgar/data/{filing_id}"
        try:
            cache_key = str(filing_id or "").strip()
            if cache_key in _CONTENT_CACHE:
                text = _CONTENT_CACHE[cache_key]
            else:
                text = await self._get_text(url)
                _CONTENT_CACHE[cache_key] = text
            payload = {"filing_id": filing_id, "form_type": "", "content": text[:50000], "filed_date": ""}
            evidence_items = _filing_evidence_items(filing_id, text)
            if evidence_items:
                payload["evidence_items"] = evidence_items
            return payload
        except Exception as exc:
            return {"filing_id": filing_id, "form_type": "", "content": f"Error: {exc}", "filed_date": ""}

    async def _sec_filings(
        self,
        symbol: str,
        filing_type: str = "",
        limit: int = 10,
        date_from: str | None = None,
        date_to: str | None = None,
        **_: Any,
    ) -> list[dict]:
        results = await self._search(
            query=symbol,
            form_type=filing_type or None,
            date_from=date_from,
            date_to=date_to,
        )
        return results[:limit]

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
        """Download primary SEC filing documents without the crawler's local side effects."""
        cik = await self._resolve_cik(company_name)
        if not cik:
            return []
        if cik in _SUBMISSIONS_CACHE:
            submissions = _SUBMISSIONS_CACHE[cik]
        else:
            submissions = await self._get_json(
                f"https://data.sec.gov/submissions/CIK{cik}.json"
            )
            if isinstance(submissions, dict):
                _SUBMISSIONS_CACHE[cik] = submissions
        recent = submissions.get("filings", {}).get("recent", {})
        wanted_forms = {str(form).upper() for form in (form_types or ["10-K", "10-Q"])}
        target_dir = Path(output_dir or f"data/sec_filings/{company_name}")
        target_dir.mkdir(parents=True, exist_ok=True)
        results: list[dict[str, Any]] = []
        for index in range(len(recent.get("accessionNumber", []))):
            form = str(recent.get("form", [""])[index]).upper()
            filed = str(recent.get("filingDate", [""])[index])
            if form not in wanted_forms or not (start_date <= filed <= end_date):
                continue
            accession = str(recent.get("accessionNumber", [""])[index])
            primary_doc = str(recent.get("primaryDocument", [""])[index])
            if not accession or not primary_doc:
                continue
            accession_clean = accession.replace("-", "")
            url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_clean}/{primary_doc}"
            safe_accession = accession.replace("/", "_")
            path = target_dir / f"{filed}_{safe_accession}_{primary_doc}"
            if not path.exists():
                await self._download_bytes(url, path)
            results.append(
                {
                    "company_name": company_name,
                    "cik": cik,
                    "accession_number": accession,
                    "form_type": form,
                    "filed_date": filed,
                    "url": url,
                    "path": str(path),
                }
            )
            if len(results) >= max(1, min(limit, 50)):
                break
        return results

    async def _download_bytes(self, url: str, path: Path) -> None:
        await self._rate_limit()
        async with httpx.AsyncClient(timeout=60, headers=self._headers) as client:
            response = await client.get(url)
            response.raise_for_status()
        path.write_bytes(response.content)

    # ── Financials (XBRL) ────────────────────────────────

    async def _resolve_cik(self, symbol: str) -> str:
        cache_key = _symbol_cache_key(symbol)
        if cache_key in _CIK_CACHE:
            return _CIK_CACHE[cache_key]
        tickers = await self._get_json(
            "https://www.sec.gov/files/company_tickers.json"
        )
        for record in tickers.values():
            if (
                isinstance(record, dict)
                and str(record.get("ticker", "")).upper() == symbol.upper()
            ):
                cik = str(record.get("cik_str", "")).zfill(10)
                _CIK_CACHE[cache_key] = cik
                return cik
            if (
                isinstance(record, dict)
                and str(record.get("title", "")).casefold() == symbol.casefold()
            ):
                cik = str(record.get("cik_str", "")).zfill(10)
                _CIK_CACHE[cache_key] = cik
                return cik
        legacy = await self._get_json(
            "https://www.sec.gov/cgi-bin/browse-edgar",
            params={
                "action": "getcompany",
                "company": symbol,
                "output": "json",
            },
        )
        if isinstance(legacy, dict) and legacy.get("CIK"):
            cik = str(legacy["CIK"]).zfill(10)
            _CIK_CACHE[cache_key] = cik
            return cik
        _CIK_CACHE[cache_key] = ""
        return ""

    async def _income(self, **kwargs: Any) -> dict:
        return await self._financials(statement="income", **kwargs)

    async def _balance(self, **kwargs: Any) -> dict:
        return await self._financials(statement="balance", **kwargs)

    async def _cashflow(self, **kwargs: Any) -> dict:
        return await self._financials(statement="cashflow", **kwargs)

    async def _financials(
        self,
        symbol: str,
        statement: str,
        period: str = "annual",
        start_date: str | None = None,
        end_date: str | None = None,
        **_: Any,
    ) -> dict:
        """Fetch one canonical financial statement via SEC XBRL company facts."""
        structured = await self._statement_fields(
            symbol=symbol,
            fields=list(_SECTION_DEFAULT_FIELDS.get(statement, ())),
            statement=statement,
            period=period,
            start_date=start_date,
            end_date=end_date,
        )
        values = structured.get("values") if isinstance(structured, dict) else {}
        if not isinstance(values, dict) or not values:
            return structured if isinstance(structured, dict) else {}
        period_info = {}
        periods = structured.get("periods") if isinstance(structured.get("periods"), list) else []
        if periods and isinstance(periods[0], dict):
            period_info = periods[0]
        result: dict[str, Any] = {
            "symbol": symbol,
            "date": str(period_info.get("period_end") or ""),
            "source": "sec_edgar",
        }
        result.update(values)
        if "stockholders_equity" in result and "total_equity" not in result:
            result["total_equity"] = result["stockholders_equity"]
        if "operating_cash_flow" in result and "operating_cf" not in result:
            result["operating_cf"] = result["operating_cash_flow"]
        if "investing_cash_flow" in result and "investing_cf" not in result:
            result["investing_cf"] = result["investing_cash_flow"]
        if "financing_cash_flow" in result and "financing_cf" not in result:
            result["financing_cf"] = result["financing_cash_flow"]
        return result

    async def _statement_fields(
        self,
        symbol: str,
        fields: list[str] | None = None,
        normalized_fields: list[str] | None = None,
        statement: str | None = None,
        period: str = "annual",
        fiscal_year: int | None = None,
        fiscal_period: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        requested_fields = [
            _canonical_field(field)
            for field in (normalized_fields or fields or [])
            if str(field or "").strip()
        ]
        requested_fields = list(dict.fromkeys(requested_fields))
        if not requested_fields:
            return {
                "status": "invalid_request",
                "symbol": symbol,
                "source": "sec_edgar",
                "values": {},
                "missing_fields": [],
                "reason": "fields must contain at least one statement field",
            }
        cik_padded = await self._resolve_cik(symbol)
        if not cik_padded:
            return {
                "status": "empty_result",
                "symbol": symbol,
                "source": "sec_edgar",
                "values": {},
                "missing_fields": requested_fields,
                "reason": "cik_not_found",
            }
        facts_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_padded}.json"
        if cik_padded in _COMPANYFACTS_CACHE:
            facts = _COMPANYFACTS_CACHE[cik_padded]
        else:
            facts = await self._get_json(facts_url)
            if isinstance(facts, dict):
                _COMPANYFACTS_CACHE[cik_padded] = facts
        gaap = facts.get("facts", {}).get("us-gaap", {}) if isinstance(facts, dict) else {}
        if not isinstance(gaap, dict) or not gaap:
            return {
                "status": "empty_result",
                "symbol": symbol,
                "source": "sec_edgar",
                "values": {},
                "missing_fields": requested_fields,
                "reason": "companyfacts_us_gaap_empty",
            }

        normalized_period = str(period or "annual").strip().lower()
        expected_form = "10-Q" if normalized_period in {"quarter", "quarterly"} else "10-K"
        requested_fp = str(fiscal_period or "").strip().upper()
        statement_name = _statement_for_fields(requested_fields, statement)
        field_candidates: dict[str, list[tuple[tuple[str, int | None, str, str, str], str, str, dict[str, Any]]]] = {}
        candidate_cache: dict[str, list[tuple[tuple[str, int | None, str, str, str], str, str, dict[str, Any]]]] = {}

        def simple_candidates_for_field(
            field: str,
        ) -> list[tuple[tuple[str, int | None, str, str, str], str, str, dict[str, Any]]]:
            candidates: list[tuple[tuple[str, int | None, str, str, str], str, str, dict[str, Any]]] = []
            for tag in _STATEMENT_FIELD_TAGS.get(field, ()):
                unit, entries = _field_units(gaap.get(tag, {}))
                for entry in entries:
                    if not isinstance(entry, dict) or not entry.get("end"):
                        continue
                    form = str(entry.get("form") or "")
                    if form and form != expected_form:
                        continue
                    fp = str(entry.get("fp") or "").upper()
                    if requested_fp and fp != requested_fp:
                        continue
                    if normalized_period in {"annual", "year", "fy"} and fp and fp != "FY":
                        continue
                    if normalized_period in {"quarter", "quarterly"} and fp and not fp.startswith("Q"):
                        continue
                    try:
                        entry_fy = int(entry.get("fy")) if entry.get("fy") not in (None, "") else None
                    except (TypeError, ValueError):
                        entry_fy = None
                    if fiscal_year is not None and entry_fy != int(fiscal_year):
                        continue
                    end = str(entry.get("end") or "")
                    if start_date and end < start_date:
                        continue
                    if end_date and end > end_date:
                        continue
                    candidates.append((_period_key(entry), tag, unit, entry))
            candidates.sort(
                key=lambda item: (
                    item[0][0],
                    item[0][4],
                    _STATEMENT_FIELD_TAGS.get(item[1], (item[1],)).index(item[1])
                    if item[1] in _STATEMENT_FIELD_TAGS.get(field, ())
                    else 999,
                ),
                reverse=True,
            )
            return candidates

        def candidates_for_field(
            field: str,
        ) -> list[tuple[tuple[str, int | None, str, str, str], str, str, dict[str, Any]]]:
            if field in candidate_cache:
                return candidate_cache[field]
            components = _COMPOSITE_FIELDS.get(field)
            if not components:
                candidate_cache[field] = simple_candidates_for_field(field)
                return candidate_cache[field]
            component_candidates = {
                component: candidates_for_field(component)
                for component in components
            }
            component_by_key: dict[str, dict[tuple[str, int | None, str, str, str], tuple[str, str, dict[str, Any]]]] = {}
            for component, candidates in component_candidates.items():
                by_key: dict[tuple[str, int | None, str, str, str], tuple[str, str, dict[str, Any]]] = {}
                for key, tag, unit, entry in candidates:
                    by_key.setdefault(key, (tag, unit, entry))
                component_by_key[component] = by_key
            common_keys = set.intersection(
                *[set(by_key) for by_key in component_by_key.values()]
            ) if component_by_key else set()
            candidates: list[tuple[tuple[str, int | None, str, str, str], str, str, dict[str, Any]]] = []
            for key in common_keys:
                parts = [component_by_key[component][key] for component in components]
                units = [unit for _tag, unit, _entry in parts if unit]
                unit = units[0] if units and all(item == units[0] for item in units) else ""
                tags = [tag for tag, _unit, _entry in parts]
                value = sum(safe_float(entry.get("val")) for _tag, _unit, entry in parts)
                combined_entry = dict(parts[0][2])
                combined_entry["val"] = value
                candidates.append((key, "+".join(tags), unit, combined_entry))
            candidates.sort(key=lambda item: (item[0][0], item[0][4]), reverse=True)
            candidate_cache[field] = candidates
            return candidate_cache[field]

        for field in requested_fields:
            field_candidates[field] = candidates_for_field(field)

        period_scores: dict[tuple[str, int | None, str, str, str], int] = {}
        for candidates in field_candidates.values():
            seen_for_field: set[tuple[str, int | None, str, str, str]] = set()
            for key, _tag, _unit, _entry in candidates:
                if key in seen_for_field:
                    continue
                period_scores[key] = period_scores.get(key, 0) + 1
                seen_for_field.add(key)
        if not period_scores:
            return {
                "status": "field_missing",
                "symbol": symbol,
                "source": "sec_edgar",
                "statement": statement_name,
                "values": {},
                "missing_fields": requested_fields,
                "reason": "requested_tags_not_found_for_period",
            }
        all_period_keys = sorted(period_scores, key=lambda key: (key[0], key[4], key[2]))
        best_period = max(
            period_scores,
            key=lambda key: (period_scores[key], key[0], key[4]),
        )
        values: dict[str, Any] = {}
        tag_provenance: dict[str, str] = {}
        units: dict[str, str] = {}
        for field, candidates in field_candidates.items():
            for key, tag, unit, entry in candidates:
                if key != best_period:
                    continue
                values[field] = safe_float(entry.get("val"))
                tag_provenance[field] = tag
                if unit:
                    units[field] = unit
                break
        missing_fields = [field for field in requested_fields if field not in values]
        range_requested = bool(start_date or end_date) and fiscal_year is None and not requested_fp
        selected_fields = _series_rows_for_periods(
            field_candidates,
            requested_fields,
            [best_period],
        )
        series = (
            _series_rows_for_periods(field_candidates, requested_fields, all_period_keys)
            if range_requested
            else []
        )
        metadata_periods = all_period_keys if range_requested else [best_period]
        period_ends = [str(key[0]) for key in metadata_periods if key[0]]
        filed_dates = [str(key[4]) for key in metadata_periods if key[4]]
        labels = [str(key[2]) for key in metadata_periods if key[2]]
        period_field_counts: dict[tuple[str, int | None, str, str, str], int] = {}
        for row in series:
            row_key = (
                str(row.get("period_end") or ""),
                row.get("fiscal_year") if isinstance(row.get("fiscal_year"), int) else None,
                str(row.get("fiscal_period") or ""),
                str(row.get("form") or ""),
                str(row.get("filed") or ""),
            )
            period_field_counts[row_key] = period_field_counts.get(row_key, 0) + 1
        range_has_partial_period = bool(series) and any(
            count < len(requested_fields) for count in period_field_counts.values()
        )
        coverage_completeness = (
            "known"
            if not missing_fields and not range_has_partial_period
            else "partial"
        )
        result = {
            "status": "success" if not missing_fields else ("partial" if values else "field_missing"),
            "symbol": symbol,
            "source": "sec_edgar",
            "statement": statement_name,
            "period": normalized_period,
            "values": values,
            "missing_fields": missing_fields,
            "periods": [_period_from_key(best_period)],
            "fields": selected_fields,
            "tag_provenance": tag_provenance,
            "units": units,
            "temporal_metadata": {
                "requested_start": start_date,
                "requested_end": end_date,
                "actual_min_date": min(period_ends) if period_ends else None,
                "actual_max_date": max(period_ends) if period_ends else None,
                "date_dimension": "fiscal_period",
                "fiscal_period_labels": labels,
                "period_ends": period_ends,
                "published_at": filed_dates,
                "filed_at": filed_dates,
                "temporal_safe": bool(period_ends),
                "date_verification": "verified" if period_ends else "unverified",
                "coverage_completeness": coverage_completeness,
            },
        }
        if series:
            result["series"] = series
        return result

    async def health_check(self) -> bool:
        try:
            await self._get_json("https://efts.sec.gov/LATEST/search-index", {"q": "AAPL"})
            return True
        except Exception:
            return False


register_connector("sec_edgar", SECEdgarConnector)
