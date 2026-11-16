"""Fundamentals tools — income, balance sheet, cash flow, profile, ratios, enterprise value, growth, ESG, Piotroski."""

from __future__ import annotations

import math
import re
from typing import Any

from ..connectors.base import fetch_with_failover
from ..errors import DataValidationError
from ..symbols import classify_instrument
from ._external import call_external_tool
from ._region import require_region


def _fundamental_instrument(symbol: str, region: str) -> tuple[str, str]:
    """Normalize an equity symbol and infer its market only for AUTO calls."""
    instrument = classify_instrument(symbol)
    if instrument.kind == "macro_series":
        raise DataValidationError(
            "symbol",
            f"{symbol} is macro data; call macro_get_series("
            f"series_id={instrument.provider_symbol!r}, region='US')",
        )
    requested_region = str(region or "AUTO").strip().upper()
    if requested_region == "AUTO":
        return instrument.provider_symbol, instrument.region or "US"
    return instrument.provider_symbol, requested_region


_DATE_FIELDS = {"date", "period_end", "fiscal_period_end", "fiscal_year", "period", "period_label", "filed_date"}
_IDENTITY_FIELDS = {"symbol", "ticker", "status", "source", "provider", "currency", "unit", "reason", "period", "period_label"}


def _canonical_field_name(field: str) -> str:
    text = " ".join(str(field or "").strip().lower().replace("-", " ").replace("_", " ").split())
    aliases = {
        "assets": "total_assets",
        "total assets": "total_assets",
        "current assets": "current_assets",
        "liabilities": "total_liabilities",
        "total liabilities": "total_liabilities",
        "current liabilities": "current_liabilities",
        "stockholders equity": "stockholders_equity",
        "stockholders' equity": "stockholders_equity",
        "shareholders equity": "stockholders_equity",
        "shareholders' equity": "stockholders_equity",
        "revenue": "revenue",
        "revenues": "revenue",
        "sales": "revenue",
        "operating income": "operating_income",
        "net income": "net_income",
        "operating cash flow": "operating_cash_flow",
        "cash flow from operations": "operating_cash_flow",
        "capital expenditures": "capital_expenditures",
        "capex": "capital_expenditures",
        "free cash flow": "free_cash_flow",
    }
    return aliases.get(text, text.replace(" ", "_"))


def _normalize_statement_period_args(
    *,
    period: str,
    fiscal_period: str | None,
    fiscal_year: int | None,
) -> tuple[str, str | None, int | None]:
    normalized_period = str(period or "annual").strip().lower()
    fp_text = str(fiscal_period or "").strip().upper()
    quarter_match = re.fullmatch(r"(Q[1-4])\s+(20\d{2}|19\d{2})", fp_text)
    if quarter_match:
        return normalized_period, quarter_match.group(1), int(quarter_match.group(2))
    fiscal_year_match = re.fullmatch(r"FY\s*(20\d{2}|19\d{2})", fp_text)
    if fiscal_year_match:
        return "annual", "FY", int(fiscal_year_match.group(1))
    return normalized_period, fp_text or None, fiscal_year


def _normalize_statement_fields_result(result: Any, requested_fields: list[str]) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {
            "status": "schema_invalid",
            "values": {},
            "missing_fields": requested_fields,
            "reason": "statement_fields provider returned a non-object payload",
        }
    payload = dict(result)
    values = payload.get("values")
    payload["values"] = values if isinstance(values, dict) else {}
    missing = payload.get("missing_fields")
    payload["missing_fields"] = [str(item) for item in missing] if isinstance(missing, list) else []
    if not payload["values"] and not payload.get("status"):
        payload["status"] = "field_missing"
        payload["missing_fields"] = payload["missing_fields"] or requested_fields
        payload["reason"] = payload.get("reason") or "requested statement fields were not found"
    elif not payload.get("status"):
        payload["status"] = "partial" if payload["missing_fields"] else "success"
    return payload


def _is_date_value(value: Any) -> bool:
    text = str(value or "").strip().replace("/", "-")
    if (len(text) >= 4 and text[:4].isdigit()) or text.upper().startswith("FY20"):
        return True
    return False


def _numeric_value(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _is_undated_zero_mapping(payload: dict[str, Any]) -> bool:
    if any(_is_date_value(payload.get(field)) for field in _DATE_FIELDS):
        return False
    numbers = [
        number
        for key, value in payload.items()
        if str(key).lower() not in _IDENTITY_FIELDS
        for number in [_numeric_value(value)]
        if number is not None
    ]
    if not numbers or any(number != 0 for number in numbers):
        return False
    substantive_text = [
        str(value).strip()
        for key, value in payload.items()
        if str(key).lower() not in _IDENTITY_FIELDS
        and not _numeric_value(value) is not None
        and str(value).strip()
    ]
    return not substantive_text


def normalize_fundamental_result(result: Any) -> Any:
    """Reject undated all-zero statement placeholders as usable evidence."""

    if isinstance(result, dict):
        status = str(result.get("status") or "").lower()
        if status in {"source_unavailable", "error", "failed", "permission"}:
            return result
        rows = result.get("rows")
        if isinstance(rows, list) and rows and all(
            isinstance(row, dict) and _is_undated_zero_mapping(row) for row in rows
        ):
            return {
                "status": "empty_result",
                "temporal_safe": False,
                "reason": "undated_zero_placeholder",
            }
        if _is_undated_zero_mapping(result):
            return {
                "status": "empty_result",
                "temporal_safe": False,
                "reason": "undated_zero_placeholder",
            }
    return result


def register(mcp) -> None:

    @mcp.tool()
    async def fundamentals_get_income(
        symbol: str,
        period: str = "annual",
        region: str = "AUTO",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict:
        symbol, resolved_region = _fundamental_instrument(symbol, region)
        return normalize_fundamental_result(await fetch_with_failover(
            "financial_statement",
            "income",
            region=resolved_region,
            symbol=symbol,
            period=period,
            start_date=start_date,
            end_date=end_date,
        ))

    @mcp.tool()
    async def fundamentals_get_balance(
        symbol: str,
        period: str = "annual",
        region: str = "AUTO",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict:
        symbol, resolved_region = _fundamental_instrument(symbol, region)
        return normalize_fundamental_result(await fetch_with_failover(
            "financial_statement",
            "balance",
            region=resolved_region,
            symbol=symbol,
            period=period,
            start_date=start_date,
            end_date=end_date,
        ))

    @mcp.tool()
    async def fundamentals_get_cashflow(
        symbol: str,
        period: str = "annual",
        region: str = "AUTO",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict:
        symbol, resolved_region = _fundamental_instrument(symbol, region)
        return normalize_fundamental_result(await fetch_with_failover(
            "financial_statement",
            "cashflow",
            region=resolved_region,
            symbol=symbol,
            period=period,
            start_date=start_date,
            end_date=end_date,
        ))

    @mcp.tool()
    async def fundamentals_get_statement_fields(
        symbol: str,
        fields: list[str],
        statement: str | None = None,
        period: str = "annual",
        fiscal_year: int | None = None,
        fiscal_period: str | None = None,
        region: str = "AUTO",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict:
        normalized_fields = [
            _canonical_field_name(field)
            for field in fields or []
            if str(field or "").strip()
        ]
        normalized_fields = list(dict.fromkeys(normalized_fields))
        if not normalized_fields:
            return {
                "status": "invalid_request",
                "values": {},
                "missing_fields": [],
                "reason": "fields must contain at least one statement field",
            }
        symbol, resolved_region = _fundamental_instrument(symbol, region)
        period, fiscal_period, fiscal_year = _normalize_statement_period_args(
            period=period,
            fiscal_period=fiscal_period,
            fiscal_year=fiscal_year,
        )
        result = await fetch_with_failover(
            "financial_statement",
            "statement_fields",
            region=resolved_region,
            symbol=symbol,
            fields=fields,
            normalized_fields=normalized_fields,
            statement=statement,
            period=period,
            fiscal_year=fiscal_year,
            fiscal_period=fiscal_period,
            start_date=start_date,
            end_date=end_date,
        )
        return _normalize_statement_fields_result(result, normalized_fields)

    @mcp.tool()
    async def fundamentals_get_profile(
        symbol: str,
        region: str = "US",
        start_date: str | None = None,
        end_date: str | None = None,
        as_of_date: str | None = None,
    ) -> dict:
        return await fetch_with_failover(
            "company_profile",
            "profile",
            region=region,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            as_of_date=as_of_date,
        )

    @mcp.tool()
    async def fundamentals_get_ratios(
        symbol: str,
        period: str = "annual",
        limit: int = 10,
        region: str = "US",
        start_date: str | None = None,
        end_date: str | None = None,
        as_of_date: str | None = None,
    ) -> dict | list[dict]:
        if period == "ttm":
            return await fetch_with_failover(
                "financial_ratios",
                "ratios_ttm",
                region=region,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                as_of_date=as_of_date,
            )
        return await fetch_with_failover(
            "financial_ratios",
            "ratios",
            region=region,
            symbol=symbol,
            period=period,
            limit=limit,
            start_date=start_date,
            end_date=end_date,
            as_of_date=as_of_date,
        )

    @mcp.tool()
    async def fundamentals_get_enterprise_value(
        symbol: str,
        period: str = "annual",
        limit: int = 10,
        region: str = "US",
        start_date: str | None = None,
        end_date: str | None = None,
        as_of_date: str | None = None,
    ) -> list[dict]:
        return await fetch_with_failover(
            "enterprise_value",
            "enterprise_value",
            region=region,
            symbol=symbol,
            period=period,
            limit=limit,
            start_date=start_date,
            end_date=end_date,
            as_of_date=as_of_date,
        )

    @mcp.tool()
    async def fundamentals_get_growth(
        symbol: str,
        period: str = "annual",
        limit: int = 10,
        region: str = "US",
        start_date: str | None = None,
        end_date: str | None = None,
        as_of_date: str | None = None,
    ) -> list[dict]:
        return await fetch_with_failover(
            "financial_growth",
            "growth",
            region=region,
            symbol=symbol,
            period=period,
            limit=limit,
            start_date=start_date,
            end_date=end_date,
            as_of_date=as_of_date,
        )

    @mcp.tool()
    async def fundamentals_get_esg(
        symbol: str,
        region: str = "US",
        start_date: str | None = None,
        end_date: str | None = None,
        as_of_date: str | None = None,
    ) -> list[dict]:
        return await fetch_with_failover(
            "esg_score",
            "esg",
            region=region,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            as_of_date=as_of_date,
        )

    @mcp.tool()
    async def fundamentals_get_piotroski(
        symbol: str,
        period: str = "annual",
        limit: int = 10,
        region: str = "US",
        start_date: str | None = None,
        end_date: str | None = None,
        as_of_date: str | None = None,
    ) -> list[dict]:
        return await fetch_with_failover(
            "piotroski_score",
            "piotroski",
            region=region,
            symbol=symbol,
            period=period,
            limit=limit,
            start_date=start_date,
            end_date=end_date,
            as_of_date=as_of_date,
        )

    @mcp.tool()
    async def fundamentals_get_consensus_estimates(
        identifier: str,
        period_type: str = "annual",
        num_periods_forward: int = 4,
        provider: str = "spglobal",
        region: str = "US",
    ) -> Any:
        require_region(region, allowed={"US"}, data_type="consensus_estimates")
        if provider == "lseg":
            return await call_external_tool(
                "lseg",
                "qa_ibes_consensus",
                identifier=identifier,
                period_type=period_type,
                num_periods_forward=num_periods_forward,
            )
        return await call_external_tool(
            "spglobal",
            "get_consensus_estimates_from_identifiers",
            identifier=identifier,
            period_type=period_type,
            num_periods_forward=num_periods_forward,
        )
