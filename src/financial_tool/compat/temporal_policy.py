"""Canonical temporal capabilities and argument binding for ProFinR tools."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Literal


TemporalCapability = Literal[
    "range", "as_of_only", "snapshot_only", "discovery", "unknown"
]

START_ALIASES = ("start_date", "date_from", "from_date", "start")
END_ALIASES = ("end_date", "date_to", "to_date", "end")
AS_OF_ALIASES = ("as_of_date", "as_of", "decision_date")


@dataclass(frozen=True)
class ToolTemporalPolicy:
    tool_name: str
    capability: TemporalCapability
    start_fields: tuple[str, ...] = ()
    end_fields: tuple[str, ...] = ()
    as_of_fields: tuple[str, ...] = ()
    output_date_fields: tuple[str, ...] = ()
    fiscal_period_fields: tuple[str, ...] = ()
    publication_fields: tuple[str, ...] = ()
    requires_dated_output: bool = True
    supports_warmup: bool = False


_KNOWN_CAPABILITIES: dict[str, TemporalCapability] = {
    "market_data_get_history": "range",
    "market_data_get_multi_history": "range",
    "market_data_get_technical": "range",
    "macro_get_series": "range",
    "fundamentals_get_income": "range",
    "fundamentals_get_balance": "range",
    "fundamentals_get_cashflow": "range",
    "fundamentals_get_profile": "range",
    "analytics_get_correlation": "range",
    "technical_detect_patterns": "range",
    "risk_get_var": "range",
    "filings_search": "range",
    "filing_get_company_filings": "range",
    "filings_get_sec": "range",
    "news_get_headlines": "range",
    "news_get_alpaca_articles": "range",
    "news_get_finnhub_news": "range",
    "news_get_alphavantage_news": "range",
    "news_search": "range",
    "news_search_headlines": "range",
    "search_news": "range",
    "market_data_get_quote": "as_of_only",
    "market_data_get_market_status": "as_of_only",
    "valuation_get_dcf": "snapshot_only",
    "ratios_ttm": "snapshot_only",
    "key_metrics_ttm": "snapshot_only",
    "search_web": "discovery",
    "search_web_read": "discovery",
    "macro_search_fred": "discovery",
}


def _date(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value).strip().replace("/", "-")[:10])
    except (TypeError, ValueError):
        return None


def _bounded_date(value: Any, fallback: str | None, *, lower: bool) -> str | None:
    candidate = _date(value)
    default = _date(fallback)
    if candidate is None:
        return default.isoformat() if default else None
    if default is None:
        return candidate.isoformat()
    return (max(candidate, default) if lower else min(candidate, default)).isoformat()


def _declared_aliases(parameter_names: set[str], aliases: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(alias for alias in aliases if alias in parameter_names)


def temporal_policy_for(
    tool_name: str,
    parameter_names: set[str] | frozenset[str] | None = None,
    *,
    declared_capability: TemporalCapability | None = None,
) -> ToolTemporalPolicy:
    """Build one policy from catalog/callable metadata and known tool facts."""

    name = str(tool_name)
    parameters = set(parameter_names or ())
    start_fields = _declared_aliases(parameters, START_ALIASES)
    end_fields = _declared_aliases(parameters, END_ALIASES)
    as_of_fields = _declared_aliases(parameters, AS_OF_ALIASES)
    capability = declared_capability or _KNOWN_CAPABILITIES.get(name)
    if capability is None:
        if start_fields or end_fields:
            capability = "range"
        elif as_of_fields:
            capability = "as_of_only"
        else:
            capability = "unknown"
    return ToolTemporalPolicy(
        tool_name=name,
        capability=capability,
        start_fields=start_fields,
        end_fields=end_fields,
        as_of_fields=as_of_fields,
        output_date_fields=(
            "date",
            "observation_date",
            "period_end",
            "fiscal_period_end",
            "published_at",
            "filed_at",
        ),
        fiscal_period_fields=("fiscal_period", "period", "period_label"),
        publication_fields=("published_at", "filed_at", "publishedDate", "publishedDateTime"),
        supports_warmup=name in {"market_data_get_technical", "technical_detect_patterns"},
    )


def bind_temporal_arguments(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    parameter_names: set[str] | frozenset[str],
    query_start: str | None,
    query_end: str | None,
    query_as_of: str | None,
    declared_capability: TemporalCapability | None = None,
) -> tuple[dict[str, Any], ToolTemporalPolicy]:
    """Clamp every declared physical alias to the logical query window."""

    bounded = dict(arguments)
    policy = temporal_policy_for(
        tool_name,
        parameter_names,
        declared_capability=declared_capability,
    )
    if not query_as_of and not query_start and not query_end:
        return bounded, policy

    start_field = next(
        (field for field in policy.start_fields if field in bounded),
        policy.start_fields[0] if policy.start_fields else None,
    )
    end_field = next(
        (field for field in policy.end_fields if field in bounded),
        policy.end_fields[0] if policy.end_fields else None,
    )
    as_of_field = next(
        (field for field in policy.as_of_fields if field in bounded),
        policy.as_of_fields[0] if policy.as_of_fields else None,
    )
    if start_field and query_start:
        bounded[start_field] = _bounded_date(
            bounded.get(start_field), query_start, lower=True
        )
    if end_field and query_end:
        bounded[end_field] = _bounded_date(
            bounded.get(end_field), query_end, lower=False
        )
    if as_of_field and query_as_of:
        bounded[as_of_field] = _bounded_date(
            bounded.get(as_of_field), query_as_of, lower=False
        )
    return bounded, policy


def temporal_unsafe_marker(
    policy: ToolTemporalPolicy,
    *,
    requested_window: dict[str, Any],
    reason: str | None = None,
) -> dict[str, Any]:
    """Return a provider-free deterministic failure for unsafe historical calls."""

    if reason is None:
        reason = {
            "snapshot_only": "snapshot_only_tool_for_historical_query",
            "unknown": "temporal_capability_unknown",
            "discovery": "temporal_discovery_requires_evidence_date",
        }.get(policy.capability, "temporal_binding_unsupported")
    return {
        "status": "temporal_unsafe",
        "tool": policy.tool_name,
        "reason": reason,
        "requested_window": dict(requested_window),
        "temporal_safe": False,
    }


def technical_warmup_start_date(
    output_start_date: str | None,
    period: Any = 14,
) -> str | None:
    """Return a conservative calendar start for technical-indicator warmup."""

    parsed = _date(output_start_date)
    if parsed is None:
        return output_start_date
    try:
        bars = max(1, int(period))
    except (TypeError, ValueError):
        bars = 14
    return (parsed - timedelta(days=bars * 3 + 7)).isoformat()
