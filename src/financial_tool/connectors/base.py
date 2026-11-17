"""Base connector interface, source router, and failover engine."""

from __future__ import annotations

import asyncio
import importlib
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import httpx
import yaml

from ..errors import SourceUnavailableError, UnsupportedRegionError
from finance_tool_universe.config import catalog_dir

logger = logging.getLogger("financial_tool.connector")

CATALOGS_DIR = catalog_dir()


def decode_json_response(response: httpx.Response, *, source: str) -> Any:
    """Decode JSON without copying provider response bodies into errors."""

    try:
        return response.json()
    except ValueError as exc:
        text = str(getattr(response, "text", "") or "").strip()
        content = getattr(response, "content", b"") or b""
        reason = "empty JSON response" if not text and not content else "non-JSON response"
        headers = getattr(response, "headers", {}) or {}
        content_type = str(headers.get("content-type", ""))
        status = getattr(response, "status_code", "unknown")
        raise SourceUnavailableError(
            source,
            f"{reason} status={status} content_type={content_type or 'unknown'}",
        ) from exc


# ── Abstract base ────────────────────────────────────────────

class BaseConnector(ABC):
    name: str = "base"
    actions: frozenset[str] = frozenset()

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}

    @abstractmethod
    async def fetch(self, action: str, **kwargs: Any) -> Any:
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        ...


# ── Connector registry ───────────────────────────────────────

_CONNECTORS: dict[str, type[BaseConnector]] = {}


def register_connector(name: str, cls: type[BaseConnector]) -> None:
    _CONNECTORS[name] = cls


def get_connector_class(name: str) -> type[BaseConnector]:
    if name not in _CONNECTORS:
        raise KeyError(f"Unknown connector: {name}")
    return _CONNECTORS[name]


def list_connectors() -> list[str]:
    return list(_CONNECTORS.keys())


def declared_actions_for(source_name: str) -> frozenset[str]:
    """Return the connector's explicitly declared, supported actions.

    Priority rules and MCP tools use action names as a cross-module contract;
    exposing them here lets tests detect a route that points to an action the
    connector cannot dispatch before a live provider call is attempted.
    """
    return get_connector_class(source_name).actions


def _build_connector(name: str) -> BaseConnector:
    """Instantiate a connector with its config from sources.yaml."""
    cls = _CONNECTORS[name]
    sources_cfg = _load_sources_config()
    cfg = _runtime_source_config(sources_cfg.get(name, {}))
    return cls(config=cfg)


def get_source_connector(source_name: str) -> BaseConnector:
    """Return a connector by explicit source name."""
    if source_name not in _CONNECTORS:
        raise KeyError(f"Unknown connector: {source_name}")
    return _build_connector(source_name)


def _load_sources_config() -> dict:
    from finance_tool_universe.config import load_config
    sources = dict(load_config().get("sources") or {})
    if "local_files" in sources:
        sources["local_files"] = dict(sources["local_files"])
        sources["local_files"]["base_path"] = os.environ.get("FTU_DATA_DIR", "data")
    return sources


def source_config_is_available(config: Any) -> bool:
    """Return whether one source config can produce runtime credentials.

    This mirrors connector runtime resolution: ``token_env`` is resolved from
    the environment first, then non-environment-name values are treated as
    configured literal tokens for backward compatibility. It returns only a
    boolean and never exposes credential values.
    """
    if not isinstance(config, dict) or config.get("enabled") is False:
        return False
    if str(config.get("api_key") or "").strip():
        return True

    credential_values: list[str] = []
    token_env = str(config.get("token_env") or "").strip()
    if token_env:
        credential_values.append(_configured_token_value(token_env))
    for field in ("key_id_env", "secret_key_env", "api_key_env", "auth_token_env"):
        raw_value = str(config.get(field) or "").strip()
        if not raw_value:
            continue
        credential_values.append(_configured_credential_value(raw_value))

    if not credential_values:
        return True
    return all(bool(value) for value in credential_values)


def source_is_configured(source_name: str) -> bool:
    """Return whether an enabled source has the credentials it declares."""

    return source_config_is_available(_load_sources_config().get(source_name, {}))


_ENVIRONMENT_NAME = re.compile(r"^[A-Z_][A-Z0-9_]*$")


def _configured_credential_value(value: str) -> str:
    """Resolve a strict environment-variable reference without literal fallback."""

    candidate = str(value or "").strip()
    if not _ENVIRONMENT_NAME.fullmatch(candidate):
        return ""
    return str(os.getenv(candidate, "")).strip()


def _configured_token_value(value: str) -> str:
    """Resolve token_env only as an environment variable name."""

    candidate = str(value or "").strip()
    if not candidate:
        return ""
    return _configured_credential_value(candidate)


def _runtime_source_config(config: Any) -> dict[str, Any]:
    """Resolve a configured token environment variable for connector use."""
    runtime = dict(config) if isinstance(config, dict) else {}
    token_value = _configured_token_value(str(runtime.get("token_env") or ""))
    if token_value:
        runtime["api_key"] = token_value
    return runtime


# ── Source priority router ──────────────────────────────────

_priority_rules: list[dict] | None = None
_failover_cfg: dict = {}


def _load_priority_rules() -> tuple[list[dict], dict]:
    global _priority_rules, _failover_cfg
    if _priority_rules is not None:
        return _priority_rules, _failover_cfg
    path = CATALOGS_DIR / "source_priority.yaml"
    if path.exists():
        data = yaml.safe_load(path.read_text())
        _priority_rules = data.get("priority_rules", [])
        _failover_cfg = data.get("failover", {})
    else:
        _priority_rules = []
        _failover_cfg = {}
    return _priority_rules, _failover_cfg


def resolve_sources(data_type: str, region: str = "CN") -> list[str]:
    """Return sources for an exact regional rule or a genuinely global rule."""
    rules, _ = _load_priority_rules()
    normalized_region = str(region).strip().upper()
    matching_rules = [
        rule for rule in rules if rule["data_type"] == data_type
    ]
    regional_rules = [
        rule for rule in matching_rules if "region" in rule
    ]
    for rule in regional_rules:
        if str(rule.get("region", "")).strip().upper() == normalized_region:
            sources = list(rule.get("sources") or [])
            if not sources:
                raise UnsupportedRegionError(data_type, normalized_region)
            return sources
    if regional_rules:
        raise UnsupportedRegionError(data_type, normalized_region)

    for rule in matching_rules:
        if "region" not in rule:
            return list(rule.get("sources") or [])

    for rule in rules:
        if rule["data_type"] == "*" and "region" not in rule:
            return rule["sources"]
    return []


# ── Circuit breaker ─────────────────────────────────────────

class CircuitBreaker:
    def __init__(self, threshold: int = 5, reset_timeout: float = 60.0) -> None:
        self._threshold = threshold
        self._reset_timeout = reset_timeout
        self._failures: dict[str, int] = {}
        self._open_until: dict[str, float] = {}

    def configure(self, *, threshold: Any, reset_timeout: Any) -> None:
        """Normalize failover configuration before it reaches time arithmetic."""

        self._threshold = max(1, int(threshold))
        self._reset_timeout = max(0.0, _duration_seconds(reset_timeout))

    def is_open(self, source: str) -> bool:
        if source not in self._open_until:
            return False
        if time.time() < self._open_until[source]:
            return True
        # half-open: allow one probe
        del self._open_until[source]
        self._failures[source] = 0
        return False

    def record_failure(self, source: str) -> None:
        self._failures[source] = self._failures.get(source, 0) + 1
        if self._failures[source] >= self._threshold:
            self._open_until[source] = time.time() + self._reset_timeout
            logger.warning("Circuit breaker OPEN for source=%s", source)

    def record_success(self, source: str) -> None:
        self._failures[source] = 0
        self._open_until.pop(source, None)


# ── Fetch with failover ──────────────────────────────────────

_breaker = CircuitBreaker()
_SOURCE_CACHE: dict[tuple[Any, ...], Any] = {}
_SOURCE_SEMAPHORES: dict[str, asyncio.Semaphore] = {}
_RATE_LIMIT_OPEN_UNTIL: dict[str, float] = {}
_AUTH_OPEN_UNTIL: dict[str, float] = {}

_TEMPORAL_DATE_FIELDS = (
    "date",
    "published_at",
    "publishedAt",
    "filed_date",
    "filing_date",
    "fillingDate",
    "accepted_date",
    "acceptedDate",
    "date_reported",
    "dateReported",
    "transaction_date",
    "transactionDate",
    "fiscalDateEnding",
    "end",
    "period",
)
_TEMPORAL_LIST_FIELDS = ("rows", "articles", "data", "historical", "results", "feed", "observations")
_TEMPORAL_MIN_LIMIT = 100
_HISTORICAL_SNAPSHOT_ACTIONS = {
    "quote",
    "realtime",
    "market_status",
    "market_cap",
    "dcf",
    "rating",
    "ratios_ttm",
    "key_metrics_ttm",
    "price_target",
    "sentiment",
    "options",
}


def _is_empty_result(result: Any) -> bool:
    if result is None:
        return True
    if isinstance(result, (list, tuple, set, str, bytes)):
        return len(result) == 0
    if isinstance(result, dict):
        if not result:
            return True
        if "rows" in result and isinstance(result["rows"], list):
            return len(result["rows"]) == 0
    return False


def _duration_seconds(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower()
    multipliers = {
        "ms": 0.001,
        "s": 1.0,
        "m": 60.0,
    }
    for suffix, multiplier in multipliers.items():
        if text.endswith(suffix):
            return float(text[: -len(suffix)] or 0) * multiplier
    return float(text)


def _retry_delay_for_source(source_name: str, default_delay: float, failover_cfg: dict) -> float:
    source_delays = failover_cfg.get("source_retry_delays", {})
    if source_name in source_delays:
        return _duration_seconds(source_delays[source_name])
    if source_name == "yfinance":
        return 30.0
    return default_delay


def _source_rate_limit_cooldown(source_name: str, failover_cfg: dict) -> float:
    cooldowns = failover_cfg.get("source_rate_limit_cooldown", {})
    if source_name in cooldowns:
        return _duration_seconds(cooldowns[source_name])
    if source_name == "yfinance":
        return 30.0
    return 0.0


def _is_rate_limited_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "too many requests" in text or "rate limited" in text or "429" in text


def _is_auth_error(exc: Exception) -> bool:
    return isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in {401, 403}


def _is_non_retryable_error(exc: Exception) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in {400, 401, 403, 404, 422}
    text = str(exc).lower()
    return "unsupported action" in text or "proxy/network" in text


def _source_auth_cooldown(source_name: str, failover_cfg: dict) -> float:
    cooldowns = failover_cfg.get("source_auth_cooldown", {})
    return _duration_seconds(cooldowns.get(source_name, "300s"))


def _temporal_requested(kwargs: dict[str, Any]) -> bool:
    return any(
        kwargs.get(key)
        for key in (
            "start_date",
            "end_date",
            "as_of_date",
            "from_date",
            "to_date",
            "date_from",
            "date_to",
            "start",
            "end",
        )
    )


def _temporal_window(kwargs: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    start_date = kwargs.get("start_date") or kwargs.get("from_date") or kwargs.get("date_from") or kwargs.get("start")
    end_date = kwargs.get("end_date") or kwargs.get("to_date") or kwargs.get("date_to") or kwargs.get("end")
    as_of_date = kwargs.get("as_of_date") or end_date
    return (
        str(start_date)[:10] if start_date else None,
        str(end_date)[:10] if end_date else None,
        str(as_of_date)[:10] if as_of_date else None,
    )


def expand_temporal_limit_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Increase limit for historical requests so limit-style APIs cover the target window."""
    if not _temporal_requested(kwargs) or "limit" not in kwargs:
        return kwargs
    expanded = dict(kwargs)
    try:
        expanded["limit"] = max(int(expanded.get("limit") or 0), _TEMPORAL_MIN_LIMIT)
    except (TypeError, ValueError):
        expanded["limit"] = _TEMPORAL_MIN_LIMIT
    return expanded


def _normalize_temporal_date(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    match = re.search(r"(\d{4})[-/]?(\d{2})[-/]?(\d{2})", text)
    if not match:
        return ""
    return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"


def _row_temporal_date(row: dict[str, Any]) -> str:
    for field in _TEMPORAL_DATE_FIELDS:
        date = _normalize_temporal_date(row.get(field))
        if date:
            return date
    return ""


def _temporal_marker(
    status: str,
    source: str,
    action: str,
    kwargs: dict[str, Any],
    rows: list[dict[str, Any]] | None = None,
    reason: str = "",
) -> dict[str, Any]:
    start_date, end_date, as_of_date = _temporal_window(kwargs)
    dates = sorted(date for date in (_row_temporal_date(row) for row in (rows or [])) if date)
    verification = {
        "out_of_window": "out_of_window",
        "snapshot_blocked": "snapshot_only",
        "temporal_unverified": "unverified",
    }.get(status, "unverified")
    marker = {
        "status": status,
        "source": source,
        "action": action,
        "requested_window": {
            "start_date": start_date,
            "end_date": end_date,
            "as_of_date": as_of_date,
        },
        "temporal_safe": False,
        "reason": reason,
        "temporal_metadata": {
            "requested_start": start_date,
            "requested_end": end_date,
            "requested_as_of": as_of_date,
            "actual_min_date": dates[0] if dates else None,
            "actual_max_date": dates[-1] if dates else None,
            "date_dimension": "news_publication" if "news" in action else "market_observation",
            "fiscal_period_labels": [],
            "period_ends": [],
            "published_at": [],
            "filed_at": [],
            "temporal_safe": False,
            "date_verification": verification,
            "coverage_completeness": "unknown",
        },
    }
    if "symbol" in kwargs:
        marker["symbol"] = kwargs["symbol"]
    if dates:
        marker["actual_min_date"] = dates[0]
        marker["actual_max_date"] = dates[-1]
    return marker


def _contains_undated_rows(rows: list[Any]) -> bool:
    return any(
        isinstance(row, dict) and not _row_temporal_date(row)
        for row in rows
    )


def _attach_temporal_metadata(
    result: dict[str, Any],
    *,
    rows: list[Any],
    kwargs: dict[str, Any],
    action: str,
) -> dict[str, Any]:
    start_date, end_date, as_of_date = _temporal_window(kwargs)
    dates = sorted(date for date in (_row_temporal_date(row) for row in rows) if date)
    metadata = dict(result.get("temporal_metadata") or {})
    metadata.update(
        {
            "requested_start": start_date,
            "requested_end": end_date,
            "requested_as_of": as_of_date,
            "actual_min_date": dates[0] if dates else None,
            "actual_max_date": dates[-1] if dates else None,
            "date_dimension": "news_publication" if "news" in action else "market_observation",
            "temporal_safe": bool(rows) and not _contains_undated_rows(rows),
            "date_verification": "verified" if dates and not _contains_undated_rows(rows) else "empty",
            "coverage_completeness": "known",
        }
    )
    guarded = dict(result)
    guarded["temporal_metadata"] = metadata
    return guarded


def _filter_temporal_rows(rows: list[Any], kwargs: dict[str, Any]) -> tuple[list[Any], int]:
    start_date, end_date, as_of_date = _temporal_window(kwargs)
    filtered: list[Any] = []
    removed = 0
    for row in rows:
        if not isinstance(row, dict):
            filtered.append(row)
            continue
        date = _row_temporal_date(row)
        if not date:
            filtered.append(row)
            continue
        if start_date and date < start_date:
            removed += 1
            continue
        if end_date and date > end_date:
            removed += 1
            continue
        if as_of_date and date > as_of_date:
            removed += 1
            continue
        filtered.append(row)
    return filtered, removed


def _apply_temporal_guard(
    result: Any,
    *,
    source_name: str,
    action: str,
    kwargs: dict[str, Any],
) -> Any:
    if not _temporal_requested(kwargs):
        return result
    if action in _HISTORICAL_SNAPSHOT_ACTIONS:
        return _temporal_marker(
            "snapshot_blocked",
            "temporal_guard",
            action,
            kwargs,
            reason="Current snapshot action is not safe for historical as-of requests",
        )
    if isinstance(result, dict) and result.get("status") in {"out_of_window", "snapshot_blocked"}:
        return result
    if isinstance(result, list):
        if _contains_undated_rows(result):
            return _temporal_marker(
                "temporal_unverified",
                source_name,
                action,
                kwargs,
                rows=[row for row in result if isinstance(row, dict)],
                reason="Historical result contains rows without a verifiable date",
            )
        filtered, removed = _filter_temporal_rows(result, kwargs)
        if filtered:
            return filtered
        if removed and result and all(isinstance(row, dict) and _row_temporal_date(row) for row in result):
            return _temporal_marker(
                "out_of_window",
                source_name,
                action,
                kwargs,
                rows=[row for row in result if isinstance(row, dict)],
                reason="Source returned rows outside requested temporal window",
            )
        return filtered
    if isinstance(result, dict):
        for field in _TEMPORAL_LIST_FIELDS:
            value = result.get(field)
            if isinstance(value, list):
                if _contains_undated_rows(value):
                    return _temporal_marker(
                        "temporal_unverified",
                        source_name,
                        action,
                        kwargs,
                        rows=[row for row in value if isinstance(row, dict)],
                        reason="Historical result contains rows without a verifiable date",
                    )
                filtered, removed = _filter_temporal_rows(value, kwargs)
                guarded = _attach_temporal_metadata(
                    dict(result), rows=filtered, kwargs=kwargs, action=action
                )
                guarded[field] = filtered
                if removed:
                    guarded["filtered_temporal_rows"] = guarded.get("filtered_temporal_rows", 0) + removed
                if not filtered and removed and value and all(isinstance(row, dict) and _row_temporal_date(row) for row in value):
                    return _temporal_marker(
                        "out_of_window",
                        source_name,
                        action,
                        kwargs,
                        rows=[row for row in value if isinstance(row, dict)],
                        reason="Source returned rows outside requested temporal window",
                    )
                return guarded
        date = _row_temporal_date(result)
        if date:
            start_date, end_date, as_of_date = _temporal_window(kwargs)
            if (start_date and date < start_date) or (end_date and date > end_date) or (as_of_date and date > as_of_date):
                return _temporal_marker(
                    "out_of_window",
                    source_name,
                    action,
                    kwargs,
                    rows=[result],
                    reason="Source returned a row outside requested temporal window",
                )
    return result


def guard_temporal_result(
    result: Any,
    *,
    source_name: str,
    action: str,
    **kwargs: Any,
) -> Any:
    """Apply the same temporal safety contract used by failover routing."""
    return _apply_temporal_guard(
        result,
        source_name=source_name,
        action=action,
        kwargs=kwargs,
    )


def _cache_key(source_name: str, action: str, kwargs: dict[str, Any]) -> tuple[Any, ...]:
    def freeze(value: Any) -> Any:
        if isinstance(value, dict):
            return tuple(sorted((key, freeze(item)) for key, item in value.items()))
        if isinstance(value, (list, tuple)):
            return tuple(freeze(item) for item in value)
        if isinstance(value, set):
            return tuple(sorted(freeze(item) for item in value))
        return value

    return (
        source_name,
        action,
        tuple(sorted((key, freeze(value)) for key, value in kwargs.items())),
    )


def _source_semaphore(source_name: str, failover_cfg: dict) -> asyncio.Semaphore:
    limits = failover_cfg.get("source_concurrency", {})
    limit = int(limits.get(source_name, 1 if source_name == "yfinance" else 8))
    semaphore = _SOURCE_SEMAPHORES.get(source_name)
    if semaphore is None:
        semaphore = asyncio.Semaphore(max(1, limit))
        _SOURCE_SEMAPHORES[source_name] = semaphore
    return semaphore


async def _fetch_source(conn: BaseConnector, source_name: str, action: str, fo_cfg: dict, kwargs: dict[str, Any]) -> Any:
    if source_name == "yfinance":
        async with _source_semaphore(source_name, fo_cfg):
            return await conn.fetch(action, **kwargs)
    return await conn.fetch(action, **kwargs)


async def fetch_with_failover(
    data_type: str,
    action: str,
    region: str = "CN",
    **kwargs: Any,
) -> Any:
    """Try each explicitly routed source in priority order with retry.

    The failover engine never rewrites or remaps a caller-provided symbol.
    Region changes and provider-format conversions must be explicit at the
    tool boundary; this prevents a valid symbol from becoming an unrelated or
    invalid yfinance symbol during fallback.
    """
    if _temporal_requested(kwargs) and action in _HISTORICAL_SNAPSHOT_ACTIONS:
        return _temporal_marker(
            "snapshot_blocked",
            "temporal_guard",
            action,
            kwargs,
            reason="Current snapshot action is not safe for historical as-of requests",
        )
    sources = resolve_sources(data_type, region)
    _, fo_cfg = _load_priority_rules()
    max_retries = max(0, int(fo_cfg.get("max_retries", 2)))
    retry_delay = _duration_seconds(fo_cfg.get("retry_delay", 1))
    breaker_cfg = fo_cfg.get("circuit_breaker", {})
    _breaker.configure(
        threshold=breaker_cfg.get("failure_threshold", 5),
        reset_timeout=breaker_cfg.get("reset_timeout", 60),
    )

    errors: list[str] = []
    attempts: list[dict[str, str]] = []
    for source_name in sources:
        if time.time() < _AUTH_OPEN_UNTIL.get(source_name, 0):
            logger.info("Skipping source=%s (auth cooldown open)", source_name)
            continue
        if time.time() < _RATE_LIMIT_OPEN_UNTIL.get(source_name, 0):
            logger.info("Skipping source=%s (rate-limit cooldown open)", source_name)
            continue
        if _breaker.is_open(source_name):
            logger.info("Skipping source=%s (circuit open)", source_name)
            continue
        if source_name not in _CONNECTORS:
            logger.warning("Source %s not registered, skipping", source_name)
            continue

        # A source attempt should trip the circuit breaker exactly once, and
        # only when a real call failed.  Empty results and generic errors fall
        # through to the trailing record below; deterministic no-retry paths
        # (TypeError, SourceUnavailableError) set this and record inline.
        record_failure = True
        for attempt in range(max_retries + 1):
            try:
                request_kwargs = expand_temporal_limit_kwargs(kwargs)
                key = _cache_key(source_name, action, request_kwargs)
                if source_name == "yfinance" and key in _SOURCE_CACHE:
                    return _apply_temporal_guard(
                        _SOURCE_CACHE[key],
                        source_name=source_name,
                        action=action,
                        kwargs=kwargs,
                    )
                conn = _build_connector(source_name)
                result = await _fetch_source(conn, source_name, action, fo_cfg, request_kwargs)
                result = _apply_temporal_guard(
                    result,
                    source_name=source_name,
                    action=action,
                    kwargs=kwargs,
                )
                if _is_empty_result(result):
                    attempts.append(
                        {"source": source_name, "action": action, "outcome": "empty_result"}
                    )
                    errors.append(
                        f"{source_name}(attempt {attempt + 1}): empty result"
                    )
                    logger.info(
                        "Source %s returned an empty result for data_type=%s action=%s",
                        source_name,
                        data_type,
                        action,
                    )
                    record_failure = False
                    break
                _breaker.record_success(source_name)
                if source_name == "yfinance":
                    _SOURCE_CACHE[key] = result
                return result
            except TypeError as exc:
                # A provider/connector TypeError is deterministic for unchanged
                # arguments, so retrying the same source cannot repair it.
                errors.append(f"{source_name}(attempt {attempt + 1}): {str(exc)}")
                attempts.append(
                    {
                        "source": source_name,
                        "action": action,
                        "outcome": "type_error",
                        "error_type": type(exc).__name__,
                    }
                )
                logger.info(
                    "Source %s TypeError (no retry): %s", source_name, exc,
                )
                _breaker.record_failure(source_name)
                record_failure = False
                break  # Skip remaining retries — TypeError won't resolve
            except SourceUnavailableError as exc:
                errors.append(f"{source_name}(attempt {attempt + 1}): {str(exc)}")
                attempts.append(
                    {
                        "source": source_name,
                        "action": action,
                        "outcome": "source_unavailable",
                        "error_type": type(exc).__name__,
                    }
                )
                logger.info(
                    "Source %s unavailable (no retry): %s", source_name, exc,
                )
                # A deterministic unavailability (e.g. non-JSON response) is not
                # a transient failure worth tripping the breaker over; skip it
                # and let failover continue without poisoning the source.
                record_failure = False
                break
            except Exception as exc:
                rate_limited = _is_rate_limited_error(exc)
                errors.append(f"{source_name}(attempt {attempt + 1}): {str(exc)}")
                attempts.append(
                    {
                        "source": source_name,
                        "action": action,
                        "outcome": "rate_limited" if rate_limited else "error",
                        "error_type": type(exc).__name__,
                    }
                )
                logger.info(
                    "Source %s attempt %d failed: %s", source_name, attempt + 1, exc,
                )
                if source_name == "yfinance" and rate_limited:
                    cooldown = _source_rate_limit_cooldown(source_name, fo_cfg)
                    if cooldown > 0:
                        _RATE_LIMIT_OPEN_UNTIL[source_name] = time.time() + cooldown
                if _is_auth_error(exc):
                    _AUTH_OPEN_UNTIL[source_name] = time.time() + _source_auth_cooldown(
                        source_name, fo_cfg,
                    )
                if _is_non_retryable_error(exc):
                    break
                if attempt < max_retries:
                    await asyncio.sleep(
                        _retry_delay_for_source(source_name, retry_delay, fo_cfg)
                    )

        if record_failure:
            _breaker.record_failure(source_name)

    reason = "All sources failed: " + "; ".join(str(error) for error in errors)
    logger.error(
        "All sources failed for data_type=%s action=%s: %s",
        data_type,
        action,
        reason,
    )
    raise SourceUnavailableError(
        data_type,
        reason=reason,
        attempts=attempts,
    )


# ── Convenience accessor for tools ──────────────────────────

def get_connector(group: str) -> BaseConnector:
    """Backward-compatible single-connector accessor (no failover)."""
    from finance_tool_universe.catalog import load_catalog

    metadata = load_catalog()["groups"].get(group, {})
    default = metadata.get("default_source")
    if default and default in _CONNECTORS:
        return _build_connector(default)
    # fallback: first registered connector
    for name in _CONNECTORS:
        return _build_connector(name)
    raise KeyError(f"No connector available for group: {group}")


# ── Auto-register all built-in connectors ───────────────────

def _auto_register() -> None:
    modules = [
        "akshare", "yfinance", "tushare", "okx", "ccxt", "fmp",
        "sec_edgar", "fred", "news", "alpaca_news", "onchain", "local_files", "external_mcp",
        "tavily", "google_search", "alphavantage", "finnhub", "crawler_sources", "sse_scfi",
    ]
    package = __name__.rsplit(".", 1)[0]
    for module in modules:
        try:
            importlib.import_module(f"{package}.{module}")
        except ImportError as exc:
            logger.info("Skipping optional connector %s: %s", module, exc)

_auto_register()
