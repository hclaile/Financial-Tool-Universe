"""Frequency-aware coverage checks for macro time series."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any


def _parse_date(value: str) -> date:
    return datetime.strptime(str(value), "%Y-%m-%d").date()


def _period_end_for_observation(observation_date: date, frequency_short: str) -> date:
    freq = str(frequency_short or "").upper()
    if freq == "Q":
        quarter = ((observation_date.month - 1) // 3) + 1
        month = quarter * 3
        if month == 3:
            return date(observation_date.year, 3, 31)
        if month == 6:
            return date(observation_date.year, 6, 30)
        if month == 9:
            return date(observation_date.year, 9, 30)
        return date(observation_date.year, 12, 31)
    if freq == "A":
        return date(observation_date.year, 12, 31)
    return observation_date


def _expected_count(start: date, end: date, frequency_short: str) -> int:
    freq = str(frequency_short or "").upper()
    if freq == "Q":
        start_q = start.year * 4 + ((start.month - 1) // 3)
        end_q = end.year * 4 + ((end.month - 1) // 3)
        return max(0, end_q - start_q + 1)
    if freq == "M":
        return max(0, (end.year - start.year) * 12 + end.month - start.month + 1)
    if freq == "A":
        return max(0, end.year - start.year + 1)
    return 1


def assess_macro_series_coverage(
    *,
    series_id: str,
    requested_start: str,
    requested_end: str,
    observations: list[dict[str, Any]],
    frequency_short: str | None,
) -> dict[str, Any]:
    start = _parse_date(requested_start)
    end = _parse_date(requested_end)
    parsed_dates = [
        _parse_date(str(item.get("date")))
        for item in observations
        if isinstance(item, dict) and item.get("date")
    ]
    expected = _expected_count(start, end, str(frequency_short or ""))
    if not parsed_dates:
        return {
            "series_id": series_id,
            "status": "missing",
            "expected_observation_count": expected,
            "observed_observation_count": 0,
            "covered_period_end": None,
        }
    covered_end = max(
        _period_end_for_observation(item, str(frequency_short or ""))
        for item in parsed_dates
    )
    observed_in_range = [
        item
        for item in parsed_dates
        if start <= _period_end_for_observation(item, str(frequency_short or "")) <= end
        or start <= item <= end
    ]
    status = "covered" if len(observed_in_range) >= expected and covered_end >= end else "partial"
    return {
        "series_id": series_id,
        "status": status,
        "expected_observation_count": expected,
        "observed_observation_count": len(observed_in_range),
        "covered_period_end": covered_end.isoformat(),
    }
