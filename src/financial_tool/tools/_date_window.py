"""Date-window normalization shared by historical market-data tools."""

from __future__ import annotations

from datetime import date, datetime

from ..errors import DataValidationError


def _parse_iso_date(value: str | None) -> str | None:
    """Return an ISO calendar date for an ISO date/datetime, otherwise None."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10]).isoformat()
    except ValueError:
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()
        except ValueError:
            return None


def normalize_market_data_window(
    *,
    start_date: str | None,
    end_date: str | None,
    as_of_date: str | None,
) -> tuple[str | None, str | None, str | None]:
    """Return a historical request window that cannot extend past ``as_of_date``.

    Invalid start/end values are left to the existing provider validation path.
    A supplied but invalid ``as_of_date`` is rejected because it cannot safely
    anchor a historical request. Omitting ``end_date`` means "through as-of".
    """
    parsed_as_of = _parse_iso_date(as_of_date)
    if as_of_date is not None and str(as_of_date).strip() and not parsed_as_of:
        raise DataValidationError("as_of_date", "must be an ISO date or datetime")
    # ``as_of_date`` is an evidence cutoff, not a synonym for the server clock.
    # A historical benchmark query without one must preserve its requested
    # interval; its temporal context is resolved by the benchmark workflow.
    anchor = parsed_as_of

    parsed_start = _parse_iso_date(start_date)
    if anchor and parsed_start and parsed_start > anchor:
        raise DataValidationError(
            "start_date",
            f"requested window is entirely after as_of_date ({anchor})",
        )

    parsed_end = _parse_iso_date(end_date)
    if parsed_end is not None and anchor:
        effective_end = min(parsed_end, anchor)
    elif parsed_end is not None:
        effective_end = parsed_end
    elif anchor:
        effective_end = anchor
    else:
        effective_end = end_date

    return start_date, effective_end, anchor
