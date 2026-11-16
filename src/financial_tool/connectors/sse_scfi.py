"""Shanghai Shipping Exchange SCFI connector."""

from __future__ import annotations

from datetime import date
from html.parser import HTMLParser
from typing import Any

import httpx

from .base import BaseConnector, register_connector
from ..errors import SourceUnavailableError
from ..network_config import DEFAULT_USER_AGENT


_SCFI_URL = "https://www.sse.net.cn/index/singleIndex?indexType=scfi"


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        tag = tag.lower()
        if tag == "tr":
            if self._row:
                self.rows.append(self._row)
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            self._cell = []

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"td", "th"} and self._row is not None and self._cell is not None:
            value = " ".join(" ".join(self._cell).split())
            self._row.append(value)
            self._cell = None
        elif tag == "tr" and self._row:
            self.rows.append(self._row)
            self._row = None


def _as_date(value: str) -> str | None:
    try:
        return date.fromisoformat(value.strip()).isoformat()
    except ValueError:
        return None


def _as_number(value: str) -> float | None:
    cleaned = value.strip().replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _parse_scfi_html(html: str) -> list[dict[str, Any]]:
    parser = _TableParser()
    parser.feed(html)
    for row in parser.rows:
        if not row or "comprehensive index" not in row[0].lower():
            continue
        dates = [item for item in row if _as_date(item)]
        numbers = [number for item in row if (number := _as_number(item)) is not None]
        if len(dates) < 2 or len(numbers) < 2:
            continue
        previous_value = numbers[-3] if len(numbers) >= 3 else numbers[-2]
        current_value = numbers[-2] if len(numbers) >= 3 else numbers[-1]
        change = numbers[-1] if len(numbers) >= 3 else current_value - previous_value
        return [
            {
                "series_id": "SCFI",
                "date": _as_date(dates[-1]),
                "value": current_value,
                "previous_value": previous_value,
                "change": change,
                "frequency": "weekly",
                "unit": "points",
                "source": "sse_scfi",
            }
        ]
    return []


class SSESCFIConnector(BaseConnector):
    name = "sse_scfi"
    actions = frozenset({"series"})

    async def fetch(
        self,
        action: str,
        series_id: str = "SCFI",
        start_date: str | None = None,
        end_date: str | None = None,
        **_: Any,
    ) -> Any:
        if action != "series":
            raise ValueError(f"SSE SCFI: unsupported action '{action}'")
        if str(series_id).strip().upper() != "SCFI":
            raise ValueError("SSE SCFI supports only series_id='SCFI'")
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(
                    _SCFI_URL,
                    headers={"User-Agent": DEFAULT_USER_AGENT},
                )
            response.raise_for_status()
        except Exception as exc:
            raise SourceUnavailableError(
                self.name,
                reason=f"official SCFI page unavailable: {exc}",
            ) from exc

        rows = _parse_scfi_html(response.text)
        if not rows:
            raise SourceUnavailableError(
                self.name,
                reason="official SCFI page did not expose a current composite observation",
            )
        filtered = [
            row
            for row in rows
            if (not start_date or row["date"] >= str(start_date)[:10])
            and (not end_date or row["date"] <= str(end_date)[:10])
        ]
        if filtered:
            return filtered
        return {
            "status": "source_unavailable",
            "series_id": "SCFI",
            "retryable": False,
            "reason": "official page exposes current data only; requested historical observation is unavailable",
        }

    async def health_check(self) -> bool:
        return True


register_connector("sse_scfi", SSESCFIConnector)
