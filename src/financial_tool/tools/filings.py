"""Filings tools — search and retrieve regulatory filings with failover."""

from __future__ import annotations

from ..connectors.base import fetch_with_failover


def normalize_filing_rows(payload: Any) -> dict[str, Any]:
    """Keep only filing rows that can be followed by a content request."""

    if isinstance(payload, dict):
        rows = payload.get("rows") or payload.get("results") or []
    else:
        rows = payload
    rows = [dict(row) for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []
    usable: list[dict[str, Any]] = []
    rejected = 0
    for row in rows:
        if any(str(row.get(field) or "").strip() for field in ("filing_id", "accession_number", "document_url")):
            usable.append(row)
        else:
            rejected += 1
    result: dict[str, Any] = {
        "status": "success" if usable else "empty_result",
        "results": usable,
        "rejected_rows": rejected,
    }
    if not usable and rejected:
        result["reason"] = "unusable_filing_reference"
    return result


def register(mcp) -> None:

    @mcp.tool()
    async def filings_search(
        query: str,
        form_type: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        region: str = "US",
    ) -> dict:
        return normalize_filing_rows(await fetch_with_failover(
            "filing_list", "search", region=region,
            query=query, form_type=form_type, date_from=date_from, date_to=date_to,
        ))

    @mcp.tool()
    async def filings_get_content(
        filing_id: str,
        region: str = "US",
    ) -> dict:
        return await fetch_with_failover(
            "filing_content",
            "content",
            region=region,
            filing_id=filing_id,
        )

    @mcp.tool()
    async def filings_get_sec(
        symbol: str,
        filing_type: str | None = None,
        limit: int = 10,
        region: str = "US",
        date_from: str | None = None,
        date_to: str | None = None,
        as_of_date: str | None = None,
    ) -> list[dict]:
        return await fetch_with_failover(
            "sec_filing_list",
            "sec_filings",
            region=region,
            symbol=symbol,
            filing_type=filing_type or "",
            limit=limit,
            date_from=date_from,
            date_to=date_to,
            as_of_date=as_of_date,
        )
