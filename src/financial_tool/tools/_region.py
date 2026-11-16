"""Small guards for explicitly sourced regional tools."""

from __future__ import annotations

from ..errors import UnsupportedRegionError


def require_region(
    region: str,
    *,
    allowed: set[str],
    data_type: str,
) -> str:
    normalized = str(region).strip().upper()
    if normalized not in allowed:
        raise UnsupportedRegionError(data_type, normalized)
    return normalized
