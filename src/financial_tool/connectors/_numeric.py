"""Numeric coercion helpers for external API payloads."""

from __future__ import annotations

import math
from typing import Any


def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert API numeric fields without letting malformed payloads mask source errors."""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        try:
            return default if math.isnan(float(value)) else float(value)
        except (TypeError, ValueError):
            return default
    if isinstance(value, str):
        text = value.strip()
        if not text or text in {"-", "--", ".", "N/A", "n/a", "NaN", "nan", "None", "null"}:
            return default
        text = text.replace(",", "").replace("%", "")
        try:
            result = float(text)
            return default if math.isnan(result) else result
        except (TypeError, ValueError):
            return default
    try:
        result = float(value)
        return default if math.isnan(result) else result
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    return int(safe_float(value, float(default)))
