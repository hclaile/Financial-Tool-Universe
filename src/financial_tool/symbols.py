"""Classify public instrument identifiers before provider routing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from .errors import DataValidationError


@dataclass(frozen=True)
class Instrument:
    kind: Literal[
        "equity",
        "index",
        "macro_series",
        "shipping_series",
        "crypto",
        "commodity",
        "forex",
    ]
    provider_symbol: str
    region: str


_MACRO_ALIASES = {
    "USCPI": "CPIAUCSL",
    "CPIAUCSL": "CPIAUCSL",
    "PPIUS": "PPIACO",
    "PPIACO": "PPIACO",
    "HOUST": "HOUST",
    "$HOUST": "HOUST",
}
_BLOOMBERG_INDICES = {
    "$SPX": "^GSPC",
    "$NDX": "^NDX",
    "$SOX": "^SOX",
    "$VIX": "^VIX",
    "$HSI": "^HSI",
    "$DX-Y-NYB": "DX-Y.NYB",
}
_INDEX_ALIASES = {
    "DXY": "DX-Y.NYB",
    "$DXY": "DX-Y.NYB",
    "DX-Y-NYB": "DX-Y.NYB",
    "DX-Y.NYB": "DX-Y.NYB",
    "VIX": "^VIX",
    "SPX": "^GSPC",
    "GSPC": "^GSPC",
    "TNX": "^TNX",
}
_CRYPTO_ALIASES = {
    "BTC": "BTC-USD",
    "BTCUSD": "BTC-USD",
    "ETH": "ETH-USD",
    "ETHUSD": "ETH-USD",
}
_COMMODITY_ALIASES = {
    "XAG": "SI=F",
    "XAG/USD": "SI=F",
    "XAGUSD": "SI=F",
    "SILVER": "SI=F",
    "GOLD": "GC=F",
    "WTI": "CL=F",
    "BRENT": "BZ=F",
}
_SHIPPING_ALIASES = {
    "SCFI": "SCFI",
    "$SCFI": "SCFI",
}
_CANONICAL_INDEXES = {
    "^GSPC",
    "^NDX",
    "^SOX",
    "^VIX",
    "^HSI",
    "^DJI",
    "^RUT",
    "^TNX",
    "^TYX",
    "^FVX",
    "^IRX",
    "^IXIC",
}
_HK_SYMBOL = re.compile(r"^\$?(\d+)\.HK$", re.IGNORECASE)
_CN_VENDOR_PREFIX = re.compile(
    r"^(SZ|XSHE|SH|SSE|XSHG|BJ|BSE)[:.]?(\d{6})$",
    re.IGNORECASE,
)
_CN_VENDOR_SUFFIX = re.compile(r"^(\d{6})\.(XSHE|XSHG|BSE)$", re.IGNORECASE)
_CN_CANONICAL = re.compile(r"^(\d{6})\.(SZ|SH|BJ)$", re.IGNORECASE)
_BARE_CN_CODE = re.compile(r"^\d{6}$")
_UNAMBIGUOUS_CN_PREFIXES = {
    "SH": ("600", "601", "603", "605", "688"),
    "SZ": ("002", "003", "300", "301"),
    "BJ": ("4", "8", "92"),
}
_CN_EXCHANGE_SUFFIX = {
    "SZ": "SZ",
    "XSHE": "SZ",
    "SH": "SH",
    "SSE": "SH",
    "XSHG": "SH",
    "BJ": "BJ",
    "BSE": "BJ",
}


def _alias_key(symbol: str) -> str:
    return re.sub(r"[\s_-]+", "", symbol).upper()


def _cn_instrument(exchange: str, code: str) -> Instrument:
    suffix = _CN_EXCHANGE_SUFFIX[exchange.upper()]
    return Instrument("equity", f"{code}.{suffix}", "CN")


def _hk_instrument(code: str) -> Instrument:
    """Normalize a numeric HK code without guessing its identity.

    A Hong Kong suffix makes short numeric codes unambiguous, so zero-padding
    to the exchange's five-digit form is safe.  A six-digit input is repaired
    only if deleting one leading zero yields that exact five-digit form.
    """
    if 1 <= len(code) <= 5:
        return Instrument("equity", f"{code.zfill(5)}.HK", "HK")
    if len(code) == 6 and code.startswith("0"):
        return Instrument("equity", f"{code[1:]}.HK", "HK")
    raise DataValidationError(
        "symbol",
        "Hong Kong equity symbols must contain one to five digits before .HK, "
        "or six digits beginning with one redundant zero",
    )


def _unambiguous_cn_instrument(code: str) -> Instrument | None:
    """Return an exchange only for stock-code prefixes that cannot be indices."""
    for exchange, prefixes in _UNAMBIGUOUS_CN_PREFIXES.items():
        if code.startswith(prefixes):
            return Instrument("equity", f"{code}.{exchange}", "CN")
    return None


def classify_instrument(symbol: str) -> Instrument:
    """Return a provider-safe representation or reject an ambiguous code."""
    raw = str(symbol or "").strip()
    if not raw:
        raise DataValidationError("symbol", "symbol is required")

    macro_id = _MACRO_ALIASES.get(_alias_key(raw))
    if macro_id:
        return Instrument("macro_series", macro_id, "US")

    shipping_id = _SHIPPING_ALIASES.get(_alias_key(raw))
    if shipping_id:
        return Instrument("shipping_series", shipping_id, "CN")

    crypto_symbol = _CRYPTO_ALIASES.get(_alias_key(raw))
    if crypto_symbol:
        return Instrument("crypto", crypto_symbol, "GLOBAL")

    commodity_symbol = _COMMODITY_ALIASES.get(_alias_key(raw)) or _COMMODITY_ALIASES.get(raw.upper())
    if commodity_symbol:
        return Instrument("commodity", commodity_symbol, "GLOBAL")

    index_symbol = _BLOOMBERG_INDICES.get(raw.upper())
    if index_symbol:
        return Instrument("index", index_symbol, "US")
    index_alias = _INDEX_ALIASES.get(raw.upper())
    if index_alias:
        return Instrument("index", index_alias, "US")
    canonical_index = raw.upper()
    if canonical_index in _CANONICAL_INDEXES:
        return Instrument("index", canonical_index, "US")

    hk_match = _HK_SYMBOL.fullmatch(raw)
    if hk_match:
        return _hk_instrument(hk_match.group(1))

    cn_prefix_match = _CN_VENDOR_PREFIX.fullmatch(raw)
    if cn_prefix_match:
        return _cn_instrument(cn_prefix_match.group(1), cn_prefix_match.group(2))

    cn_suffix_match = _CN_VENDOR_SUFFIX.fullmatch(raw)
    if cn_suffix_match:
        return _cn_instrument(cn_suffix_match.group(2), cn_suffix_match.group(1))

    cn_canonical_match = _CN_CANONICAL.fullmatch(raw)
    if cn_canonical_match:
        return Instrument(
            "equity",
            f"{cn_canonical_match.group(1)}.{cn_canonical_match.group(2).upper()}",
            "CN",
        )

    if _BARE_CN_CODE.fullmatch(raw):
        instrument = _unambiguous_cn_instrument(raw)
        if instrument:
            return instrument
        raise DataValidationError(
            "symbol",
            "bare six-digit Chinese code is ambiguous; use an exchange-qualified "
            "symbol such as 000001.SZ or 600519.SH",
        )
    if raw.startswith("$"):
        raise DataValidationError(
            "symbol",
            "unknown vendor $ symbol; resolve it to an exchange-qualified ticker",
        )

    return Instrument("equity", raw.upper(), "")
