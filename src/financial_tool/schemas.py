"""Pydantic schemas for tool inputs and outputs."""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── Market Data ──────────────────────────────────────────────

class QuoteInput(BaseModel):
    symbol: str
    exchange: str | None = None


class QuoteOutput(BaseModel):
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    change: float
    change_pct: float
    timestamp: str


class OHLCVInput(BaseModel):
    symbol: str
    start_date: str | None = None
    end_date: str | None = None
    interval: str = "1d"


class OHLCVOutput(BaseModel):
    symbol: str
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int


# ── Fundamentals ─────────────────────────────────────────────

class StatementInput(BaseModel):
    symbol: str
    period: str = "annual"


class IncomeStatement(BaseModel):
    symbol: str
    date: str
    revenue: float | None = None
    cogs: float | None = None
    gross_profit: float | None = None
    net_income: float | None = None


# ── Filings ──────────────────────────────────────────────────

class FilingSearchInput(BaseModel):
    query: str
    form_type: str | None = None
    date_from: str | None = None
    date_to: str | None = None


class FilingContentInput(BaseModel):
    filing_id: str


# ── Macro ────────────────────────────────────────────────────

class MacroSeriesInput(BaseModel):
    series_id: str
    start_date: str | None = None
    end_date: str | None = None


# ── News ─────────────────────────────────────────────────────

class NewsHeadlinesInput(BaseModel):
    query: str | None = None
    symbols: list[str] | None = None
    limit: int = Field(default=20, ge=1, le=100)


class AlpacaNewsArticlesInput(BaseModel):
    symbols: list[str] | None = None
    start: str | None = None
    end: str | None = None
    sort: str = Field(default="desc", pattern="^(asc|desc)$")
    limit: int = Field(default=10, ge=1, le=50)
    include_content: bool = False
    exclude_contentless: bool = False
    page_token: str | None = None


# ── Crypto ───────────────────────────────────────────────────

class CryptoPriceInput(BaseModel):
    symbol: str
    vs_currency: str = "USDT"


# ── On-chain ─────────────────────────────────────────────────

class OnchainBalanceInput(BaseModel):
    address: str
    chain: str = Field(pattern="^(eth|btc|sol)$")


# ── Risk ─────────────────────────────────────────────────────

class VaRInput(BaseModel):
    portfolio: list[str]
    confidence: float = Field(default=0.95, ge=0.9, le=0.99)
    horizon_days: int = Field(default=1, ge=1)


# ── Search ───────────────────────────────────────────────────

class SearchInput(BaseModel):
    query: str
    data_types: list[str] | None = None
    limit: int = Field(default=10, ge=1, le=50)
