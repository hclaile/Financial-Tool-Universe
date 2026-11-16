"""Custom error types."""

from __future__ import annotations


class FinancialToolError(Exception):
    """Base error for the financial tool service."""


class ToolNotFoundError(FinancialToolError):
    def __init__(self, name: str) -> None:
        super().__init__(f"Tool not found: {name}")


class SourceUnavailableError(FinancialToolError):
    def __init__(
        self,
        source: str,
        reason: str = "",
        attempts: list[dict[str, str]] | None = None,
    ) -> None:
        self.source = source
        self.reason = reason
        self.attempts = list(attempts or [])
        msg = f"Source unavailable: {source}"
        if reason:
            msg += f" — {reason}"
        super().__init__(msg)


class UnsupportedRegionError(FinancialToolError):
    def __init__(self, data_type: str, region: str) -> None:
        self.data_type = data_type
        self.region = region
        super().__init__(
            f"Unsupported region '{region}' for data type '{data_type}'"
        )


class RateLimitExceededError(FinancialToolError):
    def __init__(self, tool: str, limit: str) -> None:
        super().__init__(f"Rate limit exceeded for {tool}: {limit}")


class PermissionDeniedError(FinancialToolError):
    def __init__(self, role: str, tool: str) -> None:
        super().__init__(f"Role '{role}' cannot access tool '{tool}'")


class DataValidationError(FinancialToolError):
    def __init__(self, field: str, detail: str) -> None:
        super().__init__(f"Validation error on '{field}': {detail}")


class CacheError(FinancialToolError):
    pass
